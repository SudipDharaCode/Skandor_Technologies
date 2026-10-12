"""LLM extraction with a recovery ladder:

    1. clean     - model output parses and passes strict schema validation
    2. repaired  - output was JSON-ish but off-schema; deterministic repair fixed it (no extra call)
    3. retried   - output was unusable; we re-asked once with the validation error attached
    4. degraded  - nothing usable (or every model errored); return an all-null spec, everything low confidence

The mode and every step taken are returned to the caller so the API response can say what happened.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.llm_client import LLMClient, LLMResult
from app.schemas import LLMExtraction, ProductSpec

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You extract a structured jewelry product spec from a sales rep's brief.
Return ONLY one JSON object, no prose, no markdown, with exactly these keys:

{{
  "product_type":  one of ["ring","pendant","earrings","bracelet","necklace"] or null,
  "material":      one of ["yellow_gold","white_gold","rose_gold","silver","platinum"] or null,
  "stone_type":    one of ["diamond","sapphire","ruby","emerald","cubic_zirconia","none"] or null,
  "stone_count":   integer total number of stones requested in the brief, or null,
  "size_value":    number as written in the brief, or null,
  "size_unit":     one of ["mm","cm","inch","us_ring_size"] or null,
  "setting_style": one of ["solitaire","halo","pave","channel","cluster","plain"] or null,
  "sources": {{"product_type": S, "material": S, "stone_type": S, "stone_count": S, "size": S, "setting_style": S}},
  "image_stone_count": integer or null
}}
where S is "stated" (explicitly written in the brief), "inferred" (implied by the brief or taken from the image) or "missing".

Rules:
- Use null when the brief does not give a value. Never invent numbers.
- Do NOT convert units; copy the size number and give its unit. "size 7" for a ring means us_ring_size 7.
  For rings, size means inner diameter; for bracelets and necklaces it means length.
- Plain "gold" means yellow_gold (mark "inferred"). "Sterling" means silver. "CZ" means cubic_zirconia.
- Stone count must be the TOTAL number of stones written in the brief (e.g. "a center stone with 12 around it" = 13).
- If no stones are wanted, stone_type is "none" and stone_count is 0.
- {image_rule}

Brief:
\"\"\"{brief}\"\"\"
"""

IMAGE_RULE_WITH = (
    "An image is attached. Set image_stone_count to the number of individual gemstones you can see in the image, "
    "counted independently of the brief. You may use the image to fill a field the brief leaves out; mark it \"inferred\"."
)
IMAGE_RULE_WITHOUT = "No image is attached. image_stone_count must be null."

RETRY_SUFFIX = """
Your previous reply could not be used: {error}
Reply again with ONLY the JSON object described above."""


@dataclass
class ExtractionOutcome:
    extraction: LLMExtraction
    mode: str
    steps: list[str] = field(default_factory=list)
    model_used: str | None = None
    repaired_fields: list[str] = field(default_factory=list)
    calls: list[LLMResult] = field(default_factory=list)


# JSON parsing
def parse_json_loose(text: str | None) -> dict | None:
    """Best-effort: strip fences/prose, fix trailing commas and quote styles. Returns a dict or None."""
    if text is None or not text.strip():
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    candidates = [t]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        candidates.append(t[start:end + 1])

    for c in candidates:
        for variant in (c, _fix_common_json_errors(c)):
            try:
                obj = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                obj = _python_literal(variant)
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]
            if isinstance(obj, dict):
                return obj
    return None


def _fix_common_json_errors(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    return s


def _python_literal(s: str):
    try:
        py = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", s)))
        return ast.literal_eval(py)
    except Exception:
        return None


# payload repair
KEY_ALIASES = {
    "type": "product_type", "product": "product_type", "category": "product_type",
    "metal": "material",
    "stone": "stone_type", "gem": "stone_type", "gemstone": "stone_type", "stones_type": "stone_type",
    "stones": "stone_count", "num_stones": "stone_count", "count": "stone_count", "number_of_stones": "stone_count",
    "size": "size_value", "size_mm": "size_value",
    "unit": "size_unit",
    "setting": "setting_style", "style": "setting_style",
}

ENUM_SYNONYMS = {
    "product_type": {
        "ring": "ring", "band": "ring", "engagement_ring": "ring", "wedding_band": "ring",
        "pendant": "pendant", "charm": "pendant",
        "earrings": "earrings", "earring": "earrings", "studs": "earrings", "stud_earrings": "earrings", "hoops": "earrings",
        "bracelet": "bracelet", "bangle": "bracelet", "tennis_bracelet": "bracelet",
        "necklace": "necklace", "chain": "necklace", "choker": "necklace",
    },
    "material": {
        "yellow_gold": "yellow_gold", "gold": "yellow_gold", "18k_gold": "yellow_gold", "14k_gold": "yellow_gold",
        "22k_gold": "yellow_gold", "18k_yellow_gold": "yellow_gold", "14k_yellow_gold": "yellow_gold",
        "white_gold": "white_gold", "18k_white_gold": "white_gold", "14k_white_gold": "white_gold",
        "rose_gold": "rose_gold", "pink_gold": "rose_gold", "18k_rose_gold": "rose_gold",
        "silver": "silver", "sterling_silver": "silver", "sterling": "silver", "925_silver": "silver",
        "platinum": "platinum", "pt950": "platinum",
    },
    "stone_type": {
        "diamond": "diamond", "diamonds": "diamond",
        "sapphire": "sapphire", "sapphires": "sapphire", "blue_sapphire": "sapphire",
        "ruby": "ruby", "rubies": "ruby",
        "emerald": "emerald", "emeralds": "emerald",
        "cubic_zirconia": "cubic_zirconia", "cz": "cubic_zirconia", "zirconia": "cubic_zirconia",
        "none": "none", "no_stones": "none", "no_stone": "none", "plain": "none",
    },
    "setting_style": {
        "solitaire": "solitaire", "halo": "halo", "cluster": "cluster", "plain": "plain", "none": "plain",
        "pave": "pave", "pav\u00e9": "pave", "micro_pave": "pave",
        "channel": "channel", "channel_set": "channel", "eternity": "channel",
    },
    "size_unit": {
        "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
        "cm": "cm", "centimeter": "cm", "centimeters": "cm",
        "inch": "inch", "inches": "inch", "in": "inch", '"': "inch",
        "us_ring_size": "us_ring_size", "us": "us_ring_size", "ring_size": "us_ring_size", "us_size": "us_ring_size",
    },
}
SOURCE_VALUES = {"stated", "inferred", "missing"}
SOURCE_KEYS = ("product_type", "material", "stone_type", "stone_count", "size", "setting_style")
SOURCE_TO_VALUE_KEY = {"size": "size_value"}


def _norm_token(v) -> str:
    return re.sub(r"[\s\-]+", "_", str(v).strip().lower())


def _to_int(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(round(v)) if v >= 0 else None
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _to_float(v) -> tuple[float | None, str | None]:
    """Returns (number, unit-hint-from-string)."""
    if v is None or isinstance(v, bool):
        return None, None
    if isinstance(v, (int, float)):
        return (float(v) if v > 0 else None), None
    s = str(v).lower()
    m = re.search(r"\d+(?:\.\d+)?", s)
    num = float(m.group()) if m else None
    unit = None
    for token, u in (("mm", "mm"), ("cm", "cm"), ("inch", "inch"), ('"', "inch"), ("size", "us_ring_size")):
        if token in s:
            unit = u
            break
    return (num if num and num > 0 else None), unit


def repair_payload(data: dict) -> tuple[dict, list[str]]:
    """Coerce an off-schema dict into the LLMExtraction shape. Returns (payload, fields that were changed)."""
    changed: list[str] = []
    src = {}
    for k, v in data.items():
        key = _norm_token(k)
        mapped = KEY_ALIASES.get(key, key)
        if mapped != key:
            changed.append(f"{k}->{mapped}")
        src.setdefault(mapped, v)

    out: dict = {}
    for f in ("product_type", "material", "stone_type", "setting_style", "size_unit"):
        raw = src.get(f)
        if raw is None:
            out[f] = None
            continue
        val = ENUM_SYNONYMS[f].get(_norm_token(raw))
        if val != raw:
            changed.append(f)
        out[f] = val

    for f in ("stone_count", "image_stone_count"):
        raw = src.get(f)
        val = _to_int(raw)
        if val != raw:
            changed.append(f)
        out[f] = val

    size, unit_hint = _to_float(src.get("size_value"))
    if size != src.get("size_value"):
        changed.append("size_value")
    out["size_value"] = size
    if out["size_unit"] is None and unit_hint and size is not None:
        out["size_unit"] = unit_hint
        changed.append("size_unit")

    raw_sources = src.get("sources") if isinstance(src.get("sources"), dict) else {}
    out["sources"] = {}
    for k in SOURCE_KEYS:
        s = _norm_token(raw_sources.get(k, "")) if raw_sources.get(k) is not None else ""
        if s not in SOURCE_VALUES:
            value_present = out.get(SOURCE_TO_VALUE_KEY.get(k, k)) is not None
            s = "inferred" if value_present else "missing"
            changed.append(f"sources.{k}")
        out["sources"][k] = s
    return out, sorted(set(changed))


def reconcile_sources(ex: LLMExtraction) -> list[str]:
    """Make sources consistent with values (a null value cannot be 'stated'). Mutates ex in place."""
    fixed = []
    for k in SOURCE_KEYS:
        value = getattr(ex, SOURCE_TO_VALUE_KEY.get(k, k))
        src = getattr(ex.sources, k)
        if value is None and src != "missing":
            setattr(ex.sources, k, "missing")
            fixed.append(k)
        elif value is not None and src == "missing":
            setattr(ex.sources, k, "inferred")
            fixed.append(k)
    return fixed


# main entry
def build_prompt(brief: str, has_image: bool) -> str:
    return PROMPT_TEMPLATE.format(brief=brief.strip() or "(empty)", image_rule=IMAGE_RULE_WITH if has_image else IMAGE_RULE_WITHOUT)


def extract(brief: str, client: LLMClient | None, image: bytes | None = None,
            image_mime: str | None = None, max_attempts: int = 2) -> ExtractionOutcome:
    steps: list[str] = []
    calls: list[LLMResult] = []

    if client is None:
        steps.append("no LLM client configured")
        return ExtractionOutcome(LLMExtraction(), "degraded", steps)

    base_prompt = build_prompt(brief, image is not None)
    prompt = base_prompt
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.generate(prompt, image, image_mime)
        except Exception as exc:
            steps.append(f"attempt {attempt}: LLM call failed ({type(exc).__name__}: {str(exc)[:200]})")
            break
        calls.append(res)

        data = parse_json_loose(res.text)
        if data is None:
            preview = (res.text or "").strip()[:80]
            steps.append(f"attempt {attempt}: output not parseable as JSON ({preview!r})")
            prompt = base_prompt + RETRY_SUFFIX.format(error="it was not a valid JSON object")
            continue

        try:
            ex = LLMExtraction.model_validate(data)
            fixed = reconcile_sources(ex)
            if fixed:
                steps.append(f"attempt {attempt}: reconciled sources for {fixed}")
            steps.append(f"attempt {attempt}: valid output from {res.model}")
            return ExtractionOutcome(ex, "clean" if attempt == 1 else "retried", steps, res.model, [], calls)
        except ValidationError as ve:
            steps.append(f"attempt {attempt}: schema validation failed ({ve.error_count()} errors)")
            repaired, changed = repair_payload(data)
            try:
                ex = LLMExtraction.model_validate(repaired)
            except ValidationError as ve2:
                steps.append(f"attempt {attempt}: repair failed")
                prompt = base_prompt + RETRY_SUFFIX.format(error=_short_errors(ve2))
                continue
            reconcile_sources(ex)
            steps.append(f"attempt {attempt}: repaired fields {changed}")
            return ExtractionOutcome(ex, "repaired" if attempt == 1 else "retried", steps, res.model, changed, calls)

    steps.append("degraded to empty spec")
    return ExtractionOutcome(LLMExtraction(), "degraded", steps, calls[-1].model if calls else None, [], calls)


def _short_errors(ve: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in ve.errors()[:5])


# normalization
def size_to_mm(value: float | None, unit: str | None) -> float | None:
    """Convert a size in the given unit to millimeters."""
    if value is None:
        return None
    if unit is None or unit == "mm":
        return round(value, 2)
    if unit == "cm":
        return round(value * 10, 2)
    if unit == "inch":
        return round(value * 25.4, 2)
    if unit == "us_ring_size":
        return round(11.63 + 0.8128 * value, 2)  # US ring size -> inner diameter in mm
    return None


def to_spec(ex: LLMExtraction) -> ProductSpec:
    return ProductSpec(
        product_type=ex.product_type,
        material=ex.material,
        stone_type=ex.stone_type,
        stone_count=ex.stone_count,
        size_mm=size_to_mm(ex.size_value, ex.size_unit),
        setting_style=ex.setting_style,
    )
