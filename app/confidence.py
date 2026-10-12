"""Per-field confidence levels and human-review flags."""

from app.extractor import ExtractionOutcome
from app.schemas import FieldResult, ImageEstimate, ProductSpec, ReviewFlag
from app.vision import CVResult

# Plausible ranges. Anything outside is kept but flagged and forced to low confidence.
SIZE_RANGE_MM = {  # ring: inner diameter (US 3..15); others: height or length
    "ring": (14.0, 24.0),
    "pendant": (5.0, 60.0),
    "earrings": (3.0, 70.0),
    "bracelet": (140.0, 240.0),
    "necklace": (300.0, 1000.0),
}
STONE_COUNT_MAX = 150

# Counts "agree" when they differ by at most this much (absolute OR relative, whichever is looser).
AGREE_ABS, AGREE_REL = 1, 0.10

SOURCE_CONFIDENCE = {"stated": "high", "inferred": "medium", "missing": "low"}
SOURCE_REASON = {
    "stated": "explicitly stated in the brief",
    "inferred": "inferred by the model (not written in the brief)",
    "missing": "not mentioned in the brief",
}
FIELD_TO_SOURCE_KEY = {"size_mm": "size"}


def counts_agree(a: int, b: int) -> bool:
    return abs(a - b) <= max(AGREE_ABS, AGREE_REL * max(a, b))


def image_estimate(cv: CVResult | None, vlm_count: int | None, image_status: str) -> ImageEstimate | None:
    """Combine the classical CV count with the vision-LLM count into one image-side estimate."""
    if image_status == "none":
        return None
    if cv is None:
        return ImageEstimate(cv_count=None, cv_confidence=None, cv_details=None, vlm_count=vlm_count, final_count=vlm_count,
                             confidence="low", reason=f"image {image_status}; only the vision-LLM count is available")
    if vlm_count is None:
        return ImageEstimate(cv_count=cv.count, cv_confidence=cv.confidence, cv_details=cv.details(), vlm_count=None,
                             final_count=cv.count, confidence=cv.confidence, reason=f"CV only: {cv.reason}")
    if counts_agree(cv.count, vlm_count):
        conf = "high" if cv.confidence != "low" else "medium"
        reason = f"CV ({cv.count}) and vision-LLM ({vlm_count}) agree; {cv.reason}"
        final = cv.count
    else:
        conf = "low"
        reason = f"CV ({cv.count}) and vision-LLM ({vlm_count}) disagree; {cv.reason}"
        # Prefer the CV count when its own two methods agree, otherwise trust neither more than the other.
        final = cv.count if cv.confidence == "high" else vlm_count
    return ImageEstimate(cv_count=cv.count, cv_confidence=cv.confidence, cv_details=cv.details(), vlm_count=vlm_count,
                         final_count=final, confidence=conf, reason=reason)


def score_fields(spec: ProductSpec, outcome: ExtractionOutcome, img: ImageEstimate | None
                 ) -> tuple[dict[str, FieldResult], ProductSpec, list[ReviewFlag]]:
    """Returns (per-field results, the spec actually used for matching, flags)."""
    flags: list[ReviewFlag] = []
    results: dict[str, FieldResult] = {}
    used = spec.model_copy()
    sources = outcome.extraction.sources

    if outcome.mode == "degraded":
        flags.append(ReviewFlag(code="extraction_degraded", message="LLM output unusable; spec is empty: " + " | ".join(outcome.steps[-2:])))

    for f in ("product_type", "material", "stone_type", "size_mm", "setting_style"):
        value = getattr(spec, f)
        src = getattr(sources, FIELD_TO_SOURCE_KEY.get(f, f))
        conf, reason = SOURCE_CONFIDENCE[src], SOURCE_REASON[src]
        if outcome.mode == "degraded":
            conf, reason = "low", "extraction degraded to safe default"
        elif f in outcome.repaired_fields and conf == "high":
            conf, reason = "medium", reason + "; value was repaired from off-schema model output"
        results[f] = FieldResult(value=value, confidence=conf, reason=reason)

    # size range check (needs product type to know what "size" means)
    size = spec.size_mm
    if size is not None:
        lo, hi = SIZE_RANGE_MM.get(spec.product_type or "", (1.0, 1000.0))
        if not lo <= size <= hi:
            results["size_mm"] = FieldResult(value=size, confidence="low",
                                             reason=f"{size} mm outside plausible range {lo}-{hi} mm for {spec.product_type or 'unknown type'}")
            flags.append(ReviewFlag(code="out_of_range", field="size_mm", message=results["size_mm"].reason))
            used.size_mm = None  # don't let an implausible size steer the matcher

    # stone count: brief vs image
    results["stone_count"], count_flags, used.stone_count = _stone_count(spec, outcome, img)
    flags += count_flags

    if spec.stone_type == "none" and (spec.stone_count or 0) > 0:
        flags.append(ReviewFlag(code="inconsistent_spec", field="stone_count",
                                message=f"stone_type is 'none' but stone_count is {spec.stone_count}"))

    for f, r in results.items():
        if r.confidence == "low" and r.value is None and f in ("product_type", "material"):
            flags.append(ReviewFlag(code="missing_key_field", field=f, message=f"{f} is missing; matching is less reliable"))
    return results, used, flags


def _stone_count(spec: ProductSpec, outcome: ExtractionOutcome, img: ImageEstimate | None):
    flags: list[ReviewFlag] = []
    brief = spec.stone_count
    src = outcome.extraction.sources.stone_count
    image = img.final_count if img else None

    if brief is not None and brief > STONE_COUNT_MAX:
        flags.append(ReviewFlag(code="out_of_range", field="stone_count", message=f"{brief} stones exceeds plausible max {STONE_COUNT_MAX}"))
        return FieldResult(value=brief, confidence="low", reason="implausibly high stone count"), flags, None

    if brief is not None and image is not None:
        if counts_agree(brief, image):
            conf = "high" if src == "stated" else "medium"
            return FieldResult(value=brief, confidence=conf, reason=f"brief ({brief}) and image ({image}) agree"), flags, brief
        flags.append(ReviewFlag(code="source_disagreement", field="stone_count",
                                message=f"brief asks for {brief} stones but the reference image shows ~{image}"))
        # The brief is the customer's request; the photo is only a reference. Keep the brief value.
        return FieldResult(value=brief, confidence="low",
                           reason=f"using brief value {brief}; image estimate {image} disagrees"), flags, brief

    if brief is not None:
        conf = SOURCE_CONFIDENCE[src] if outcome.mode != "degraded" else "low"
        return FieldResult(value=brief, confidence=conf, reason=SOURCE_REASON[src]), flags, brief

    if image is not None:
        conf = "medium" if img.confidence == "high" else "low"
        result = FieldResult(value=image, confidence=conf, reason=f"not in brief; taken from image ({img.reason})")
        if conf == "low":
            flags.append(ReviewFlag(code="low_confidence", field="stone_count", message=f"count taken from image only: {img.reason}"))
            return result, flags, None  # reported, but a shaky guess must not steer the matcher
        return result, flags, image

    return FieldResult(value=None, confidence="low", reason="no count in brief and no usable image"), flags, None
