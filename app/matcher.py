"""Deterministic, weighted spec -> inventory matcher. No LLM, no randomness.

score(item) = sum(w_f * sim_f) / sum(w_f)   over the fields the spec actually knows.
coverage    = sum(w_f for known fields) / sum(all w_f)

An item is auto-selected only if: score >= AUTO_SELECT_MIN_SCORE, coverage >= MIN_COVERAGE,
it beats the runner-up by at least MIN_MARGIN, and every HARD_FIELD the spec specifies matches
exactly (a white-gold piece is never auto-quoted for a platinum request, however close the rest is).
Otherwise we still return the ranked list but leave the decision to a human.
"""

import csv
from pathlib import Path

from app.schemas import InventoryItem, MatchCandidate, MatchResult, ProductSpec

WEIGHTS: dict[str, float] = {
    "product_type": 0.30,
    "material": 0.25,
    "stone_type": 0.15,
    "stone_count": 0.15,
    "size_mm": 0.10,
    "setting_style": 0.05,
}

AUTO_SELECT_MIN_SCORE = 0.80
MIN_COVERAGE = 0.60
MIN_MARGIN = 0.02
TOP_K = 5
HARD_FIELDS = ("product_type", "material", "stone_type")

# Partial credit for "close but not equal" categorical values (symmetric).
MATERIAL_SIM = {
    frozenset({"yellow_gold", "rose_gold"}): 0.5,
    frozenset({"yellow_gold", "white_gold"}): 0.4,
    frozenset({"white_gold", "rose_gold"}): 0.4,
    frozenset({"white_gold", "platinum"}): 0.4,
    frozenset({"white_gold", "silver"}): 0.2,
    frozenset({"silver", "platinum"}): 0.2,
}
STONE_SIM = {
    frozenset({"diamond", "cubic_zirconia"}): 0.3,
}
STYLE_SIM = {
    frozenset({"pave", "channel"}): 0.4,
    frozenset({"halo", "cluster"}): 0.3,
}
# Size difference (mm) at which size similarity reaches 0, per product type.
SIZE_TOLERANCE_MM = {"ring": 1.5, "pendant": 8.0, "earrings": 8.0, "bracelet": 25.0, "necklace": 60.0}


def load_inventory(path: Path) -> list[InventoryItem]:
    with open(path, newline="", encoding="utf-8") as f:
        items = [InventoryItem.model_validate(row) for row in csv.DictReader(f)]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("inventory contains duplicate ids")
    return items


def _categorical(a: str, b: str, table: dict) -> float:
    if a == b:
        return 1.0
    return table.get(frozenset({a, b}), 0.0)


def _count_sim(a: int, b: int) -> float:
    if a == b:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / max(a, b))


def _size_sim(a: float, b: float, product_type: str) -> float:
    tol = SIZE_TOLERANCE_MM.get(product_type, 10.0)
    return max(0.0, 1.0 - abs(a - b) / tol)


def field_similarities(spec: ProductSpec, item: InventoryItem) -> dict[str, float]:
    """Similarity in [0,1] for every field the spec knows. Unknown fields are omitted."""
    sims: dict[str, float] = {}
    if spec.product_type is not None:
        sims["product_type"] = 1.0 if spec.product_type == item.product_type else 0.0
    if spec.material is not None:
        sims["material"] = _categorical(spec.material, item.material, MATERIAL_SIM)
    if spec.stone_type is not None:
        sims["stone_type"] = _categorical(spec.stone_type, item.stone_type, STONE_SIM)
    if spec.stone_count is not None:
        sims["stone_count"] = _count_sim(spec.stone_count, item.stone_count)
    if spec.size_mm is not None:
        sims["size_mm"] = _size_sim(spec.size_mm, item.size_mm, item.product_type)
    if spec.setting_style is not None:
        sims["setting_style"] = _categorical(spec.setting_style, item.setting_style, STYLE_SIM)
    return sims


def match(spec: ProductSpec, inventory: list[InventoryItem], top_k: int = TOP_K) -> MatchResult:
    known_weight = sum(WEIGHTS[f] for f in WEIGHTS if getattr(spec, f) is not None)
    coverage = round(known_weight / sum(WEIGHTS.values()), 4)

    if known_weight == 0:
        return MatchResult(
            selected_id=None, auto_selected=False, coverage=0.0, candidates=[],
            confidence="low", reason="spec has no usable fields; nothing to match on",
        )

    scored = []
    for item in inventory:
        sims = field_similarities(spec, item)
        score = sum(WEIGHTS[f] * s for f, s in sims.items()) / known_weight
        scored.append((round(score, 4), item, {f: round(s, 3) for f, s in sims.items()}))

    # Tie-break on id so ordering is fully deterministic regardless of file order.
    scored.sort(key=lambda t: (-t[0], t[1].id))
    candidates = [MatchCandidate(id=i.id, name=i.name, score=s, breakdown=b) for s, i, b in scored[:top_k]]

    top = scored[0][0]
    margin = top - scored[1][0] if len(scored) > 1 else top
    reasons = []
    if top < AUTO_SELECT_MIN_SCORE:
        reasons.append(f"best score {top:.2f} < {AUTO_SELECT_MIN_SCORE}")
    if coverage < MIN_COVERAGE:
        reasons.append(f"spec coverage {coverage:.2f} < {MIN_COVERAGE}")
    if margin < MIN_MARGIN:
        reasons.append(f"top two candidates within {margin:.3f} (ambiguous)")
    mismatched = [f for f in HARD_FIELDS if scored[0][2].get(f, 1.0) < 1.0]
    if mismatched:
        reasons.append(f"best item differs on {', '.join(mismatched)}")

    auto = not reasons
    if auto:
        conf = "high" if top >= 0.95 and coverage >= 0.85 else "medium"
        reason = f"score {top:.2f}, coverage {coverage:.2f}, margin {margin:.2f}"
    else:
        conf = "low"
        reason = "not auto-selected: " + "; ".join(reasons)

    return MatchResult(
        selected_id=candidates[0].id if auto else None,
        auto_selected=auto,
        coverage=coverage,
        candidates=candidates,
        confidence=conf,
        reason=reason,
    )
