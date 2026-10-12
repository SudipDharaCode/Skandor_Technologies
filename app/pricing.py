"""Transparent price estimate: matched item base price, adjusted for a different stone count."""

from app.schemas import Confidence, InventoryItem, MatchResult, PriceEstimate, ProductSpec

CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


def _min_conf(*levels: Confidence) -> Confidence:
    return min(levels, key=lambda c: CONF_ORDER[c])


def estimate_price(match: MatchResult, spec: ProductSpec, count_conf: Confidence,
                   inventory: dict[str, InventoryItem]) -> PriceEstimate:
    if not match.auto_selected or match.selected_id is None:
        return PriceEstimate(value_usd=None, confidence="low", reason="no inventory item auto-selected; price needs a human")

    item = inventory[match.selected_id]
    price = item.base_price_usd
    parts = [f"base {item.id} ${item.base_price_usd:,.0f}"]
    conf = match.confidence

    wanted = spec.stone_count
    if wanted is not None and wanted != item.stone_count and item.stone_unit_price_usd > 0:
        delta = (wanted - item.stone_count) * item.stone_unit_price_usd
        price += delta
        parts.append(f"{wanted - item.stone_count:+d} stones x ${item.stone_unit_price_usd:,.0f} = {delta:+,.0f}")
        conf = _min_conf(conf, count_conf)

    return PriceEstimate(value_usd=round(max(price, 0.0), 2), confidence=conf, reason="; ".join(parts))
