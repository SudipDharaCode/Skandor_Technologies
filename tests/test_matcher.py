import random

from app.matcher import match
from app.schemas import ProductSpec


def test_exact_spec_auto_selects_correct_item(inventory):
    spec = ProductSpec(product_type="ring", material="white_gold", stone_type="diamond",
                       stone_count=12, size_mm=17.3, setting_style="halo")
    r = match(spec, inventory)
    assert r.auto_selected
    assert r.selected_id == "R003"
    assert r.candidates[0].score == 1.0
    assert r.confidence == "high"


def test_same_input_same_ranking_even_if_inventory_order_changes(inventory):
    spec = ProductSpec(product_type="ring", material="yellow_gold", stone_type="diamond", stone_count=10)
    baseline = [c.id for c in match(spec, inventory).candidates]
    for seed in range(5):
        shuffled = inventory[:]
        random.Random(seed).shuffle(shuffled)
        assert [c.id for c in match(spec, shuffled).candidates] == baseline


def test_ambiguous_spec_is_not_auto_selected(inventory):
    # Without material, the white- and yellow-gold 12-stone halo rings tie.
    spec = ProductSpec(product_type="ring", stone_type="diamond", stone_count=12, setting_style="halo")
    r = match(spec, inventory)
    assert not r.auto_selected
    assert {c.id for c in r.candidates[:2]} == {"R003", "R004"}
    assert "ambiguous" in r.reason


def test_low_coverage_spec_is_not_auto_selected(inventory):
    r = match(ProductSpec(setting_style="halo"), inventory)
    assert not r.auto_selected
    assert r.coverage < 0.6


def test_empty_spec_returns_no_candidates(inventory):
    r = match(ProductSpec(), inventory)
    assert r.candidates == [] and r.selected_id is None and r.confidence == "low"


def test_partial_credit_prefers_closer_stone_count(inventory):
    spec = ProductSpec(product_type="ring", material="white_gold", stone_type="diamond", stone_count=15, setting_style="halo")
    ids = [c.id for c in match(spec, inventory).candidates]
    assert ids.index("R005") < ids.index("R003")  # 16 is closer to 15 than 12


def test_hard_field_mismatch_blocks_auto_select(inventory):
    # No platinum riviera necklace exists; the white-gold one ranks first but must not be auto-quoted.
    spec = ProductSpec(product_type="necklace", material="platinum", stone_type="diamond", size_mm=420)
    r = match(spec, inventory)
    assert r.candidates[0].id == "N001"
    assert not r.auto_selected
    assert "material" in r.reason
