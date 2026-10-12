import json

import pytest

from app.extractor import extract, parse_json_loose, size_to_mm, to_spec
from app.llm_client import LLMError
from tests.conftest import FakeLLM

GOOD = {
    "product_type": "ring", "material": "white_gold", "stone_type": "diamond", "stone_count": 12,
    "size_value": 7, "size_unit": "us_ring_size", "setting_style": "halo",
    "sources": {"product_type": "stated", "material": "stated", "stone_type": "stated",
                "stone_count": "stated", "size": "stated", "setting_style": "stated"},
    "image_stone_count": None,
}


def test_clean_output():
    out = extract("white gold halo ring, 12 diamonds, size 7", FakeLLM(json.dumps(GOOD)))
    assert out.mode == "clean"
    assert out.extraction.stone_count == 12
    assert to_spec(out.extraction).size_mm == pytest.approx(17.32, abs=0.01)


def test_markdown_fenced_json_with_trailing_comma_is_parsed():
    text = "Sure! Here you go:\n```json\n" + json.dumps(GOOD)[:-1] + ",}\n```"
    out = extract("brief", FakeLLM(text))
    assert out.mode == "clean"
    assert out.extraction.material == "white_gold"


def test_off_schema_values_are_repaired_without_extra_call():
    bad = {"type": "Engagement Ring", "metal": "18k White Gold", "stone": "Diamonds",
           "stones": "about 12", "size": "17 mm", "setting": "Pav\u00e9"}
    llm = FakeLLM(json.dumps(bad))
    out = extract("brief", llm)
    assert out.mode == "repaired"
    assert len(llm.prompts) == 1
    ex = out.extraction
    assert (ex.product_type, ex.material, ex.stone_type, ex.stone_count, ex.size_value, ex.size_unit, ex.setting_style) == \
        ("ring", "white_gold", "diamond", 12, 17.0, "mm", "pave")
    assert ex.sources.material == "inferred"  # source was missing, so it cannot be "stated"


@pytest.mark.parametrize("garbage", ["", "   ", "I cannot help with that.", "{not json at all", None])
def test_garbage_then_valid_triggers_retry(garbage):
    llm = FakeLLM(garbage, json.dumps(GOOD))
    out = extract("brief", llm)
    assert out.mode == "retried"
    assert len(llm.prompts) == 2
    assert "previous reply could not be used" in llm.prompts[1]


def test_garbage_twice_degrades_to_safe_default():
    out = extract("brief", FakeLLM("nope", "[1, 2, 3]"))
    assert out.mode == "degraded"
    assert out.extraction.product_type is None and out.extraction.stone_count is None
    assert any("degraded" in s for s in out.steps)


def test_llm_exception_degrades_without_crashing():
    out = extract("brief", FakeLLM(LLMError("quota exceeded")))
    assert out.mode == "degraded"
    assert any("quota exceeded" in s for s in out.steps)


def test_no_client_degrades():
    assert extract("brief", None).mode == "degraded"


def test_negative_count_is_rejected_then_repaired_to_null():
    bad = dict(GOOD, stone_count=-3)
    out = extract("brief", FakeLLM(json.dumps(bad)))
    assert out.mode == "repaired"
    assert out.extraction.stone_count is None
    assert out.extraction.sources.stone_count == "missing"


def test_parse_json_loose_python_style_dict():
    assert parse_json_loose("{'product_type': 'ring', 'stone_count': None}") == {"product_type": "ring", "stone_count": None}


@pytest.mark.parametrize("value,unit,expected", [(18, "mm", 18), (1.8, "cm", 18), (7, "inch", 177.8), (7, "us_ring_size", 17.32), (None, "mm", None)])
def test_size_conversion(value, unit, expected):
    assert size_to_mm(value, unit) == (pytest.approx(expected, abs=0.01) if expected is not None else None)
