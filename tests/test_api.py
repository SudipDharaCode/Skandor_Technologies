import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import Analyzer
from tests.conftest import FakeLLM

STATED = {k: "stated" for k in ("product_type", "material", "stone_type", "stone_count", "size", "setting_style")}


def reply(**kw):
    base = {"product_type": None, "material": None, "stone_type": None, "stone_count": None, "size_value": None,
            "size_unit": None, "setting_style": None, "sources": STATED, "image_stone_count": None}
    base.update(kw)
    return json.dumps(base)


@pytest.fixture
def client_with(inventory):
    def make(*replies):
        app.state.analyzer = Analyzer(FakeLLM(*replies), inventory)
        return TestClient(app)
    yield make
    del app.state.analyzer


def test_happy_path_text_only(client_with):
    c = client_with(reply(product_type="ring", material="white_gold", stone_type="diamond", stone_count=12,
                          size_value=17.3, size_unit="mm", setting_style="halo"))
    r = c.post("/analyze", data={"brief": "white gold halo ring with 12 diamonds, 17.3mm"})
    assert r.status_code == 200
    body = r.json()
    assert body["match"]["selected_id"] == "R003"
    assert body["price"]["value_usd"] == 1450
    assert body["spec"]["material"]["confidence"] == "high"
    assert body["needs_review"] is False


def test_brief_vs_image_disagreement_is_flagged(client_with, images_dir):
    c = client_with(reply(product_type="ring", material="white_gold", stone_type="diamond", stone_count=20,
                          setting_style="halo", image_stone_count=12))
    with open(images_dir / "img01_halo12_clean.png", "rb") as f:
        r = c.post("/analyze", data={"brief": "halo ring, 20 diamonds"}, files={"image": ("ring.png", f, "image/png")})
    body = r.json()
    assert body["image_estimate"]["cv_count"] == 12
    assert body["spec"]["stone_count"]["confidence"] == "low"
    assert "source_disagreement" in {f["code"] for f in body["review_flags"]}
    assert body["needs_review"] is True


def test_malformed_model_output_degrades_but_returns_200(client_with):
    c = client_with("<<<garbage>>>", "still garbage")
    r = c.post("/analyze", data={"brief": "something nice"})
    assert r.status_code == 200
    body = r.json()
    assert body["extraction"]["mode"] == "degraded"
    assert body["match"]["auto_selected"] is False
    assert body["price"]["value_usd"] is None
    assert body["needs_review"] is True


def test_out_of_range_size_is_flagged(client_with):
    c = client_with(reply(product_type="ring", material="yellow_gold", stone_type="none", stone_count=0,
                          size_value=80, size_unit="mm", setting_style="plain"))
    body = c.post("/analyze", data={"brief": "plain gold ring 80mm"}).json()
    assert body["spec"]["size_mm"]["confidence"] == "low"
    assert any(f["code"] == "out_of_range" for f in body["review_flags"])


def test_unreadable_image_is_ignored_and_flagged(client_with):
    c = client_with(reply(product_type="ring"))
    r = c.post("/analyze", data={"brief": "a ring"}, files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 200
    assert any(f["code"] == "image_unusable" for f in r.json()["review_flags"])


def test_empty_request_rejected(client_with):
    assert client_with().post("/analyze", data={"brief": "  "}).status_code == 422
