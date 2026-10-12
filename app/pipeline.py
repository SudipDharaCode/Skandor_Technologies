"""Orchestration: extract -> image estimate -> confidence -> match -> price -> flags."""

import json
import logging
import time
import uuid

from app.confidence import image_estimate, score_fields
from app.extractor import extract, to_spec
from app.llm_client import LLMClient
from app.matcher import match
from app.pricing import estimate_price
from app.schemas import AnalyzeResponse, ExtractionInfo, InventoryItem, ReviewFlag
from app.vision import count_stones, decode_image

log = logging.getLogger("analyze")

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


class Analyzer:
    def __init__(self, llm: LLMClient | None, inventory: list[InventoryItem], max_image_bytes: int = 10 * 1024 * 1024):
        self.llm = llm
        self.inventory = inventory
        self.by_id = {i.id: i for i in inventory}
        self.max_image_bytes = max_image_bytes

    def analyze(self, brief: str, image: bytes | None = None, image_mime: str | None = None) -> AnalyzeResponse:
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        flags: list[ReviewFlag] = []

        # image intake: decide once whether the image is usable for both CV and the LLM
        img_array, image_status = None, "none"
        if image:
            if len(image) > self.max_image_bytes:
                image_status = "too_large"
            elif image_mime and image_mime not in ALLOWED_MIME:
                image_status = "unsupported_type"
            else:
                img_array = decode_image(image)
                image_status = "ok" if img_array is not None else "unreadable"
            if image_status != "ok":
                flags.append(ReviewFlag(code="image_unusable", field="image", message=f"reference image ignored: {image_status}"))
                image = None

        # 1. LLM extraction
        t = time.perf_counter()
        outcome = extract(brief, self.llm, image, image_mime)
        timings["extract_ms"] = _ms(t)
        spec = to_spec(outcome.extraction)

        # 2. CV estimate
        t = time.perf_counter()
        cv = count_stones(img_array) if img_array is not None else None
        timings["cv_ms"] = _ms(t)
        vlm_count = outcome.extraction.image_stone_count if image_status == "ok" else None
        img_est = image_estimate(cv, vlm_count, "none" if image_status == "none" else image_status)
        if img_est and img_est.cv_count is not None and img_est.vlm_count is not None and img_est.confidence == "low":
            flags.append(ReviewFlag(code="source_disagreement", field="image_stone_count", message=img_est.reason))

        # 3. confidence + sanity checks
        fields, used_spec, field_flags = score_fields(spec, outcome, img_est)
        flags += field_flags

        # 4. deterministic matching + price
        t = time.perf_counter()
        m = match(used_spec, self.inventory)
        timings["match_ms"] = _ms(t)
        if not m.auto_selected:
            flags.append(ReviewFlag(code="no_confident_match", field="match", message=m.reason))
        price = estimate_price(m, used_spec, fields["stone_count"].confidence, self.by_id)

        # meta: latency + cost per request
        calls = [{"model": c.model, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                  "latency_ms": c.latency_ms, "cost_usd": c.cost_usd} for c in outcome.calls]
        costs = [c.cost_usd for c in outcome.calls if c.cost_usd is not None]
        timings["total_ms"] = _ms(t0)
        meta = {"timings": timings, "llm_calls": calls, "cost_usd": round(sum(costs), 6) if costs else None,
                "image_status": image_status}

        resp = AnalyzeResponse(
            request_id=request_id,
            spec=fields,
            extraction=ExtractionInfo(mode=outcome.mode, steps=outcome.steps, model_used=outcome.model_used,
                                      repaired_fields=outcome.repaired_fields),
            image_estimate=img_est,
            match=m,
            price=price,
            review_flags=flags,
            needs_review=bool(flags),
            meta=meta,
        )
        log.info(json.dumps({"request_id": request_id, "extraction_mode": outcome.mode, "model": outcome.model_used,
                             "needs_review": resp.needs_review, "flags": [f.code for f in flags],
                             "cost_usd": meta["cost_usd"], **timings}))
        return resp


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)
