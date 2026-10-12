"""Thin Gemini wrapper. The rest of the code depends only on the LLMClient protocol,
so tests can swap in a fake client that returns whatever (malformed) text they want."""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None


class LLMError(Exception):
    """Every configured model failed (network, quota, auth, safety block...)."""


class LLMClient(Protocol):
    def generate(self, prompt: str, image: bytes | None = None, image_mime: str | None = None) -> LLMResult: ...


class GeminiClient:
    """Calls the primary model; on any API error falls back to the secondary model."""

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self._types = types
        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.request_timeout_s * 1000)),
        )

    def _models(self) -> list[str]:
        models = [self._settings.primary_model]
        if self._settings.fallback_model and self._settings.fallback_model != self._settings.primary_model:
            models.append(self._settings.fallback_model)
        return models

    def generate(self, prompt: str, image: bytes | None = None, image_mime: str | None = None) -> LLMResult:
        types = self._types
        contents: list = []
        if image is not None:
            contents.append(types.Part.from_bytes(data=image, mime_type=image_mime or "image/jpeg"))
        contents.append(prompt)
        config = types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")

        errors = []
        for model in self._models():
            start = time.perf_counter()
            try:
                resp = self._client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # network, 4xx/5xx, timeout: try the next model
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                log.warning("gemini call failed on %s: %s", model, exc)
                continue
            latency = (time.perf_counter() - start) * 1000
            usage = resp.usage_metadata
            in_tok = (usage.prompt_token_count or 0) if usage else 0
            # thinking tokens are billed as output tokens
            out_tok = ((usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)) if usage else 0
            price = self._settings.price_for(model)
            cost = round((in_tok * price[0] + out_tok * price[1]) / 1e6, 6) if price else None
            try:
                text = resp.text
            except Exception:  # blocked / no candidates
                text = None
            return LLMResult(text=text, model=model, input_tokens=in_tok, output_tokens=out_tok,
                             latency_ms=round(latency, 1), cost_usd=cost)
        raise LLMError("; ".join(errors) or "no models configured")
