"""Runtime configuration. Everything comes from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# USD per 1M tokens (input, output). Used only for cost logging; unknown models log cost as null.
# Override with GEMINI_PRICE_IN_PER_M / GEMINI_PRICE_OUT_PER_M if prices change.
DEFAULT_PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    primary_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    fallback_model: str | None = field(default_factory=lambda: os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite") or None)
    request_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30")))
    inventory_path: Path = field(default_factory=lambda: Path(os.getenv("INVENTORY_PATH", ROOT / "data" / "inventory.csv")))
    max_image_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_MB", "10")) * 1024 * 1024)

    def price_for(self, model: str) -> tuple[float, float] | None:
        env_in, env_out = os.getenv("GEMINI_PRICE_IN_PER_M"), os.getenv("GEMINI_PRICE_OUT_PER_M")
        if env_in and env_out:
            return float(env_in), float(env_out)
        return DEFAULT_PRICES.get(model)


def get_settings() -> Settings:
    return Settings()
