import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.llm_client import LLMResult  # noqa: E402
from app.matcher import load_inventory  # noqa: E402


class FakeLLM:
    """Returns scripted replies in order. A reply that is an Exception is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt, image=None, image_mime=None):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(text=reply, model="fake-model", input_tokens=100, output_tokens=50, latency_ms=1.0, cost_usd=0.0001)


@pytest.fixture
def inventory():
    return load_inventory(ROOT / "data" / "inventory.csv")


@pytest.fixture
def images_dir():
    return ROOT / "eval" / "images"
