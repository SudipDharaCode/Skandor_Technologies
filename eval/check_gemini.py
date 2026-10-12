"""Sanity check for the Gemini setup: key loaded, models reachable, JSON extraction works.

    python eval/check_gemini.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.extractor import extract, to_spec  # noqa: E402
from app.llm_client import GeminiClient, LLMError  # noqa: E402


def main() -> int:
    s = get_settings()
    if not s.gemini_api_key:
        print("GEMINI_API_KEY is empty. Put your key in .env (GEMINI_API_KEY=...) and try again.")
        return 1
    print(f"key loaded (ends ...{s.gemini_api_key[-4:]}), primary={s.primary_model}, fallback={s.fallback_model}")

    client = GeminiClient(s)
    try:
        res = client.generate('Reply with exactly this JSON: {"ok": true}')
    except LLMError as exc:
        print(f"Gemini call failed on every model:\n  {exc}")
        print("Check the key, and that GEMINI_MODEL / GEMINI_FALLBACK_MODEL are model names your key can use.")
        return 1
    print(f"ping ok: model={res.model} latency={res.latency_ms} ms tokens={res.input_tokens}+{res.output_tokens} "
          f"cost=${res.cost_usd} reply={res.text!r}")

    out = extract("Need a white gold halo ring with 12 diamonds, US size 7.", client)
    print(f"extraction mode={out.mode} model={out.model_used}")
    print(json.dumps(to_spec(out.extraction).model_dump(), indent=2))
    return 0 if out.mode != "degraded" else 1


if __name__ == "__main__":
    sys.exit(main())
