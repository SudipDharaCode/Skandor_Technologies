"""Run the full service over the labeled test set and report metrics.

    python eval/run_eval.py --mode live     # real Gemini calls (needs GEMINI_API_KEY)
    python eval/run_eval.py --mode oracle   # no API: a fake LLM returns the gold labels.
                                            # Measures matcher + CV + confidence/flags in isolation
                                            # (an upper bound: extraction is perfect by construction).

Writes eval/results/<mode>_results.json and eval/results/<mode>_report.md.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm_client import GeminiClient, LLMResult  # noqa: E402
from app.matcher import load_inventory  # noqa: E402
from app.pipeline import Analyzer  # noqa: E402
from app.schemas import SPEC_FIELDS  # noqa: E402
from app.vision import count_stones  # noqa: E402

EVAL = ROOT / "eval"
IMAGES = EVAL / "images"
RESULTS = EVAL / "results"
SIZE_TOL_ABS, SIZE_TOL_REL = 0.3, 0.03


# oracle LLM
class OracleLLM:
    """Pretends to be a perfect text extractor: returns the gold label for the current case."""

    def __init__(self):
        self.case = None

    def generate(self, prompt, image=None, image_mime=None):
        exp = self.case["expected"]
        first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
        from_image = self.case.get("count_source") == "image"
        inferred = set(self.case.get("inferred", []))
        payload = {
            "product_type": first(exp["product_type"]),
            "material": first(exp["material"]),
            "stone_type": first(exp["stone_type"]),
            "stone_count": None if from_image else first(exp["stone_count"]),
            "size_value": first(exp["size_mm"]),
            "size_unit": "mm" if first(exp["size_mm"]) is not None else None,
            "setting_style": first(exp["setting_style"]),
            "image_stone_count": None,
        }
        src_key = {"size_value": "size"}
        payload["sources"] = {
            src_key.get(k, k): ("missing" if payload[k] is None else "inferred" if k in inferred else "stated")
            for k in ("product_type", "material", "stone_type", "stone_count", "size_value", "setting_style")
        }
        return LLMResult(text=json.dumps(payload), model="oracle")


# comparison helpers
def field_correct(field: str, got, expected) -> bool:
    options = expected if isinstance(expected, list) else [expected]
    for exp in options:
        if exp is None or got is None:
            if exp is None and got is None:
                return True
            continue
        if field == "size_mm":
            if abs(float(got) - float(exp)) <= max(SIZE_TOL_ABS, SIZE_TOL_REL * float(exp)):
                return True
        elif got == exp:
            return True
    return False


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


# main
def run(mode: str):
    settings = get_settings()
    inventory = load_inventory(settings.inventory_path)
    cases = json.loads((EVAL / "test_cases.json").read_text())["cases"]
    labels = json.loads((IMAGES / "labels.json").read_text())

    if mode == "live":
        llm = GeminiClient(settings)
    else:
        llm = OracleLLM()
    analyzer = Analyzer(llm, inventory, settings.max_image_bytes)

    rows = []
    for case in cases:
        if mode == "oracle":
            llm.case = case
        image = (IMAGES / case["image"]).read_bytes() if case.get("image") else None
        t = time.perf_counter()
        resp = analyzer.analyze(case["brief"], image, "image/png" if image else None)
        latency = (time.perf_counter() - t) * 1000

        got = {f: resp.spec[f].value for f in SPEC_FIELDS}
        conf = {f: resp.spec[f].confidence for f in SPEC_FIELDS}
        correct = {f: field_correct(f, got[f], case["expected"][f]) for f in SPEC_FIELDS}

        exp_match = case["expected_match"]
        exp_set = set(exp_match if isinstance(exp_match, list) else [exp_match]) if exp_match else set()
        ranked = [c.id for c in resp.match.candidates]

        true_count = labels[case["image"]]["true_count"] if case.get("image") else None
        ie = resp.image_estimate
        rows.append({
            "id": case["id"], "category": case["category"], "brief": case["brief"], "image": case.get("image"),
            "extraction_mode": resp.extraction.mode, "got": got, "confidence": conf, "correct": correct,
            "expected_match": exp_match, "ranked": ranked, "selected": resp.match.selected_id,
            "auto_selected": resp.match.auto_selected, "match_reason": resp.match.reason,
            "top1": bool(exp_set) and bool(ranked) and ranked[0] in exp_set,
            "top3": bool(exp_set) and any(r in exp_set for r in ranked[:3]),
            "true_count": true_count,
            "cv_count": ie.cv_count if ie else None, "vlm_count": ie.vlm_count if ie else None,
            "image_final_count": ie.final_count if ie else None,
            "cv_confidence": ie.cv_confidence if ie else None,
            "needs_review": resp.needs_review, "expected_review": case["expected_review"],
            "flags": [f"{f.code}:{f.field}" for f in resp.review_flags],
            "price": resp.price.value_usd, "latency_ms": round(latency, 1), "cost_usd": resp.meta["cost_usd"],
        })
        print(f"{case['id']} {case['category']:<12} mode={resp.extraction.mode:<9} fields={sum(correct.values())}/6 "
              f"top1={'Y' if rows[-1]['top1'] else '-'} review={resp.needs_review!s:<5} flags={rows[-1]['flags']}")

    # standalone CV run over every labeled image (independent of the briefs)
    cv_rows = []
    for f, lab in labels.items():
        r = count_stones(cv2.imread(str(IMAGES / f)))
        cv_rows.append({"image": f, "difficulty": lab["difficulty"], "true": lab["true_count"], "cv": r.count,
                        "hough": r.hough_count, "contour": r.contour_count, "confidence": r.confidence})

    report = build_report(mode, rows, cv_rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{mode}_results.json").write_text(json.dumps({"cases": rows, "cv": cv_rows}, indent=2))
    (RESULTS / f"{mode}_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def build_report(mode, rows, cv_rows) -> str:
    n = len(rows)
    L = [f"# Evaluation report ({mode})", "", f"Cases: {n} ({sum(1 for r in rows if r['image'])} with images)", ""]

    # extraction / final spec field accuracy
    L += ["## Field accuracy (final spec output)", "", "| field | accuracy |", "|---|---|"]
    total_ok = 0
    for f in SPEC_FIELDS:
        ok = sum(r["correct"][f] for r in rows)
        total_ok += ok
        L.append(f"| {f} | {ok}/{n} ({pct(ok, n)}) |")
    L.append(f"| **all fields** | **{total_ok}/{n * 6} ({pct(total_ok, n * 6)})** |")
    modes = defaultdict(int)
    for r in rows:
        modes[r["extraction_mode"]] += 1
    L += ["", "Extraction modes: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())), ""]

    # calibration: is "high" actually more accurate than "low"?
    L += ["## Confidence calibration (field level)", "", "| confidence | fields | accuracy |", "|---|---|---|"]
    for level in ("high", "medium", "low"):
        cells = [(r["correct"][f]) for r in rows for f in SPEC_FIELDS if r["confidence"][f] == level]
        L.append(f"| {level} | {len(cells)} | {pct(sum(cells), len(cells))} |")
    L.append("")

    # matching
    with_exp = [r for r in rows if r["expected_match"]]
    no_exp = [r for r in rows if not r["expected_match"]]
    auto = [r for r in rows if r["auto_selected"]]
    auto_ok = [r for r in auto if r["top1"]]
    L += ["## Matching", "", "| metric | value |", "|---|---|",
          f"| top-1 hit rate | {sum(r['top1'] for r in with_exp)}/{len(with_exp)} ({pct(sum(r['top1'] for r in with_exp), len(with_exp))}) |",
          f"| top-3 hit rate | {sum(r['top3'] for r in with_exp)}/{len(with_exp)} ({pct(sum(r['top3'] for r in with_exp), len(with_exp))}) |",
          f"| auto-select rate | {len(auto)}/{n} ({pct(len(auto), n)}) |",
          f"| auto-select precision | {len(auto_ok)}/{len(auto)} ({pct(len(auto_ok), len(auto))}) |",
          f"| correct abstention (no sensible match) | {sum(not r['auto_selected'] for r in no_exp)}/{len(no_exp)} |", ""]

    # counting
    img_rows = [r for r in rows if r["true_count"] is not None]
    def mae(key, rs):
        vals = [abs(r[key] - r["true_count"]) for r in rs if r[key] is not None]
        return (f"{statistics.mean(vals):.2f}", len(vals)) if vals else ("n/a", 0)
    cv_mae, cv_n = mae("cv_count", img_rows)
    vlm_mae, vlm_n = mae("vlm_count", img_rows)
    fin_mae, fin_n = mae("image_final_count", img_rows)
    all_cv = statistics.mean(abs(c["cv"] - c["true"]) for c in cv_rows)
    L += ["## Count error (MAE, stones)", "", "| estimator | MAE | n |", "|---|---|---|",
          f"| classical CV, all {len(cv_rows)} labeled images | {all_cv:.2f} | {len(cv_rows)} |",
          f"| classical CV, image cases | {cv_mae} | {cv_n} |",
          f"| vision-LLM, image cases | {vlm_mae} | {vlm_n} |",
          f"| combined image estimate | {fin_mae} | {fin_n} |", "",
          "| image | difficulty | true | CV | Hough | contour | CV conf |", "|---|---|---|---|---|---|---|"]
    for c in cv_rows:
        L.append(f"| {c['image']} | {c['difficulty']} | {c['true']} | {c['cv']} | {c['hough']} | {c['contour']} | {c['confidence']} |")
    if vlm_n:
        L += ["", "| case | true | CV | vision-LLM |", "|---|---|---|---|"]
        for r in img_rows:
            L.append(f"| {r['id']} | {r['true_count']} | {r['cv_count']} | {r['vlm_count']} |")
    L.append("")

    # review flags
    flagged = [r for r in rows if r["needs_review"]]
    tp = sum(r["needs_review"] and r["expected_review"] for r in rows)
    fp = sum(r["needs_review"] and not r["expected_review"] for r in rows)
    fn = sum(not r["needs_review"] and r["expected_review"] for r in rows)
    L += ["## Review flags", "", "| metric | value |", "|---|---|",
          f"| review-flag rate | {len(flagged)}/{n} ({pct(len(flagged), n)}) |",
          f"| flag precision (flagged & should be) | {tp}/{tp + fp} ({pct(tp, tp + fp)}) |",
          f"| flag recall (should be & flagged) | {tp}/{tp + fn} ({pct(tp, tp + fn)}) |", ""]

    # cost / latency
    lat = sorted(r["latency_ms"] for r in rows)
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    L += ["## Latency and cost", "", f"- latency p50 {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * 0.95) - 1]:.0f} ms",
          f"- total LLM cost ${sum(costs):.4f} over {len(costs)} priced requests" if costs else "- cost: n/a (no priced LLM calls)", ""]

    # per-case failures, worst first
    def badness(r):
        wrong_fields = 6 - sum(r["correct"].values())
        wrong_match = (1 if r["expected_match"] and not r["top1"] else 0) + (2 if r["auto_selected"] and not r["top1"] else 0)
        missed_review = 2 if r["expected_review"] and not r["needs_review"] else 0
        count_err = abs((r["image_final_count"] or 0) - r["true_count"]) / max(r["true_count"], 1) if r["true_count"] else 0
        return wrong_fields + 2 * wrong_match + missed_review + 3 * count_err
    L += ["## Cases ranked by severity of error", "", "| case | category | severity | wrong fields | match (expected -> top1, auto) | review (expected/got) | flags |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=badness, reverse=True):
        wrong = [f"{f}={r['got'][f]}" for f in SPEC_FIELDS if not r["correct"][f]]
        L.append(f"| {r['id']} | {r['category']} | {badness(r):.1f} | {', '.join(wrong) or '-'} | "
                 f"{r['expected_match']} -> {r['ranked'][0] if r['ranked'] else None}, {r['auto_selected']} | "
                 f"{r['expected_review']}/{r['needs_review']} | {', '.join(r['flags']) or '-'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "oracle"], default="live")
    run(ap.parse_args().mode)
