# Evaluation

**Test set:** 27 labeled cases in `eval/test_cases.json`; 13 of them include one of 13 synthetic images in `eval/images/`.

| category | cases |
|---|---|
| easy | 4 |
| image | 9 |
| hard image | 3 |
| brief vs. photo conflict | 1 |
| vague | 1 |
| empty | 1 |
| typos | 1 |
| unit conversion | 1 |
| "center + 11 around" counting | 1 |
| synonyms | 1 |
| out-of-range | 2 |
| contradictory text | 1 |
| ambiguous (no metal) | 1 |

**Labels:**
- Every field has a gold value; a list means any of those values is acceptable, e.g. `setting_style` is null or `channel` for "tennis bracelet".
- Each case has the item a human would pick, or null if nothing fits.
- Each case says whether a human *should* review it.
- Image ground truth is exact because the images are rendered with a known count (`labels.json`).

**Harness:** `eval/run_eval.py` runs the real `Analyzer` pipeline on every case and writes `eval/results/<mode>_report.md` and `<mode>_results.json`.

| mode | what the LLM is | what it measures |
|---|---|---|
| `oracle` | a fake LLM that returns the gold labels (with the count hidden when it must come from the image) | matcher, CV, confidence and flag logic, isolated from LLM error |
| `live` | real Gemini (`gemini-2.5-flash`, fallback `-flash-lite`) | the whole system end to end |

> **Status:** the numbers below are from the **oracle** run. The development machine had no `GEMINI_API_KEY`, so the live run hasn't been done yet. Run `python eval/run_eval.py --mode live` and paste `eval/results/live_report.md` into the Live section. That fills in extraction accuracy, vision-LLM count MAE, and cost and latency.

---

## Metrics: oracle run (perfect text extraction)

| metric | value |
|---|---|
| field accuracy, all 6 fields (final output) | 159/162 (98%). All 3 misses are image-derived stone counts |
| matching top-1 hit rate | 24/24 (100%) |
| matching top-3 hit rate | 24/24 (100%) |
| auto-select rate | 21/27 (78%) |
| **auto-select precision** (auto-chosen item was correct) | **21/21 (100%)** |
| correct abstention when no item fits | 3/3 |
| **count MAE, classical CV** (all 13 images) | **0.92 stones** |
| count MAE, vision-LLM | needs live run |
| review-flag rate | 10/27 (37%) |
| flag precision / recall vs. labels | 90% / 100% |

**Confidence calibration** (does "high" mean right?):

| confidence | fields | accuracy |
|---|---|---|
| high | 130 | 100% |
| medium | 5 | 100% |
| low | 27 | 89% |

"Low" accuracy is high because most low-confidence fields are *correctly null*: the brief didn't say, and the service didn't invent a value. The 3 wrong fields in the whole run are all marked low and flagged. No wrong value was marked high.

**CV per image:**

| image | difficulty | true | CV | Hough | contour | CV confidence |
|---|---|---|---|---|---|---|
| img01 | clean | 12 | 12 | 12 | 12 | high |
| img02 | mild noise | 20 | 20 | 20 | 20 | high |
| img03 | dark sapphires | 8 | 8 | 8 | 0 | low |
| img04 | heavy blur + noise | 16 | 16 | 16 | 16 | high |
| img05 | small stones, low contrast | 30 | 30 | 30 | 30 | high |
| img06 | stones touching | 24 | 23 | 23 | 24 | high |
| img07 | strong uneven lighting | 14 | 14 | 14 | 14 | high |
| img08 | rubies, cluster | 7 | 7 | 7 | 7 | high |
| img09 | 40 small stones | 40 | 40 | 40 | 40 | high |
| img10 | single large stone | 1 | 1 | 1 | 0 | low |
| img11 | heavy overlap + JPEG | 22 | 20 | 20 | 22 | medium |
| img12 | low resolution (128 px source) | 24 | 20 | 20 | 15 | low |
| img13 | round reflections on metal | 12 | 17 | 17 | 10 | low |

**Is the CV confidence useful?** Yes:
- 9 images are exact. 7 of them are rated high. The other 2 are rated low because the contour method can't see their stones: img03 has dark stones, and in img10 the stone touches the edge of the band. Both are false alarms, but on the safe side.
- Every image with an error above 1 stone is rated medium or low.
- The worst miss (img13, +5) is rated low.

The one high-confidence miss is img06: 23 instead of 24, which is inside the "agree" tolerance.

## Metrics: live run (Gemini)

_Pending. Paste `eval/results/live_report.md` here after `python eval/run_eval.py --mode live`._
Things to look at:
* Per-field extraction accuracy, especially C20 (typos), C23 (0.6 inch → 15.24 mm), C24 (1 + 11 = 12 stones) and C22 (contradictory brief).
* Vision-LLM vs CV count MAE per image, and whether combining them fixes C10 (reflections), where CV is wrong but a vision model shouldn't be fooled.
* The extraction-mode distribution: how often repair or retry was needed.
* p50 / p95 latency and cost per request.

---

## Failure analysis: 3 worst cases (oracle run)

### 1. C10: round reflections on metal, 17 counted vs 12 true (img13)
**What happened:**
- Five bright round glints on the white-gold band look exactly like stones to Hough circles, so Hough counted 17.
- The contour method requires a dark outline around each stone, so it skipped the glints. But it also lost two real stones that the glints overlapped, so it counted 10.
- The methods disagreed by 41%, so CV confidence was low and the count was flagged.

**Bug this exposed:**
- In the first version, that low-confidence count of 17 was still passed to the matcher. It pushed R005 (16 stones) ahead of the correct R003 (12 stones), and R005 was **auto-selected and priced**.
- So the flag was right, but the system acted on a value it had already said it didn't trust.

**Fix (done):** an image-only count with low confidence is reported but excluded from matching. R003 and R005 then tie, so the matcher abstains and asks a human. Auto-select precision went from 22/23 to 21/21.

**What I'd change next:**
- Use a stone-vs-glint classifier on each detection. Real stones have facets, an internal texture and a girdle edge; glints are flat saturated white.
- Even a small CNN trained on 32×32 crops would do it.
- With a key, the vision-LLM count is a third opinion that shouldn't be fooled by glints.

### 2. C11: low-resolution photo, 20 counted vs 24 true (img12)
**What happened:**
- The source was about 128 px wide, so each stone is about 2 px of real detail upscaled.
- Neighbouring stones blur together. Hough finds 20 and contours find 15.
- They disagree, so the count was low confidence and flagged. It was correctly not used for matching, and R016 was still selected on material, type and style.

**Root cause:** there's no information in the image to recover; no classical method will count stones that aren't resolved.

**What I'd change:**
- Check image quality up front: estimate stone pixel size from the Hough radius and refuse to count below about 5 px, saying "photo too low-res" instead of returning a number.
- Ask the rep for a closer photo.
- This is a "know when you can't" fix, not an accuracy fix.

### 3. C12: overlapping stones with JPEG artifacts, 20 counted vs 22 true (img11)
**What happened:**
- In an eternity band of large, touching, size-varied stones, non-max suppression merged two pairs of overlapping circles.
- JPEG blocking at quality 25 weakened the edges.
- The contour method got exactly 22. The disagreement was 9%, which gave medium CV confidence and a review flag that my label says wasn't needed. This is the one false-positive flag (precision 90%).

**Root cause:** Hough uses one global NMS radius; it can't separate circles whose centers are closer than 0.8 × radius.

**What I'd change:**
- Use a watershed split on a distance transform of the stone mask for touching blobs.
- Or make NMS radius-aware, using the median radius instead of the max.
- When both methods are individually plausible, a weighted vote would beat "always trust Hough". That vote should be tuned on a held-out split, not this test set.

### Bugs found by the harness (fixed before these numbers)
1. **Silent empty extraction.**
   - Problem: the strict schema used `extra="ignore"`, so a reply with renamed keys (`"metal"`, `"stones"`) validated as an empty but "clean" spec.
   - Fix: switched to `extra="forbid"`, so these replies go through repair. Covered by `test_off_schema_values_are_repaired_without_extra_call`.
2. **Auto-quoting the wrong metal.**
   - Problem: for C26 (platinum riviera necklace, which isn't in inventory), the white-gold N001 scored 0.81 and was auto-selected and priced.
   - Fix: the hard fields (product type, material, stone type) must match exactly before anything is auto-selected. Covered by `test_hard_field_mismatch_blocks_auto_select`.
3. **Acting on a flagged value.** This is C10 above.

---

## Next steps
1. **Real data.** 50+ real jewelry photos with hand counts, and 50+ real briefs from sales reps. The synthetic set is good for finding logic bugs and bad for estimating real accuracy.
2. **Split the data.** Keep a held-out test split. The thresholds here (0.80 auto-select, ±10% agreement, Hough `param2`) were set while looking at this set, so these numbers are optimistic.
3. **Stone-vs-glint classifier and an image-quality gate** for CV (failures 1 and 2).
4. **A dedicated vision-LLM counting call**, then learn when to trust CV vs. the VLM from the review data.
5. **Feed reviewer corrections back** as eval cases. Track the review rate, and auto-select precision per week, as the production quality metrics.
6. **Better pricing:** metal weight × spot price + per-carat stone price + labor.
