# Brief to Priced Estimate

A FastAPI service that turns a sales rep's short brief (plus an optional reference photo) into a structured
product spec, matches it against inventory, estimates a stone count from the photo, prices it, and says how much
it trusts every output. When it is unsure, it flags the request for a human instead of guessing.

**Domain: custom jewelry** (rings, pendants, earrings, bracelets, necklaces).
**Image quantity: number of gemstones** in the reference photo.
**LLM: Google Gemini** (`gemini-2.5-flash`, falling back to `gemini-2.5-flash-lite`; both configurable).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then put your key in GEMINI_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs, or:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "brief=White gold halo ring with 12 diamonds, US size 7" \
  -F "image=@eval/images/img01_halo12_clean.png;type=image/png"
```

Without a key the service still starts: extraction degrades to an empty spec (and says so), CV and matching still run,
and every response is flagged for review.

**Tests and evaluation**

```bash
pytest -q                                   # 37 tests, no API key needed (LLM is faked)
python eval/generate_images.py              # regenerate the synthetic images (deterministic)
python eval/run_eval.py --mode oracle       # no API key: perfect extraction, measures matcher/CV/flags
python eval/run_eval.py --mode live         # real Gemini calls, full end-to-end numbers
```

**Docker**

```bash
docker compose up --build
```

---

## How a request flows

```
brief (+ image)
   │
   ├─ image intake: size/type/decode checks; an unusable image is dropped and flagged
   │
   ├─ 1. LLM extraction (Gemini, JSON mode, temperature 0)  ──> strict Pydantic validation
   │        recovery ladder: clean → repaired → retried → degraded   (reported in `extraction.mode`)
   │        unit conversion (US ring size, inch, cm → mm) is done in code, not by the LLM
   │
   ├─ 2. CV stone count (OpenCV): Hough circles + outline contours, cross-checked
   │        + the vision-LLM's own count from the same Gemini call
   │
   ├─ 3. confidence + sanity rules: per-field high/medium/low with a reason; range checks;
   │        brief vs image vs vision-LLM disagreement → review flags
   │
   ├─ 4. deterministic weighted matching against data/inventory.csv
   │
   └─ 5. transparent price: matched item base price ± stone-count adjustment
```

| file | role |
|---|---|
| `app/main.py` | FastAPI app, `POST /analyze`, `GET /health` |
| `app/pipeline.py` | orchestration, per-request latency/cost logging |
| `app/extractor.py` | prompt, JSON parsing, repair, retry, degrade, unit conversion |
| `app/llm_client.py` | Gemini wrapper with fallback model, token + cost accounting |
| `app/vision.py` | classical CV stone counter |
| `app/confidence.py` | confidence levels, plausible ranges, review flags |
| `app/matcher.py` | weighted scoring matcher (no LLM) |
| `app/pricing.py` | price estimate |
| `data/inventory.csv` | 40 hand-written inventory items |
| `eval/` | test set, image generator, evaluation harness, results |

---

## Design decisions

### 1. Extraction: what the LLM is and isn't trusted with
* Gemini is called with `response_mime_type="application/json"` and temperature 0, and returns enum values from a fixed vocabulary.
* Each field also comes with a **source** (`stated` / `inferred` / `missing`), which is the starting point for its confidence.
* The LLM copies the size **as written** (`7`, `us_ring_size`); code converts it to mm. Arithmetic is not left to the model.
* The strict schema uses `extra="forbid"`. The evaluation found that with `extra="ignore"`, a reply using `"metal"` instead of `"material"` validated as an *empty but "clean"* spec. That is the worst kind of silent failure.
* **Recovery ladder**, cheapest first:
  1. **clean**: valid on the first try.
  2. **repaired**: deterministic fixes without another API call. Strips markdown fences and prose, fixes trailing commas and quote styles, maps renamed keys (`metal → material`), maps synonyms (`18k White Gold → white_gold`, `CZ → cubic_zirconia`), turns `"about 12"` into 12, and nulls out-of-vocabulary values.
  3. **retried**: one more call that includes the validation error.
  4. **degraded**: an all-null spec, everything low confidence, flagged.

  If the primary model errors (timeout, quota, 5xx), the client switches to the fallback model before giving up. The response's `extraction.mode` and `extraction.steps` say exactly what happened.

### 2. Matching: deterministic weighted score

| field | weight | similarity |
|---|---|---|
| product_type | 0.30 | exact |
| material | 0.25 | exact; partial credit table (e.g. yellow vs rose gold 0.5, white gold vs platinum 0.4) |
| stone_type | 0.15 | exact; diamond vs cubic zirconia 0.3 |
| stone_count | 0.15 | `1 − |a−b| / max(a,b)` |
| size_mm | 0.10 | linear falloff to 0 at a per-type tolerance (ring 1.5 mm … necklace 60 mm) |
| setting_style | 0.05 | exact; pave vs channel 0.4, halo vs cluster 0.3 |

* `score = Σ wᵢ·simᵢ / Σ wᵢ` over the fields the spec actually knows.
* `coverage = Σ known weights / Σ all weights`.
* Ranking ties break on item id, so the same input always gives the same ranking, regardless of file order (there's a test for this).

An item is **auto-selected** only if all of these hold:
- score ≥ 0.80
- coverage ≥ 0.60
- it beats the runner-up by at least 0.02
- every hard field the spec gives (product type, material, stone type) matches exactly

Otherwise the ranked candidates are still returned, but `selected_id` is null and a review flag is raised.

**Why not let the LLM pick the item?**
* Reproducibility: the same brief must always give the same quote, and an LLM can't guarantee that, even at temperature 0, across model versions.
* Auditability: every score here breaks down per field, so a reviewer can see why R003 beat R005.
* Grounding: the LLM can invent SKUs or "close enough" items, while a scorer can only return rows that exist.
* Cost and speed: about 0.2 ms and $0 versus another API call.
* Tunability: business rules ("never auto-swap platinum for white gold") are one line of code, not a prompt-engineering exercise.

The LLM is good at the fuzzy part (reading the brief). The matcher handles the exact part.

### 3. Computer vision: counting stones
Two classical estimators that fail in *different* ways:
* **Hough circles** (`HOUGH_GRADIENT_ALT` plus non-max suppression). This is the primary count. Any round bright spot, such as a reflection, fools it.
* **Outline contours**: black-hat morphology finds the dark girdle around each stone, Otsu thresholds it, and the round enclosed holes are counted. The median is taken across 4 kernel sizes. It misses dark stones and stones without a visible outline.

When the two agree, confidence is high. When they disagree, confidence drops, and the cases where they disagree are where the count is actually wrong (see EVALUATION.md). The same Gemini call also returns its own `image_stone_count`. The service compares that with CV and flags disagreement.

### 4. Confidence and review flags
* **Field confidence** starts from the source: stated → high, inferred → medium, missing → low. It is downgraded when:
  - the value was repaired,
  - extraction degraded,
  - the value is outside a plausible range (ring 14–24 mm, bracelet 140–240 mm, ≤150 stones, …),
  - the sources disagree.
* **Stone count** comes from the brief if given, since it is the customer's request. If the brief and image disagree beyond ±max(1, 10%), the value is kept but marked low and flagged. If the brief has no count, the image count is used. A *low-confidence* image count is still reported but **not used for matching**, because the evaluation showed a wrong count picking the wrong ring.
* **Flags** (`needs_review = true` when there are any):
  - `extraction_degraded`, `out_of_range`, `source_disagreement`, `low_confidence`, `missing_key_field`, `no_confident_match`, `image_unusable`, `inconsistent_spec`.

### 5. Price
`base price of the auto-selected item + (requested stones − item stones) × per-stone price`.
* With no auto-selected item, the price is null rather than a guess.
* The price's confidence is the minimum of the match confidence and the stone-count confidence.

### 6. Operations
* **Config:** only environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_FALLBACK_MODEL`, `LLM_TIMEOUT_S`, …). Nothing is hardcoded, and `.env` is git-ignored.
* **Logging:** every request logs one JSON line with the request id, extraction mode, model, flags, per-stage latency and cost. The same numbers are returned in `meta`.
  - Cost comes from Gemini token usage (thinking tokens count as output) times a per-model price table.
  - Edit `DEFAULT_PRICES` in `app/config.py`, or set `GEMINI_PRICE_*_PER_M`, if prices change.
* **Request limits:**
  - images: max 10 MB; jpeg, png and webp only
  - brief: max 5000 characters
  - an empty request (blank brief and no image) returns 422

---

## Limitations
* **The data is synthetic.** The inventory, briefs and all 13 images are made up by me. The images are cleaner than real jewelry photos, which have specular facets, prongs, shadows and tiny melee stones. The CV numbers here are therefore an upper bound. Before production, the next step is 50+ real, hand-counted photos.
* The CV counter assumes a roughly top-down view and stones 0.8–9% of the image's short side. It doesn't segment the jewelry first, so round things in the background count as stones.
* The test set is small (27 cases), so one case moves a metric by about 4 percentage points.
* The vision-LLM count comes from the same call as text extraction (to save cost). A dedicated counting prompt might be more accurate.
* Pricing is deliberately simple: no metal weight, carat, labor or market prices.
* There is no auth, rate limiting or persistence. See the AWS note below.

## Deploying on AWS (sketch)
* **Compute:** **ECS Fargate** behind an **ALB**, using the Docker image. The service is stateless, OpenCV is CPU-only and requests are seconds-long, so Lambda would also work, but Fargate avoids cold-starting OpenCV and the 6 MB Lambda payload limit for photos.
* **Secrets:** **Secrets Manager** for `GEMINI_API_KEY`, injected as an env var by the task definition. The key never goes into the image.
* **Images:** large uploads go to **S3** through a presigned URL, and the API receives the key. S3 also keeps the reviewed photos as future training and eval data.
* **Review queue:** flagged responses go to **SQS**, then a small reviewer UI, and the reviewer's corrections are stored in **DynamoDB**. Those corrections become new labeled eval cases.
* **Observability:** **CloudWatch** logs (the JSON lines are already structured), plus metrics and alarms on degraded-extraction rate, review rate, p95 latency and daily LLM cost.
* **Inventory:** today it's a CSV loaded at startup. In production it would be a table in **RDS/DynamoDB** with a cache, or an S3 object reloaded on change.
* **Bedrock** could replace Gemini if the company wants to stay inside AWS. The `LLMClient` protocol makes that a single-file change.

## How AI tools were used
I used an AI coding assistant to speed up the parts of this project that are mostly typing: the FastAPI and Pydantic scaffolding, the synthetic image generator, the OpenCV parameter sweeps, the test boilerplate and the first drafts of this README and EVALUATION.md. I treated its output as a first draft to verify, not as finished work, and the evaluation harness was how I checked it. Running the eval exposed three flaws in the generated code, and I fixed each one and added a test. First, the strict schema silently ignored renamed keys, so a reply with `"metal"` instead of `"material"` became an empty spec marked "clean"; I switched to rejecting unknown keys so it goes through repair. Second, a low-confidence CV count was still allowed to choose the product, which picked the wrong ring; flagged values are now kept out of matching. Third, the matcher auto-quoted a white-gold necklace for a platinum request because the weighted score cleared 0.80; product type, material and stone type now have to match exactly before anything is auto-selected. I rejected suggestions to let the LLM convert units (US ring size to mm) and choose the inventory item, and moved both into deterministic code so they are repeatable and auditable. I also made the image set harder (overlap with JPEG artifacts, a low-resolution source, round reflections on the metal) after the first synthetic images turned out to be trivially easy for Hough circles. Finally, I didn't take any metric from an AI summary: every number in EVALUATION.md comes from running `eval/run_eval.py`.
