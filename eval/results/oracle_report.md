# Evaluation report (oracle)

Cases: 27 (13 with images)

## Field accuracy (final spec output)

| field | accuracy |
|---|---|
| product_type | 27/27 (100%) |
| material | 27/27 (100%) |
| stone_type | 27/27 (100%) |
| stone_count | 24/27 (89%) |
| size_mm | 27/27 (100%) |
| setting_style | 27/27 (100%) |
| **all fields** | **159/162 (98%)** |

Extraction modes: clean=27

## Confidence calibration (field level)

| confidence | fields | accuracy |
|---|---|---|
| high | 130 | 100% |
| medium | 5 | 100% |
| low | 27 | 89% |

## Matching

| metric | value |
|---|---|
| top-1 hit rate | 24/24 (100%) |
| top-3 hit rate | 24/24 (100%) |
| auto-select rate | 21/27 (78%) |
| auto-select precision | 21/21 (100%) |
| correct abstention (no sensible match) | 3/3 |

## Count error (MAE, stones)

| estimator | MAE | n |
|---|---|---|
| classical CV, all 13 labeled images | 0.92 | 13 |
| classical CV, image cases | 0.92 | 13 |
| vision-LLM, image cases | n/a | 0 |
| combined image estimate | 0.92 | 13 |

| image | difficulty | true | CV | Hough | contour | CV conf |
|---|---|---|---|---|---|---|
| img01_halo12_clean.png | clean | 12 | 12 | 12 | 12 | high |
| img02_pave20_rose.png | mild noise | 20 | 20 | 20 | 20 | high |
| img03_channel8_sapphire.png | dark colored stones | 8 | 8 | 8 | 0 | low |
| img04_halo16_blur_noise.png | heavy blur + noise | 16 | 16 | 16 | 16 | high |
| img05_pave30_small_platinum.png | small stones, low metal/stone contrast | 30 | 30 | 30 | 30 | high |
| img06_halo24_touching.png | stones touching/overlapping | 24 | 23 | 23 | 24 | high |
| img07_halo14_cz_lighting.png | strong uneven lighting | 14 | 14 | 14 | 14 | high |
| img08_cluster7_ruby.png | colored stones, cluster | 7 | 7 | 7 | 7 | high |
| img09_tennis40_two_rows.png | many small stones | 40 | 40 | 40 | 40 | high |
| img10_solitaire1.png | single large stone | 1 | 1 | 1 | 0 | low |
| img11_halo22_overlap_jpeg.png | heavy overlap + JPEG artifacts | 22 | 20 | 20 | 22 | medium |
| img12_pave24_lowres.png | low resolution (128px source) | 24 | 20 | 20 | 15 | low |
| img13_halo12_reflections.png | round reflections on metal (false positives) | 12 | 17 | 17 | 10 | low |

## Review flags

| metric | value |
|---|---|
| review-flag rate | 10/27 (37%) |
| flag precision (flagged & should be) | 9/10 (90%) |
| flag recall (should be & flagged) | 9/9 (100%) |

## Latency and cost

- latency p50 1 ms, p95 28 ms
- cost: n/a (no priced LLM calls)

## Cases ranked by severity of error

| case | category | severity | wrong fields | match (expected -> top1, auto) | review (expected/got) | flags |
|---|---|---|---|---|---|---|
| C10 | hard_image | 2.2 | stone_count=17 | R003 -> R003, False | True/True | low_confidence:stone_count, no_confident_match:match |
| C11 | hard_image | 1.5 | stone_count=20 | R016 -> R016, True | True/True | low_confidence:stone_count |
| C12 | hard_image | 1.3 | stone_count=20 | R018 -> R018, True | False/True | low_confidence:stone_count |
| C06 | conflict | 0.1 | - | R004 -> R004, True | True/True | source_disagreement:stone_count |
| C01 | easy | 0.0 | - | R003 -> R003, True | False/False | - |
| C02 | easy | 0.0 | - | R002 -> R002, True | False/False | - |
| C03 | easy | 0.0 | - | B002 -> B002, True | False/False | - |
| C04 | easy | 0.0 | - | E001 -> E001, True | False/False | - |
| C05 | image | 0.0 | - | R003 -> R003, True | False/False | - |
| C07 | image | 0.0 | - | R009 -> R009, True | False/False | - |
| C08 | image | 0.0 | - | R011 -> R011, True | False/False | - |
| C09 | image | 0.0 | - | B002 -> B002, True | False/False | - |
| C13 | image | 0.0 | - | R001 -> R001, True | False/False | - |
| C14 | image | 0.0 | - | P003 -> P003, True | False/False | - |
| C15 | image | 0.0 | - | R005 -> R005, True | False/False | - |
| C16 | image | 0.0 | - | R015 -> R015, True | False/False | - |
| C17 | image | 0.0 | - | R008 -> R008, True | False/False | - |
| C18 | vague | 0.0 | - | None -> B002, False | True/True | missing_key_field:product_type, no_confident_match:match |
| C19 | out_of_range | 0.0 | - | R013 -> R013, True | True/True | out_of_range:size_mm |
| C20 | messy | 0.0 | - | R006 -> R006, True | False/False | - |
| C21 | empty | 0.0 | - | None -> None, False | True/True | missing_key_field:product_type, missing_key_field:material, no_confident_match:match |
| C22 | conflict | 0.0 | - | None -> R015, False | True/True | no_confident_match:match |
| C23 | units | 0.0 | - | P003 -> P003, True | False/False | - |
| C24 | counting | 0.0 | - | P007 -> P007, True | False/False | - |
| C25 | synonyms | 0.0 | - | B003 -> B003, True | False/False | - |
| C26 | out_of_range | 0.0 | - | N001 -> N001, False | True/True | out_of_range:stone_count, no_confident_match:match |
| C27 | ambiguous | 0.0 | - | ['R003', 'R004'] -> R003, False | True/True | missing_key_field:material, no_confident_match:match |
