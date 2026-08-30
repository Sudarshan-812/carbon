# Submission - Carbon Crunch ML Ops Internship Task

Receipt OCR + confidence-aware information extraction. Deterministic CPU-only
batch pipeline: **image → preprocess → EasyOCR → line reconstruction →
rule-based KIE → per-field confidence + flags → per-receipt JSON → aggregate
expense summary**. Python 3.13, no runtime network calls (EasyOCR downloads its
models once).

## One-command run

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:PYTHONUTF8 = "1"
python -m src.cli batch --input data\receipts --output outputs --engine easyocr
```

Produces `outputs/json/<image_id>.json` (one per receipt), plus
`outputs/expense_summary.json` / `.csv` and `outputs/run_report.md`.

---

## Rubric mapping

| # | Rubric item | Wt | Where it lives | Notes |
|---|---|---:|---|---|
| 1 | **Extraction** - store_name, date, items[] (name+price), total_amount | 30 | `src/extract.py`, tuned via `config.yaml` | Rule-based KIE: store = suffix/vendor-match then top-scored header line; date = dateutil `dayfirst=True` → ISO; total = tier-A/tier-B keyword ranking with `SUBTOTAL`/`GST` handling; items = row clustering with name/price columns. Every value carries a provenance record. |
| 2 | **Confidence** - OCR conf + pattern + keyword heuristic + items↔total cross-check; flag < 0.7; conflicts + missing handled | 20 | `src/confidence.py` | Weighted mean of up to 4 signals with renormalisation when a signal is N/A (weights table in `docs/writeup.md` §1 and `config.yaml`). `low_confidence_fields` (incl. granular `items[i].price`); flags for missing/fallback/conflicting-total/ambiguous-date/price-sum-mismatch/etc. Conflicting tier-A totals → `conflicting_total` flag, −0.15 penalty, rejects in `meta.alternatives`. |
| 3 | **Robustness** - noise/blur, skew/rotation, lighting/contrast | 15 | `src/preprocess.py`, `src/ocr.py` | Grayscale+upscale, fastNlMeans denoise, illumination flattening (divide by large-Gaussian background), CLAHE, `minAreaRect` deskew with clamp; low-confidence pages re-OCR'd at 90/180/270 and the best kept (`src/pipeline.py::_maybe_fix_orientation`). Post-OCR split-number merge (`"12"` + `".90"` → `"12.90"`). |
| 4 | **Structuring** - nested `{value, confidence}` JSON | 10 | `src/schema.py` (pydantic v2) | Every field `{ "value": ..., "confidence": 0.0-1.0 }`; also a flat `plain` block matching the spec's exact schema. Validated by `scripts/validate_json.py`. |
| 5 | **Summary logic** - total spend, # transactions, spend per store | 10 | `src/summary.py` | Confident totals only; fuzzy-merged vendor names (`rapidfuzz.token_set_ratio`) with an `UNKNOWN` bucket; date range; per-receipt exclusion list. Rebuildable standalone via `python -m src.cli summary`. |
| 6 | **Code quality** | 10 | `src/` (11 single-purpose modules), `config.yaml` | Type hints + docstrings throughout; every tunable in `config.yaml` (frozen pydantic tree that rejects unknown keys) - no magic numbers in code; deterministic (sorted iteration, fixed behaviour); `ruff` clean; `pytest` suite. |
| 7 | **Edge cases** | 5 | `src/pipeline.py` (`run_one` never raises), `tests/test_edge_cases.py` | Unreadable image → valid JSON with `unreadable_image` flag; zero OCR tokens → `ocr_empty`, all fields null; partial receipt → `total_from_fallback`; rotated → orientation retry; non-RM currency detected (`meta.currency_detected`); identical images de-duplicated in `batch`; over-long item lists → `items_truncated`. |

---

## Deliverables checklist

- [x] **Repo runnable from scratch** - verified in a clean venv (2026-08-30): `pip install -r requirements.txt` pulls torch/easyocr/opencv/pydantic/pandas, `python -m src.cli one data/receipts/X51005663279.jpg` returns correct nested + `plain` JSON; then `pip install -r requirements-dev.txt` + `pytest -q` = 132 passed, `ruff` clean. (`jupyter` was dropped from `requirements-dev.txt` - unused, and its `jedi` stub bundle tripped Windows' 260-char path limit on install.)
- [x] **Per-receipt JSON** for every image - 371 files, `scripts/validate_json.py` = **371 valid / 0 invalid**. Nested `{value, confidence}` **and** flat `plain` block. Schema: `src/schema.py`.
- [x] **`outputs/expense_summary.json` + `.csv`** - present. Full run: **MYR 22,797.96** across 338 / 367 receipts with a confident total; 29 excluded (missing / conf < 0.5); date range 2000 to 2025; spend-per-store fuzzy-merged. `outputs/run_report.md` has the coverage + flag histogram.
- [x] **`docs/writeup.pdf`** covering all four required sub-points - Approach, Tools used, Results, Challenges (+ Improvements). §3 carries the full-run coverage **and** the eval numbers (`eval/labels.csv` = 30 hand-labelled receipts): store fuzzy 70 %, date 80 %, total +/- 0.05 37 %, n_items +/- 1 77 %; calibration 0 / 44 / 66 / 89 % across confidence buckets. Source `docs/writeup.md`; regenerate the HTML with `python scripts/export_writeup.py`.
- [x] **Confidence** - low-conf fields flagged, conflicts + missing handled. Code: `src/confidence.py`. Real example: **`outputs/json/X51005433541.json`** - `conflicting_total` flag, chosen `58.20` @ conf 0.56, rejected tier-A `62.70` kept in `meta.alternatives`.
- [x] **Edge cases** - unreadable + rotated. Tests: `tests/test_edge_cases.py`. Real example JSONs: **`outputs/json/X51005433533.json`** (deskew 7.8 deg, still extracts store + total) and **`outputs/json/X51005268408.json`** (near-illegible thermal print, `low_mean_ocr_conf` + 3 more flags, everything low-confidence, nothing silently wrong).
- [x] **Tests pass; ruff clean** - `ruff check src tests scripts eval` clean; `pytest -q` = **132 passed, 1 deselected** (2026-08-30).
- [x] **`.gitignore`** excludes `.venv/`, `data/receipts/*`, bulk `outputs/`, `Prompt.md`, `extra/`.
- [x] **README** has a one-command run (`python -m src.cli batch ...`).

---

## Known limitations

- **No ground truth.** The dataset ships images only. Accuracy/calibration
  numbers come from a 30-receipt hand-labelled eval set (`eval/labels.csv`),
  not the full 371 - treat them as indicative.
- **Total-amount accuracy is the weak spot** (≈37 % exact on the eval set). GST
  receipts expose *Total (Excluding GST)* / GST-summary / rounding lines that the
  keyword ranker sometimes prefers over *Total Inclusive of GST*; fast-food
  bundle receipts tempt it toward the first item price. These misses are almost
  always flagged (`total_from_fallback`, sub-0.7 confidence), so they surface
  rather than pass silently - but the ranker rules need another pass.
- **Tesseract not installed** on the build machine, so the engine comparison
  (`eval/compare_engines.py`) ran EasyOCR only; the Tesseract path is
  implemented and interface-compatible but unbenchmarked here.
- **Currency.** Detection is wired (`meta.currency_detected`) but EasyOCR
  frequently drops the faint `$` / `RM` glyph, so the expense summary assumes a
  single currency (RM) rather than converting.
- **Rule-based KIE ceiling.** Unusual layouts (multi-column US chains, heavily
  faded thermal print) still mis-rank the total or miss items; these mostly
  surface as low confidence + flags rather than silent errors, but the ceiling
  is real. A supervised model (LayoutLMv3 / Donut) is the next step once labels
  exist - see `docs/writeup.md` §5.
- **CPU-only, ~15-25 s/image.** Fine for a 371-image batch (~2 h); not
  interactive. GPU EasyOCR would cut this ~5×.
- **`items_truncated`.** Very long receipts cap the item list (config
  `extract.max_items`) to keep JSON bounded; the flag records it.

---

## Packaging

The submission zip = everything tracked at `HEAD` plus the six generated
evidence files (`outputs/run_report.md`, `outputs/expense_summary.json` / `.csv`,
`eval/eval_report.md`, `eval/calibration.png`, `eval/engine_comparison.md`).
It excludes `.venv`, the `data/receipts` images, the 371 bulk `outputs/json`
files (5 samples kept), `extra/`, `Prompt.md`, and caches.

```powershell
git archive --format=zip -o ..\carbon-crunch-ocr_<name>.zip HEAD
# then add the 6 generated files above (see build step in the session notes)
```

Final artifact: `carbon-crunch-ocr_<name>.zip` for the Google Form.
