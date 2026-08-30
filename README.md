# Carbon Crunch - Receipt OCR & Confidence-Aware Extraction

Deterministic batch pipeline that turns receipt images into structured,
**confidence-scored** JSON plus an aggregate expense summary. Built for the
Carbon Crunch ML Ops internship assignment.

```
image → preprocess → OCR (EasyOCR) → line reconstruction
      → key-info extraction (store / date / items / total)
      → per-field confidence + reliability flags → outputs/json/<id>.json
      → aggregate → outputs/expense_summary.json + .csv + run_report.md
```

Every field lands as `{ "value": ..., "confidence": 0.0-1.0 }`; fields below
**0.7** are flagged. All tunables (keyword lists, thresholds, weights) live in
`config.yaml` - no magic numbers in code.

## Setup

Requires **Python 3.13**. The Tesseract binary is optional (comparison engine only).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt      # runtime + dev (pytest, ruff, matplotlib)

# optional, for `--engine tesseract` / the engine comparison:
winget install --id UB-Mannheim.TesseractOCR -e
#   then set  ocr.tesseract_cmd  in config.yaml if it is not on PATH
```

The first EasyOCR call downloads its models (~100 MB) once. On Windows set
`PYTHONUTF8=1` (EasyOCR's progress bar breaks the cp1252 console).

## Data

Put the 371 receipt images flat in `data/receipts/` (git-ignored):

```powershell
Expand-Archive "AI-OCR dataset-....zip" -DestinationPath tmp
Move-Item "tmp\AI-OCR dataset\*" data\receipts\
```

## Run

```powershell
# one image (prints the JSON)
python -m src.cli one data\receipts\X51005763964.jpg

# whole folder → per-receipt JSON + summary + run_report.md
python -m src.cli batch --input data\receipts --output outputs --engine easyocr
#   [--limit N] [--workers K] [--debug] [--skip-existing]
#   --debug dumps preprocess steps; --skip-existing reuses <id>.json already
#   in outputs/json (resume a killed run without re-OCR'ing)

# rebuild only the summary from existing JSON
python -m src.cli summary --json-dir outputs\json --output outputs
```

Batch exits non-zero if > 20 % of images raise `pipeline_error`.

Dev shortcuts: `.\tasks.ps1 <task>` (Windows) or `make <task>` -
`check` (ruff + pytest), `test-all` (incl. the slow end-to-end),
`batch`, `summary`, `eval`, `compare`, `validate`, `rank`.

## Output

`outputs/json/<image_id>.json`

```jsonc
{
  "store_name":  { "value": "SATU KAMPUNG ENTERPRISE SDN BHD", "confidence": 0.94 },
  "date":        { "value": "2018-01-13", "confidence": 0.71 },
  "items":       [ { "name": {"value": "...", "confidence": 0.8},
                     "price": {"value": "5.90", "confidence": 0.9}, "qty": 2 } ],
  "total_amount": { "value": "63.00", "confidence": 0.94 },
  "low_confidence_fields": ["date"],
  "flags": ["date_order_ambiguous"],
  "meta": { "image_id": "...", "engine": "easyocr", "rotation_applied": 0.0,
            "mean_ocr_confidence": 0.73, "elapsed_ms": 10733, "pipeline_version": "0.1.0" },
  "plain": { "store_name": "...", "date": "...", "items": [{"name":"...","price":"..."}],
             "total_amount": "63.00" }          // exact PDF §4 schema
}
```

`outputs/expense_summary.json` / `.csv` - total spend (confident totals only),
transaction count, spend per store (fuzzy-merged vendors, `UNKNOWN` bucket),
date range, per-receipt exclusions.

`outputs/run_report.md` - coverage %, mean confidence per field, low-confidence
counts by type, flag histogram, wall time, throughput, config hash.

## Evaluation

No labels ship with the dataset, so `eval/labels.csv` holds 30 evenly-sampled
`image_id`s to fill by hand (`store_name,date,total_amount,n_items`; blank cells
are skipped per-field).

```powershell
python eval/evaluate.py         # → eval/eval_report.md + eval/calibration.png
python eval/compare_engines.py  # → eval/engine_comparison.md (engine + preprocess ablation)
```

`evaluate.py` reports per-field accuracy (store exact/fuzzy≥90, date exact,
total 2dp/±0.05, n_items exact/±1), dataset coverage, and a confidence-bucket
calibration table + chart.

## Repo map

| Path | What |
|---|---|
| `src/preprocess.py` | grayscale, denoise, illumination, CLAHE, deskew, orientation retry |
| `src/ocr.py` | EasyOCR / Tesseract wrappers, line reconstruction, split-number merge |
| `src/extract.py` | rule-based KIE (store / date / total / items) + provenance |
| `src/confidence.py` | field-level confidence + reliability flags |
| `src/schema.py` | pydantic output + summary models |
| `src/summary.py` | expense-summary aggregation + writers |
| `src/pipeline.py` | single-image orchestration (`run_one`, never raises) |
| `src/cli.py` | `batch` / `one` / `summary`, dedup, `run_report.md` |
| `src/config.py` | frozen pydantic config tree (rejects unknown keys) |
| `config.yaml` | every tunable |
| `eval/` | label template + accuracy/calibration/engine-comparison harness |
| `scripts/` | `coverage_report`, `validate_json`, `rank_outputs` |
| `docs/writeup.md` | the 1-2 page deliverable |

## Tests

```powershell
pytest -q                       # 132 tests; add -m "" for the slow end-to-end
ruff check src tests scripts eval
python scripts/coverage_report.py
```
