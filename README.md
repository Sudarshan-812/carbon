# Carbon

Deterministic batch pipeline that turns receipt images into structured,
**confidence-scored** JSON plus an aggregate expense summary.

Named for the carbon copy: the duplicate a receipt used to leave behind.

```
image → preprocess → OCR (EasyOCR) → line reconstruction
      → key-info extraction (store / date / items / total)
      → per-field confidence + reliability flags → outputs/json/<id>.json
      → aggregate → outputs/expense_summary.json + .csv + run_report.md
```

Every field lands as `{ "value": ..., "confidence": 0.0-1.0 }`; fields below
**0.7** are flagged. All tunables (keyword lists, thresholds, weights) live in
`config.yaml` — no magic numbers in code.

The point of the project is not the OCR. It is that the confidence numbers
mean something: see [Evaluation](#evaluation) for the calibration harness that
checks them against hand-labelled ground truth.

---

## Setup

Requires **Python 3.13**. The Tesseract binary is optional (comparison engine only).

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + dev (pytest, ruff, matplotlib)

# optional, for `--engine tesseract` / the engine comparison:
brew install tesseract                   # or: apt install tesseract-ocr
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# optional:
winget install --id UB-Mannheim.TesseractOCR -e
#   then set  ocr.tesseract_cmd  in config.yaml if it is not on PATH
```

If Tesseract is not on `PATH`, set `ocr.tesseract_cmd` in `config.yaml`.

The first EasyOCR call downloads its models (~100 MB) once. On Windows set
`PYTHONUTF8=1` — EasyOCR's progress bar breaks the cp1252 console.

## Data

The pipeline was developed against a 371-image receipt dataset. Put the images
flat in `data/receipts/` (git-ignored):

```bash
unzip receipts.zip -d tmp && mv tmp/*/* data/receipts/
```

```powershell
Expand-Archive receipts.zip -DestinationPath tmp
Move-Item "tmp\*\*" data\receipts\
```

## Run

```bash
# one image (prints the JSON)
python -m src.cli one data/receipts/X51005763964.jpg

# whole folder → per-receipt JSON + summary + run_report.md
python -m src.cli batch --input data/receipts --output outputs --engine easyocr
#   [--limit N] [--workers K] [--debug] [--skip-existing]
#   --debug dumps preprocess steps; --skip-existing reuses <id>.json already
#   in outputs/json (resume a killed run without re-OCR'ing)

# rebuild only the summary from existing JSON
python -m src.cli summary --json-dir outputs/json --output outputs
```

Batch exits non-zero if more than 20% of images raise `pipeline_error`, so it
can be dropped into a scheduler without a human watching it.

Dev shortcuts: `make <task>` (or `.\tasks.ps1 <task>` on Windows) —
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
             "total_amount": "63.00" }          // flat mirror for consumers that
                                                // don't want the confidence wrappers
}
```

`outputs/expense_summary.json` / `.csv` — total spend (confident totals only),
transaction count, spend per store (fuzzy-merged vendors, `UNKNOWN` bucket),
date range, per-receipt exclusions.

`outputs/run_report.md` — coverage %, mean confidence per field, low-confidence
counts by type, flag histogram, wall time, throughput, config hash.

## Evaluation

A confidence score nobody checks is decoration. This is the part of the repo
that checks it.

The dataset ships without labels, so `eval/labels.csv` holds 30 evenly-sampled
`image_id`s filled in by hand (`store_name,date,total_amount,n_items`; blank
cells are skipped per-field).

```bash
python eval/evaluate.py         # → eval/eval_report.md + eval/calibration.png
python eval/compare_engines.py  # → eval/engine_comparison.md
```

`evaluate.py` reports:

- **Per-field accuracy** — store exact and fuzzy (≥90), date exact,
  total to 2dp and within ±0.05, item count exact and ±1
- **Dataset coverage** — how much of the corpus produced a usable extraction
- **Calibration** — a confidence-bucket table plus `calibration.png`, showing
  whether a field scored 0.9 is actually right about 90% of the time

`compare_engines.py` runs an engine and preprocessing ablation (EasyOCR vs
Tesseract, with and without individual preprocess stages) so the default
configuration in `config.yaml` is justified by measurement rather than
preference.

## Known limitations

- **The eval set is 30 hand-labelled images out of 371.** It is large enough to
  show whether confidence is directionally calibrated, and too small to put a
  tight interval on any per-field accuracy number. Treat the accuracy figures
  as indicative.
- **Extraction is rule-based, not learned.** It generalises to receipt layouts
  close to those in the dataset and will degrade on materially different ones.
  The tunables are in `config.yaml` precisely so this is adjustable without a
  code change.
- **Confidence is a heuristic composite**, not a probability from a trained
  model. The calibration chart is what makes it trustworthy; without rerunning
  that on a new corpus, the 0.7 threshold should not be assumed to transfer.
- **Single-machine batch only.** No queue, no retry across hosts, no
  persistence beyond the output directory.

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
| `eval/` | label template + accuracy / calibration / engine-comparison harness |
| `scripts/` | `coverage_report`, `validate_json`, `rank_outputs` |
| `docs/writeup.md` | design writeup |

## Tests

```bash
pytest -q                       # 132 tests; add -m "" for the slow end-to-end
ruff check src tests scripts eval
python scripts/coverage_report.py
```
