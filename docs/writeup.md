# Receipt OCR & Confidence-Aware Extraction — Writeup

> 1–2 pages. Fill in during Phase 14. Keep each section tight.

## 1. Approach
- Pipeline: `preprocess → OCR (EasyOCR) → line reconstruction → rule-based key
  information extraction → per-field confidence → JSON → aggregate summary`.
- Why rule-based KIE: the dataset ships **no labels**, so a supervised KIE model
  (LayoutLM/Donut) can't be trained in the time budget. Heuristics tuned on a
  representative sample + a hand-labelled eval set.
- Confidence model: `conf = w_ocr·ocr + w_pattern·pattern + w_heuristic·heuristic
  (+ w_cross·cross_check)`. Weights table:

  | field | ocr | pattern | heuristic | cross_check |
  |-------|-----|---------|-----------|-------------|
  | store_name | 0.35 | 0.20 | 0.45 | — |
  | date | 0.35 | 0.40 | 0.25 | — |
  | total_amount | 0.35 | 0.20 | 0.20 | 0.25 |

## 2. Tools used
| Tool | Why |
|------|-----|
| OpenCV | denoise, illumination flatten, deskew |
| EasyOCR | primary OCR, per-box confidence, robust on noisy receipts |
| Tesseract (pytesseract) | comparison baseline / OSD orientation |
| rapidfuzz | vendor-name normalization & fuzzy accuracy |
| python-dateutil | tolerant date parsing → ISO |
| pydantic | output schema validation |
| pandas / matplotlib | summary CSV, calibration chart |

## 3. Results
- Coverage per field (non-null over 371): store __%, date __%, total __%, items __%.
- Eval-set (n=__) accuracy: store __, date __, total(2dp) __ / (±0.05) __, n_items __.
- Calibration: see `eval/calibration.png` — accuracy rises with confidence bucket.
- Engine comparison takeaway: __ (see `eval/engine_comparison.md`).

## 4. Challenges faced
- Layout diversity across vendors; `SUBTOTAL` vs `GRAND TOTAL` disambiguation.
- Uneven lighting / thermal-print fade; skew on phone photos.
- Ambiguous `dd/mm` vs `mm/dd` dates.
- No ground truth → had to build an eval set by hand.

## 5. Improvements
- LayoutLMv3 / Donut for KIE once labels exist; active-learning loop.
- Per-vendor templates keyed off the detected store name.
- Learned confidence calibrator (isotonic / Platt) on the eval set.
- Fine-tune the recognizer on receipt fonts; try PaddleOCR.
