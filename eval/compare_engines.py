"""OCR engine comparison + preprocessing ablation over the eval set.

See Prompt.md - Phase 12. Feeds the "Model Development" section of the writeup.

    python eval/compare_engines.py [--limit N] [--engines easyocr,tesseract]

For each available engine it runs ``run_one`` over the eval-set image_ids
(``eval/labels.csv``) and reports seconds/image, mean OCR confidence, mean field
confidence, and - where ``labels.csv`` is filled - field accuracy. It then runs
a preprocess ON vs OFF ablation for the best engine. Tesseract rows are skipped
(with a note) when its binary is not installed.
"""
from __future__ import annotations

# ruff: noqa: I001  (sys.path shims must sit between stdlib and local imports)
import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from evaluate import _load_labels, _match_date, _match_store, _match_total
from src.config import Config, load_config
from src.pipeline import run_one

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class EngineRun:
    label: str
    n: int = 0
    errors: int = 0
    sec_per_image: float = 0.0
    mean_ocr_conf: float = 0.0
    mean_field_conf: float = 0.0
    acc_store: float | None = None
    acc_date: float | None = None
    acc_total: float | None = None
    acc_items: float | None = None

    @property
    def pooled_accuracy(self) -> float | None:
        accs = [a for a in (self.acc_store, self.acc_date, self.acc_total) if a is not None]
        return statistics.fmean(accs) if accs else None


def _resolve(image_id: str, receipts_dir: Path) -> Path | None:
    for match in sorted(receipts_dir.glob(f"{image_id}.*")):
        if match.is_file() and match.suffix.lower() in _IMAGE_EXTS:
            return match
    return None


def _tesseract_available(cfg: Config) -> bool:
    try:
        import pytesseract

        if cfg.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.ocr.tesseract_cmd
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def _cfg_for(base: Config, engine: str, *, preprocess_on: bool) -> Config:
    return base.model_copy(update={
        "ocr": base.ocr.model_copy(update={"engine": engine}),
        "preprocess": base.preprocess.model_copy(update={"enabled": preprocess_on}),
    })


def _run_set(
    base: Config, engine: str, *, preprocess_on: bool, paths: list[Path],
    labels: dict[str, dict], label: str,
) -> EngineRun:
    cfg = _cfg_for(base, engine, preprocess_on=preprocess_on)
    times: list[float] = []
    ocr_confs: list[float] = []
    field_confs: list[float] = []
    hits: dict[str, list[bool]] = {"store": [], "date": [], "total": [], "items": []}
    errors = 0

    for path in paths:
        result = run_one(path, cfg)
        if any(f.startswith("pipeline_error") or f == "unreadable_image"
               for f in result.flags):
            errors += 1
            continue
        times.append(result.meta.get("elapsed_ms", 0.0) / 1000.0)
        ocr_confs.append(result.meta.get("mean_ocr_confidence", 0.0))
        for fc in (result.store_name, result.date, result.total_amount):
            if fc.value is not None:
                field_confs.append(fc.confidence)

        gold = labels.get(path.stem, {})
        if gold.get("store_name"):
            hits["store"].append(_match_store(result.store_name.value, gold["store_name"])[1])
        if gold.get("date"):
            hits["date"].append(_match_date(result.date.value, gold["date"]))
        if gold.get("total_amount"):
            hits["total"].append(_match_total(result.total_amount.value, gold["total_amount"])[1])
        if gold.get("n_items"):
            hits["items"].append(len(result.items) == int(gold["n_items"]))

    def acc(key: str) -> float | None:
        return statistics.fmean(hits[key]) if hits[key] else None

    return EngineRun(
        label=label, n=len(times), errors=errors,
        sec_per_image=statistics.fmean(times) if times else 0.0,
        mean_ocr_conf=statistics.fmean(ocr_confs) if ocr_confs else 0.0,
        mean_field_conf=statistics.fmean(field_confs) if field_confs else 0.0,
        acc_store=acc("store"), acc_date=acc("date"),
        acc_total=acc("total"), acc_items=acc("items"),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _table(runs: list[EngineRun]) -> list[str]:
    rows = [
        ("| Variant | n | s/img | mean OCR conf | mean field conf | "
         "store | date | total | n_items | errors |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        rows.append(
            f"| {r.label} | {r.n} | {r.sec_per_image:.1f} | {r.mean_ocr_conf:.3f} | "
            f"{r.mean_field_conf:.3f} | {_fmt(r.acc_store)} | {_fmt(r.acc_date)} | "
            f"{_fmt(r.acc_total)} | {_fmt(r.acc_items)} | {r.errors} |"
        )
    return rows


def _takeaway(engine_runs: list[EngineRun], pp_on: EngineRun, pp_off: EngineRun,
              winner: str, labelled: bool) -> str:
    if len(engine_runs) >= 2:
        a, b = engine_runs[0], engine_runs[1]
        s1 = (f"**{winner}** gave the higher mean OCR confidence "
              f"({a.mean_ocr_conf:.2f} vs {b.mean_ocr_conf:.2f}) at "
              f"{a.sec_per_image:.1f}s/image against {b.sec_per_image:.1f}s, so it is the "
              "pipeline default.")
    else:
        s1 = (f"Only **{winner}** was available (the Tesseract binary is not installed), "
              "so it is used as the pipeline default; install it and re-run for the "
              "full head-to-head.")
    delta = pp_on.mean_ocr_conf - pp_off.mean_ocr_conf
    verb = "raised" if delta >= 0 else "lowered"
    s2 = (f"Turning the preprocessing pipeline on {verb} mean OCR confidence by "
          f"{abs(delta):.2f} ({pp_off.mean_ocr_conf:.2f} to {pp_on.mean_ocr_conf:.2f}) "
          f"and cost {pp_on.sec_per_image - pp_off.sec_per_image:+.1f}s/image.")
    s3 = ("Field-accuracy columns are populated from eval/labels.csv." if labelled
          else "Field-accuracy columns are blank until eval/labels.csv is filled in.")
    return f"{s1} {s2} {s3}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap eval images (default: all 30)")
    ap.add_argument("--engines", default="easyocr,tesseract")
    ap.add_argument("--labels", type=Path, default=_HERE / "labels.csv")
    ap.add_argument("--out", type=Path, default=_HERE / "engine_comparison.md")
    args = ap.parse_args()

    base = load_config()
    receipts_dir = Path(base.paths.input_dir)
    labels = {row["image_id"]: row for row in _load_labels(args.labels)}
    labelled = any(any(r.get(k) for k in ("store_name", "date", "total_amount", "n_items"))
                   for r in labels.values())

    ids = list(labels)[: args.limit] if args.limit else list(labels)
    paths = [p for p in (_resolve(i, receipts_dir) for i in ids) if p is not None]
    if not paths:
        print("no eval images resolved under", receipts_dir, file=sys.stderr)
        return 2

    requested = [e.strip() for e in args.engines.split(",") if e.strip()]
    available = [e for e in requested
                 if e == "easyocr" or (e == "tesseract" and _tesseract_available(base))]
    skipped = [e for e in requested if e not in available]

    print(f"comparing {available} over {len(paths)} images "
          f"({'labels present' if labelled else 'no labels yet'})...")

    engine_runs = [
        _run_set(base, e, preprocess_on=True, paths=paths, labels=labels, label=e)
        for e in available
    ]
    engine_runs.sort(key=lambda r: (r.pooled_accuracy or -1, r.mean_ocr_conf), reverse=True)
    winner = engine_runs[0].label

    pp_on = next(r for r in engine_runs if r.label == winner)
    pp_off = _run_set(base, winner, preprocess_on=False, paths=paths, labels=labels,
                      label=f"{winner} (no preprocess)")

    skipped_note = "" if not skipped else f"   |   skipped (not installed): {', '.join(skipped)}"
    lines = [
        "# OCR engine comparison", "",
        f"Eval set: **{len(paths)}** images{skipped_note}",
        "", "## Engines (preprocessing on)", "",
        *_table(engine_runs), "",
        f"## Preprocessing ablation: {winner}", "",
        *_table([pp_on, pp_off]), "",
        "## Takeaway", "",
        _takeaway(engine_runs, pp_on, pp_off, winner, labelled), "",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
