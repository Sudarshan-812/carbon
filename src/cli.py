"""Command-line entry point.

See Prompt.md - Phase 8.

    python -m src.cli batch   --input data/receipts --output outputs [--engine ...] [--limit N] [--workers K] [--debug]
    python -m src.cli one     <image> [--engine ...] [--debug]
    python -m src.cli summary --json-dir outputs/json --output outputs
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .config import Config, load_config
from .schema import ReceiptExtraction
from .utils import ensure_utf8_stdout, get_logger

log = get_logger("cli")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ERROR_ABORT_FRACTION = 0.20


# --- shared helpers ---------------------------------------------

def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Fold CLI ``--engine`` / ``--input`` / ``--output`` / ``--json-dir`` into *cfg*."""
    updates: dict[str, object] = {}
    if getattr(args, "engine", None):
        updates["ocr"] = cfg.ocr.model_copy(update={"engine": args.engine})

    paths: dict[str, str] = {}
    if getattr(args, "input", None):
        paths["input_dir"] = args.input
    if getattr(args, "output", None):
        paths["output_dir"] = args.output
        paths["json_dir"] = str(Path(args.output) / "json")
        paths["debug_dir"] = str(Path(args.output) / "debug")
    if getattr(args, "json_dir", None):
        paths["json_dir"] = args.json_dir
    if paths:
        updates["paths"] = cfg.paths.model_copy(update=paths)

    return cfg.model_copy(update=updates) if updates else cfg


def _config_hash(cfg: Config) -> str:
    return hashlib.sha1(cfg.model_dump_json().encode("utf-8")).hexdigest()[:12]


def _iter_images(input_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _is_error(result: ReceiptExtraction) -> bool:
    return any(f.startswith("pipeline_error") or f == "unreadable_image"
              for f in result.flags)


# --- batch --------------------------------------------------

def _cmd_batch(args: argparse.Namespace) -> int:
    ensure_utf8_stdout()
    cfg = _apply_overrides(load_config(args.config), args)

    from .extract import VendorRegistry
    from .pipeline import run_one, write_receipt_json
    from .summary import build_summary, write_summary

    input_dir = Path(cfg.paths.input_dir)
    if not input_dir.is_dir():
        log.error("input directory not found: %s", input_dir)
        return 2
    images = _iter_images(input_dir, args.limit)
    if not images:
        log.error("no %s images in %s", "/".join(sorted(IMAGE_EXTS)), input_dir)
        return 2

    json_dir = Path(cfg.paths.json_dir)
    workers = max(1, args.workers if args.workers is not None else cfg.run.workers)

    # De-duplicate byte-identical images: process one per sha1 group, copy the
    # result to the others' JSON files (Prompt.md - Phase 9, edge case 6).
    groups: dict[str, list[Path]] = {}
    for path in images:
        groups.setdefault(_sha1(path), []).append(path)
    reps = [paths[0] for paths in groups.values()]
    dup_paths = {paths[0].stem: paths[1:] for paths in groups.values() if len(paths) > 1}
    n_duplicates = sum(len(v) for v in dup_paths.values())
    log.info("processing %d unique images (%d duplicates) engine=%s workers=%d -> %s",
             len(reps), n_duplicates, cfg.ocr.engine, workers, json_dir)

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(iterable, **_kwargs):
            return iterable

    started = time.perf_counter()
    results: list[ReceiptExtraction] = []

    if workers == 1:
        vendors = VendorRegistry(cfg.extract.store_name.vendor_merge_ratio)
        for path in tqdm(reps, desc="receipts", unit="img"):
            result = run_one(path, cfg, vendors=vendors, debug=args.debug)
            write_receipt_json(result, json_dir)
            results.append(result)
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, path, cfg, debug=args.debug) for path in reps]
            for future in tqdm(cf.as_completed(futures), total=len(futures),
                               desc="receipts", unit="img"):
                result = future.result()
                write_receipt_json(result, json_dir)
                results.append(result)
        results.sort(key=lambda r: str(r.meta.get("image_id", "")))

    for result in results:  # emit a JSON for every duplicate too
        for dup in dup_paths.get(result.meta.get("image_id", ""), []):
            clone = result.model_copy(deep=True)
            clone.meta = {**result.meta, "image_id": dup.stem,
                          "duplicate_of": result.meta.get("image_id")}
            write_receipt_json(clone, json_dir)

    wall = time.perf_counter() - started
    failed = [r for r in results if _is_error(r)]
    _write_run_report(Path(cfg.paths.output_dir) / "run_report.md", cfg, results, wall,
                      n_duplicates=n_duplicates)
    write_summary(build_summary(results, cfg), cfg.paths.output_dir)

    rate = len(results) / wall if wall > 0 else 0.0
    print(f"\n{len(results)} processed | {len(results) - len(failed)} ok | "
          f"{len(failed)} failed | {n_duplicates} duplicate(s) | {wall:.1f}s | {rate:.2f} img/s")
    print(f"JSON -> {json_dir}   report -> {Path(cfg.paths.output_dir) / 'run_report.md'}")

    if results and len(failed) / len(results) > ERROR_ABORT_FRACTION:
        log.error("%d/%d images errored (> %.0f%%)", len(failed), len(results),
                  ERROR_ABORT_FRACTION * 100)
        return 1
    return 0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _write_run_report(
    path: Path, cfg: Config, results: list[ReceiptExtraction], wall: float,
    *, n_duplicates: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    failed = [r for r in results if _is_error(r)]
    rate = total / wall if wall > 0 else 0.0

    field_conf = {
        "store_name": _mean([r.store_name.confidence for r in results if r.store_name.value]),
        "date": _mean([r.date.confidence for r in results if r.date.value]),
        "total_amount": _mean([r.total_amount.confidence for r in results if r.total_amount.value]),
        "item_name": _mean([it.name.confidence for r in results for it in r.items if it.name.value]),
        "item_price": _mean([it.price.confidence for r in results for it in r.items if it.price.value]),
    }
    coverage = {
        "store_name": sum(1 for r in results if r.store_name.value),
        "date": sum(1 for r in results if r.date.value),
        "total_amount": sum(1 for r in results if r.total_amount.value),
        "items": sum(1 for r in results if r.items),
    }
    low_counts: Counter[str] = Counter()
    for r in results:
        for field in r.low_confidence_fields:
            low_counts[re.sub(r"items\[\d+\]", "items[]", field)] += 1
    flag_counts = Counter(f for r in results for f in r.flags)

    def _table(header: str, rows: list[tuple[str, str]]) -> list[str]:
        body = rows or [("_none_", "0")]
        return [f"| {header} | Count |", "| --- | ---: |",
                *(f"| {k} | {v} |" for k, v in body), ""]

    header = (
        f"- Engine: **{cfg.ocr.engine}**   Config hash: `{_config_hash(cfg)}`   "
        f"Pipeline: `{cfg.run.pipeline_version}`\n"
        f"- Images: **{total}** unique   succeeded: **{total - len(failed)}**   "
        f"failed: **{len(failed)}**   duplicates: **{n_duplicates}**\n"
        f"- Wall time: **{wall:.1f}s**   throughput: **{rate:.2f} img/s**"
    )
    lines = [
        "# Run report", "",
        header, "",
        "## Coverage (non-null values)", "",
        "| Field | Count | % |", "| --- | ---: | ---: |",
        *(f"| {k} | {v} | {100 * v / max(total, 1):.1f}% |" for k, v in coverage.items()),
        "",
        "## Mean confidence per field (over non-null values)", "",
        "| Field | Mean confidence |", "| --- | ---: |",
        *(f"| {k} | {v:.3f} |" for k, v in field_conf.items()),
        "",
        "## Low-confidence fields (count by type)", "",
        *_table("Field", [(k, str(v)) for k, v in low_counts.most_common()]),
        "## Flags", "",
        *_table("Flag", [(k, str(v)) for k, v in flag_counts.most_common()]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- one ---------------------------------------------------

def _cmd_one(args: argparse.Namespace) -> int:
    ensure_utf8_stdout()
    cfg = _apply_overrides(load_config(args.config), args)
    from .pipeline import run_one, write_receipt_json

    result = run_one(args.image, cfg, debug=args.debug)
    path = write_receipt_json(result, cfg.paths.json_dir)
    print(path.read_text(encoding="utf-8"))
    return 1 if _is_error(result) else 0


# --- summary ---------------------------------------------

def _cmd_summary(args: argparse.Namespace) -> int:
    ensure_utf8_stdout()
    cfg = _apply_overrides(load_config(args.config), args)
    from .summary import build_summary, load_results_from_json, write_summary

    json_dir = Path(args.json_dir) if args.json_dir else Path(cfg.paths.json_dir)
    out_dir = Path(args.output) if args.output else Path(cfg.paths.output_dir)
    results = load_results_from_json(json_dir)
    if not results:
        log.error("no JSON files in %s", json_dir)
        return 2

    summary = build_summary(results, cfg)
    write_summary(summary, out_dir)
    print(summary.model_dump_json(indent=2))
    return 0


# --- parser ---------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="carbon-crunch-ocr")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("batch", help="process a folder of receipts")
    b.add_argument("--input", default=None)
    b.add_argument("--output", default=None)
    b.add_argument("--engine", choices=["easyocr", "tesseract"], default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--debug", action="store_true")
    b.set_defaults(func=_cmd_batch)

    o = sub.add_parser("one", help="process a single image")
    o.add_argument("image")
    o.add_argument("--engine", choices=["easyocr", "tesseract"], default=None)
    o.add_argument("--debug", action="store_true")
    o.set_defaults(func=_cmd_one)

    s = sub.add_parser("summary", help="(re)build the expense summary from JSON")
    s.add_argument("--json-dir", dest="json_dir", default=None)
    s.add_argument("--output", default=None)
    s.set_defaults(func=_cmd_summary)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
