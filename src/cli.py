"""Command-line entry point.

See Prompt.md - Phase 8.

    python -m src.cli batch   --input data/receipts --output outputs [--engine ...] [--limit N] [--debug]
    python -m src.cli one     <image> [--debug]
    python -m src.cli summary --json-dir outputs/json --output outputs
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import get_logger

log = get_logger("cli")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _cmd_one(args: argparse.Namespace) -> int:
    from .pipeline import run_one, write_receipt_json

    cfg = load_config(args.config)
    result = run_one(args.image, cfg)
    path = write_receipt_json(result, cfg.paths.json_dir)
    print(path.read_text(encoding="utf-8"))
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    """TODO(Phase 8): glob (sorted), tqdm, per-image try/except, then summary +
    run_report.md. Exit non-zero if >20% of images errored."""
    raise NotImplementedError("Phase 8")


def _cmd_summary(args: argparse.Namespace) -> int:
    """TODO(Phase 8): load outputs/json/*.json -> build_summary -> write_summary."""
    raise NotImplementedError("Phase 8")


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
    o.add_argument("--debug", action="store_true")
    o.set_defaults(func=_cmd_one)

    s = sub.add_parser("summary", help="(re)build the expense summary from JSON")
    s.add_argument("--json-dir", default=None)
    s.add_argument("--output", default=None)
    s.set_defaults(func=_cmd_summary)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
