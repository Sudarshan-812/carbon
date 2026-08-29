"""Single-image orchestration: imread -> preprocess -> ocr -> extract -> score.

See Prompt.md - Phase 6. ``run_one`` never raises: on any internal failure it
returns a valid all-null :class:`ReceiptExtraction` with a ``pipeline_error`` flag.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from .confidence import score
from .config import Config
from .extract import extract_fields
from .ocr import get_engine
from .preprocess import preprocess
from .schema import ReceiptExtraction
from .utils import get_logger, imread_unicode, timer

log = get_logger(__name__)

PIPELINE_VERSION = "0.1.0"


def run_one(image_path: str | Path, cfg: Config) -> ReceiptExtraction:
    """Full pipeline for one receipt image. TODO(Phase 6): flesh out steps 1-2."""
    image_path = Path(image_path)
    image_id = image_path.stem
    try:
        with timer() as elapsed:
            img = imread_unicode(image_path)
            if img is None:
                raise ValueError("unreadable_image")
            pre = preprocess(img, cfg)
            ocr = get_engine(cfg).run(pre.image)
            raw = extract_fields(ocr, cfg)
            result = score(raw, ocr, cfg)
            result.meta.update(
                image_id=image_id,
                engine=ocr.engine,
                rotation_applied=pre.rotation_applied,
                preprocess_steps=pre.steps,
                mean_ocr_confidence=round(ocr.mean_conf, 3),
                n_lines=len(ocr.lines),
                elapsed_ms=round(elapsed(), 1),
                pipeline_version=PIPELINE_VERSION,
            )
        return result
    except Exception as exc:  # noqa: BLE001 - pipeline must not crash the batch
        log.warning("pipeline_error on %s: %s", image_id, exc)
        return ReceiptExtraction(
            flags=[f"pipeline_error:{type(exc).__name__}"],
            meta={
                "image_id": image_id,
                "pipeline_version": PIPELINE_VERSION,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=4),
            },
        )


def write_receipt_json(result: ReceiptExtraction, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.meta.get('image_id', 'unknown')}.json"
    path.write_text(json.dumps(result.to_output_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path
