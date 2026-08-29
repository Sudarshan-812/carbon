"""Single-image orchestration: imread -> preprocess -> ocr -> extract -> score.

See Prompt.md - Phase 6. ``run_one`` never raises: on any internal failure it
returns a valid all-null :class:`ReceiptExtraction` carrying an explanatory flag.
"""
from __future__ import annotations

import json
import random
import traceback
from pathlib import Path

import numpy as np

from .confidence import score
from .config import Config
from .extract import VendorRegistry, extract_fields
from .ocr import get_engine
from .preprocess import dump_debug_images, preprocess
from .schema import ReceiptExtraction
from .utils import Timer, get_logger, imread_unicode

log = get_logger(__name__)

PIPELINE_VERSION = "0.1.0"


def _seed_everything(seed: int) -> None:
    """Pin every RNG so a re-run of the same image yields the same JSON."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch always present via easyocr
        pass


def run_one(
    image_path: str | Path,
    cfg: Config,
    *,
    vendors: VendorRegistry | None = None,
    debug: bool = False,
) -> ReceiptExtraction:
    """Full pipeline for one receipt image.

    Args:
        image_path: path to the receipt image.
        cfg: the loaded :class:`Config`.
        vendors: shared registry so store names are canonicalised across a batch.
        debug: also dump every preprocessing intermediate to ``paths.debug_dir``.
    """
    image_path = Path(image_path)
    image_id = image_path.stem
    try:
        with Timer() as elapsed:
            img = imread_unicode(image_path)
            if img is None:
                return _empty_result(image_id, cfg, ["unreadable_image"])

            pre = preprocess(img, cfg)
            if debug:
                dump_debug_images(pre, cfg.paths.debug_dir, image_id)

            _seed_everything(cfg.run.seed)
            ocr = get_engine(cfg).run(pre.image)
            result = score(extract_fields(ocr, cfg, vendors), ocr, cfg)
            if not ocr.tokens:
                result.flags.insert(0, "ocr_empty")

            result.meta.update(
                image_id=image_id,
                engine=ocr.engine,
                rotation_applied=pre.rotation_applied,
                preprocess_steps=pre.steps,
                mean_ocr_confidence=round(ocr.mean_conf, 4),
                n_lines=len(ocr.lines),
                n_tokens=len(ocr.tokens),
                elapsed_ms=round(elapsed(), 1),
                pipeline_version=cfg.run.pipeline_version,
            )
        return result
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
        log.warning("pipeline_error on %s: %s", image_id, exc)
        log.debug("traceback for %s:\n%s", image_id, traceback.format_exc())
        result = _empty_result(image_id, cfg, [f"pipeline_error:{type(exc).__name__}"])
        result.meta["error"] = str(exc)
        result.meta["traceback"] = traceback.format_exc(limit=4)
        return result


def _empty_result(image_id: str, cfg: Config, flags: list[str]) -> ReceiptExtraction:
    """A schema-valid result with every field null / zero-confidence."""
    return ReceiptExtraction(
        flags=list(flags),
        low_confidence_fields=["store_name", "date", "total_amount"],
        meta={"image_id": image_id, "pipeline_version": cfg.run.pipeline_version},
    )


def write_receipt_json(result: ReceiptExtraction, out_dir: str | Path) -> Path:
    """Write ``<out_dir>/<image_id>.json`` (nested form + embedded ``plain``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.meta.get('image_id', 'unknown')}.json"
    path.write_text(
        json.dumps(result.to_output_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
