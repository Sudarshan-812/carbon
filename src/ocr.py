"""OCR engine wrappers + engine-agnostic line reconstruction.

See Prompt.md - Phase 3. EasyOCR is the primary engine; Tesseract is kept for
the comparison experiment (Phase 12). Both return the same :class:`OcrResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import Config
from .utils import get_logger

log = get_logger(__name__)

BBox = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Token:
    text: str
    conf: float
    box: BBox


@dataclass
class Line:
    text: str
    conf: float
    box: BBox
    tokens: list[Token] = field(default_factory=list)


@dataclass
class OcrResult:
    tokens: list[Token]
    lines: list[Line]
    full_text: str
    mean_conf: float
    engine: str


class OcrEngine(Protocol):
    def run(self, img: np.ndarray) -> OcrResult: ...


class EasyOcrEngine:
    """Wraps :class:`easyocr.Reader`. Reader is created once and cached.

    TODO(Phase 3): lazy-init a module-level Reader per language set (gpu=False),
    map ``readtext(detail=1)`` output to :class:`Token`, then call
    :func:`reconstruct_lines`.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, img: np.ndarray) -> OcrResult:  # pragma: no cover - Phase 3
        raise NotImplementedError("Phase 3")


class TesseractEngine:
    """Wraps ``pytesseract.image_to_data``. Applies ``binarize_for_tesseract``.

    TODO(Phase 3).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, img: np.ndarray) -> OcrResult:  # pragma: no cover - Phase 3
        raise NotImplementedError("Phase 3")


def get_engine(cfg: Config) -> OcrEngine:
    name = str(cfg.ocr.engine).lower()
    if name == "easyocr":
        return EasyOcrEngine(cfg)
    if name == "tesseract":
        return TesseractEngine(cfg)
    raise ValueError(f"unknown ocr.engine: {name!r}")


def reconstruct_lines(tokens: list[Token], cfg: Config) -> list[Line]:
    """Cluster tokens into rows by vertical proximity, sort L->R within a row.

    TODO(Phase 3): tolerance = median token height * y_tolerance_factor.
    """
    raise NotImplementedError("Phase 3")
