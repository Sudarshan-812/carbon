"""Image preprocessing for OCR: denoise, illumination, deskew, orientation.

See Prompt.md - Phase 2. Public entry point is :func:`preprocess`.
Nothing here hard-binarizes; :func:`binarize_for_tesseract` is the only
thresholding path and is used solely by the Tesseract OCR engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class PreprocessResult:
    image: np.ndarray
    rotation_applied: float = 0.0
    steps: list[str] = field(default_factory=list)
    debug: dict[str, np.ndarray] = field(default_factory=dict)


def preprocess(img: np.ndarray, cfg: Config) -> PreprocessResult:
    """Return a cleaned grayscale image ready for OCR.

    TODO(Phase 2): implement the toggleable pipeline:
      grayscale -> upscale -> denoise -> illumination_correction -> clahe
      -> deskew -> (optional) orientation_fix
    Each step gated by ``cfg.preprocess.<name>``; push intermediates into
    ``result.debug`` for ``--debug`` dumps.
    """
    raise NotImplementedError("Phase 2")


def binarize_for_tesseract(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Adaptive Gaussian threshold + light median blur. Tesseract path only.

    TODO(Phase 2).
    """
    raise NotImplementedError("Phase 2")


def _deskew_angle(gray: np.ndarray, cfg: Config) -> float:
    """Estimate skew via minAreaRect over the text mask. TODO(Phase 2)."""
    raise NotImplementedError("Phase 2")
