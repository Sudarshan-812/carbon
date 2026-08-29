"""Preprocessing tests. See Prompt.md - Phase 2."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess import _deskew_angle, _rotate, preprocess


def _only(cfg, **overrides):
    """Return *cfg* with every preprocess step off except the given overrides."""
    base = {
        "enabled": True, "to_grayscale": True, "upscale": False, "denoise": False,
        "illumination_correction": False, "clahe": False, "deskew": False,
        "orientation_fix": False,
    }
    base.update(overrides)
    return cfg.model_copy(update={"preprocess": cfg.preprocess.model_copy(update=base)})


def _text_image() -> np.ndarray:
    """A clean white page with several lines of black text."""
    img = np.full((500, 720), 255, np.uint8)
    for i in range(5):
        cv2.putText(img, "INVOICE  TOTAL  RM 123.45", (40, 80 + i * 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    return img


# --- deskew -----------------------------------------------------------

@pytest.mark.parametrize("angle", [5.0, 7.0, -6.0, 11.0])
def test_deskew_corrects_synthetic_rotation(cfg, angle):
    skewed = _rotate(_text_image(), angle)          # rotate CCW by `angle`
    detected = _deskew_angle(skewed, cfg)
    assert detected is not None
    assert detected == pytest.approx(-angle, abs=1.5)

    result = preprocess(skewed, _only(cfg, deskew=True))
    residual = _deskew_angle(result.image, cfg)
    assert abs(residual) < 1.5
    assert abs(result.rotation_applied) == pytest.approx(abs(angle), abs=1.5)
    assert any(step.startswith("deskew(") for step in result.steps)


def test_deskew_ignores_tiny_angle(cfg):
    result = preprocess(_text_image(), _only(cfg, deskew=True))
    assert result.rotation_applied == 0.0
    assert not any(step.startswith("deskew(") for step in result.steps)


def test_deskew_skips_blank_image(cfg):
    blank = np.full((400, 400), 255, np.uint8)
    assert _deskew_angle(blank, cfg) is None


# --- contrast (CLAHE) ----------------------------------------------

def test_clahe_increases_contrast(cfg):
    low_contrast = np.random.default_rng(0).integers(120, 137, size=(300, 300), dtype=np.uint8)
    result = preprocess(low_contrast, _only(cfg, clahe=True))
    assert result.image.std() > low_contrast.std() * 2
    assert "clahe" in result.steps


# --- contract -----------------------------------------------------

def test_output_is_2d_uint8_with_full_pipeline(cfg):
    result = preprocess(cv2.cvtColor(_text_image(), cv2.COLOR_GRAY2BGR), cfg)
    assert result.image.ndim == 2
    assert result.image.dtype == np.uint8
    assert result.debug["00_input"].ndim == 3


def test_disabled_pipeline_still_returns_grayscale(cfg):
    colour = cv2.cvtColor(_text_image(), cv2.COLOR_GRAY2BGR)
    result = preprocess(colour, cfg.model_copy(
        update={"preprocess": cfg.preprocess.model_copy(update={"enabled": False})}))
    assert result.image.ndim == 2
    assert result.steps == ["disabled"]


def test_empty_image_raises(cfg):
    with pytest.raises(ValueError, match="empty image"):
        preprocess(np.empty((0, 0), np.uint8), cfg)
