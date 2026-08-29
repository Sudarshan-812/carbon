"""Image preprocessing for OCR: denoise, illumination, deskew, orientation.

See Prompt.md - Phase 2. Public entry point is :func:`preprocess`. Every step is
individually toggleable under ``preprocess:`` in ``config.yaml``.

Nothing here hard-binarizes - EasyOCR reads grayscale better than a threshold
map. :func:`binarize_for_tesseract` is the only thresholding path and is used
solely by the Tesseract engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .utils import get_logger, imwrite_unicode

log = get_logger(__name__)

_EXACT_ROT = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess`.

    Attributes:
        image: cleaned single-channel ``uint8`` image ready for OCR.
        rotation_applied: net degrees the image was rotated (deskew + orientation).
        steps: human-readable names of the steps that actually ran.
        debug: intermediate images keyed by step name, for ``--debug`` dumps.
    """

    image: np.ndarray
    rotation_applied: float = 0.0
    steps: list[str] = field(default_factory=list)
    debug: dict[str, np.ndarray] = field(default_factory=dict)


def preprocess(img: np.ndarray, cfg: Config) -> PreprocessResult:
    """Clean *img* for OCR according to ``cfg.preprocess``.

    The pipeline is: grayscale -> upscale -> denoise -> illumination correction
    -> CLAHE -> deskew -> (optional) orientation fix. A disabled pipeline still
    returns a grayscale copy so callers always get a 2-D ``uint8`` image.
    """
    if img is None or img.size == 0:
        raise ValueError("empty image")

    p = cfg.preprocess
    steps: list[str] = []
    debug: dict[str, np.ndarray] = {"00_input": img.copy()}
    rotation = 0.0

    # Grayscale is required by the later steps and by the module contract; the
    # toggle only controls whether we record it as an explicit step.
    work = _to_gray(img)
    if p.to_grayscale:
        steps.append("grayscale")
    debug["01_gray"] = work.copy()

    if not p.enabled:
        return PreprocessResult(image=work, steps=["disabled"], debug=debug)

    if p.upscale:
        work, factor = _resize(work, p.min_side, p.max_side)
        if factor is not None:
            steps.append(f"resize(x{factor:.2f})")
            debug["02_resize"] = work.copy()

    if p.denoise:
        work = _denoise(work, p.denoise_method, p.denoise_h)
        steps.append(f"denoise({p.denoise_method})")
        debug["03_denoise"] = work.copy()

    if p.illumination_correction:
        work = _correct_illumination(work)
        steps.append("illumination")
        debug["04_illumination"] = work.copy()

    if p.clahe:
        work = _apply_clahe(work, p.clahe_clip, p.clahe_tile)
        steps.append("clahe")
        debug["05_clahe"] = work.copy()

    if p.deskew:
        angle = _deskew_angle(work, cfg)
        if angle is not None and p.deskew_min_deg <= abs(angle) <= p.deskew_max_deg:
            work = _rotate(work, angle)
            rotation += angle
            steps.append(f"deskew({angle:+.2f})")
            debug["06_deskew"] = work.copy()
        elif angle is not None:
            log.debug("deskew angle %.2f outside [%.2f, %.2f] - skipped",
                      angle, p.deskew_min_deg, p.deskew_max_deg)

    if p.orientation_fix:
        quarter = _detect_orientation(work)
        if quarter:
            work = cv2.rotate(work, _EXACT_ROT[quarter])
            rotation += quarter
            steps.append(f"orientation({quarter})")
            debug["07_orientation"] = work.copy()

    return PreprocessResult(
        image=work, rotation_applied=round(rotation, 2), steps=steps, debug=debug
    )


def binarize_for_tesseract(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Adaptive Gaussian threshold + light median blur. Tesseract path only."""
    gray = _to_gray(img)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    return cv2.medianBlur(binary, 3)


def dump_debug_images(result: PreprocessResult, debug_dir: str | Path, image_id: str) -> None:
    """Write every ``result.debug`` image to ``<debug_dir>/<image_id>/<step>.png``."""
    target = Path(debug_dir) / image_id
    for name, image in result.debug.items():
        imwrite_unicode(target / f"{name}.png", image)


# --- steps ---------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _resize(gray: np.ndarray, min_side: int, max_side: int) -> tuple[np.ndarray, float | None]:
    """Scale so the short side >= *min_side* while the long side <= *max_side*."""
    h, w = gray.shape[:2]
    short, long_ = min(h, w), max(h, w)
    scale = 1.0
    if short < min_side:
        scale = min_side / short
    if long_ * scale > max_side:
        scale = max_side / long_
    if abs(scale - 1.0) < 1e-3:
        return gray, None
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp), scale


def _denoise(gray: np.ndarray, method: str, strength: float) -> np.ndarray:
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=7, sigmaColor=strength * 5, sigmaSpace=strength * 5)
    return cv2.fastNlMeansDenoising(gray, None, h=float(strength),
                                    templateWindowSize=7, searchWindowSize=21)


def _correct_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting by dividing out a heavily blurred background."""
    h, w = gray.shape[:2]
    ksize = max(31, (min(h, w) // 16) | 1)
    background = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float32) + 1.0
    flattened = gray.astype(np.float32) / background
    return cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _apply_clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _deskew_angle(gray: np.ndarray, cfg: Config) -> float | None:
    """Skew estimate in degrees via ``minAreaRect`` over the text mask.

    The return value is the counter-clockwise rotation :func:`_rotate` must
    apply to make the text horizontal (negative = clockwise). ``None`` when
    there is too little foreground to measure.
    """
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0)).astype(np.float32)
    if coords.shape[0] < 50:
        return None
    # column_stack(where(...)) yields (row, col); minAreaRect wants (x, y).
    angle = cv2.minAreaRect(coords[:, ::-1])[-1]
    # minAreaRect reports a near-vertical edge as ~+-90; fold into (-45, 45].
    if angle < -45.0:
        angle += 90.0
    elif angle > 45.0:
        angle -= 90.0
    return angle


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by *angle*, expanding the canvas, median border."""
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    border = float(np.median(gray))
    return cv2.warpAffine(
        gray, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )


def _detect_orientation(gray: np.ndarray) -> int:
    """Return 90/180/270 if Tesseract OSD says the page is turned, else 0.

    Needs the Tesseract binary; any failure is swallowed and returns 0.
    """
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
        return rotate if rotate in _EXACT_ROT else 0
    except Exception as exc:  # noqa: BLE001 - optional path, needs external binary
        log.debug("orientation_fix skipped: %s", exc)
        return 0
