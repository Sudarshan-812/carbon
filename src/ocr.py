"""OCR engine wrappers + engine-agnostic line reconstruction.

See Prompt.md - Phase 3. EasyOCR is the primary engine; Tesseract is kept for
the comparison experiment (Phase 12). Both map their native output onto the same
:class:`OcrResult` so the rest of the pipeline never has to care which ran.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import Config, load_config
from .preprocess import binarize_for_tesseract, preprocess
from .utils import ensure_utf8_stdout, get_logger, imread_unicode

log = get_logger(__name__)

BBox = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Token:
    """One OCR text box: its string, probability, and pixel bounding box."""

    text: str
    conf: float
    box: BBox

    @property
    def x(self) -> float:
        return self.box[0]

    @property
    def y(self) -> float:
        return self.box[1]

    @property
    def right(self) -> float:
        return self.box[0] + self.box[2]

    @property
    def bottom(self) -> float:
        return self.box[1] + self.box[3]

    @property
    def cx(self) -> float:
        return self.box[0] + self.box[2] / 2.0

    @property
    def cy(self) -> float:
        return self.box[1] + self.box[3] / 2.0


@dataclass
class Line:
    """A row of tokens reconstructed from their geometry."""

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


# --- EasyOCR -----------------------------------------------------------

_READERS: dict[tuple[int, tuple[str, ...]], object] = {}
_READERS_LOCK = threading.Lock()


def _easyocr_reader(languages: list[str], gpu: bool) -> object:
    """Return a cached ``easyocr.Reader``, one per (thread, language set).

    EasyOCR's ``Reader`` is not thread-safe, so a batch run with ``workers > 1``
    gets an independent reader per worker thread; single-threaded runs reuse one.
    """
    key = (threading.get_ident(), tuple(languages))
    with _READERS_LOCK:
        reader = _READERS.get(key)
        if reader is None:
            import easyocr  # heavy (pulls torch); import only when actually used

            log.info("building EasyOCR reader for languages=%s gpu=%s", list(key[1]), gpu)
            reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
            _READERS[key] = reader
    return reader


class EasyOcrEngine:
    """Wraps :class:`easyocr.Reader`; the reader itself is a module singleton."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, img: np.ndarray) -> OcrResult:
        reader = _easyocr_reader(self.cfg.ocr.languages, self.cfg.ocr.gpu)
        raw = reader.readtext(img, detail=1, paragraph=False)
        tokens: list[Token] = []
        for poly, text, conf in raw:
            text = str(text).strip()
            if not text:
                continue
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            x, y = min(xs), min(ys)
            tokens.append(Token(text, float(conf), (x, y, max(xs) - x, max(ys) - y)))
        return _assemble(tokens, self.cfg, engine="easyocr")


# --- Tesseract ---------------------------------------------------------

class TesseractEngine:
    """Wraps ``pytesseract.image_to_data``; binarizes the image first."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, img: np.ndarray) -> OcrResult:
        import pytesseract
        from pytesseract import Output

        if self.cfg.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.ocr.tesseract_cmd

        data = pytesseract.image_to_data(
            binarize_for_tesseract(img, self.cfg), output_type=Output.DICT
        )
        tokens: list[Token] = []
        for i, raw_conf in enumerate(data["conf"]):
            conf = float(raw_conf)
            text = str(data["text"][i]).strip()
            if conf < 0 or not text:  # conf == -1 marks a non-text row
                continue
            tokens.append(Token(
                text, conf / 100.0,
                (float(data["left"][i]), float(data["top"][i]),
                 float(data["width"][i]), float(data["height"][i])),
            ))
        return _assemble(tokens, self.cfg, engine="tesseract")


def get_engine(cfg: Config) -> OcrEngine:
    name = str(cfg.ocr.engine).lower()
    if name == "easyocr":
        return EasyOcrEngine(cfg)
    if name == "tesseract":
        return TesseractEngine(cfg)
    raise ValueError(f"unknown ocr.engine: {name!r}")


# --- line reconstruction (engine-agnostic) ---------------------------

def _assemble(tokens: list[Token], cfg: Config, engine: str) -> OcrResult:
    """Merge split number fragments, group tokens into lines, build the result."""
    tokens = merge_split_numbers(tokens)
    lines = reconstruct_lines(tokens, cfg)
    mean_conf = statistics.fmean(t.conf for t in tokens) if tokens else 0.0
    full_text = "\n".join(line.text for line in lines)
    return OcrResult(tokens, lines, full_text, mean_conf, engine)


def _median_height(tokens: list[Token]) -> float:
    return statistics.median(t.box[3] for t in tokens) if tokens else 0.0


def reconstruct_lines(tokens: list[Token], cfg: Config) -> list[Line]:
    """Cluster tokens into rows by y-centre proximity, order them for reading.

    Tolerance is ``median token height * cfg.ocr.line_grouping.y_tolerance_factor``.
    Rows are sorted top-to-bottom, tokens left-to-right within a row.
    """
    if not tokens:
        return []

    tol = _median_height(tokens) * cfg.ocr.line_grouping.y_tolerance_factor
    tol = max(tol, 1.0)

    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda t: t.cy):
        for row in rows:
            row_cy = statistics.fmean(t.cy for t in row)
            if abs(token.cy - row_cy) <= tol:
                row.append(token)
                break
        else:
            rows.append([token])

    lines: list[Line] = []
    for row in rows:
        row.sort(key=lambda t: t.x)
        x0 = min(t.x for t in row)
        y0 = min(t.y for t in row)
        x1 = max(t.right for t in row)
        y1 = max(t.bottom for t in row)
        lines.append(Line(
            text=" ".join(t.text for t in row),
            conf=statistics.fmean(t.conf for t in row),
            box=(x0, y0, x1 - x0, y1 - y0),
            tokens=row,
        ))

    lines.sort(key=lambda line: line.box[1])
    return lines


def merge_split_numbers(tokens: list[Token]) -> list[Token]:
    """Rejoin a bare ``.`` / ``.dd`` fragment onto the preceding numeric token.

    EasyOCR sometimes cuts ``12.90`` into ``12`` + ``.90`` (or ``12`` ``.`` ``90``).
    Only fragments that are punctuation-led (``.90``, ``,90``, ``.``) and sit
    immediately to the right of a digit-ending token on the same row are merged,
    so ordinary adjacent words are left alone.
    """
    if len(tokens) < 2:
        return list(tokens)

    ordered = sorted(tokens, key=lambda t: (t.cy, t.x))
    gap_limit = max(_median_height(tokens) * 0.6, 4.0)
    out: list[Token] = []
    for token in ordered:
        if out and _is_number_continuation(out[-1], token, gap_limit):
            prev = out[-1]
            frag = token.text.strip()
            x0, y0 = min(prev.x, token.x), min(prev.y, token.y)
            x1, y1 = max(prev.right, token.right), max(prev.bottom, token.bottom)
            out[-1] = Token(prev.text + frag, min(prev.conf, token.conf),
                            (x0, y0, x1 - x0, y1 - y0))
            continue
        out.append(token)
    return out


def _is_number_continuation(prev: Token, token: Token, gap_limit: float) -> bool:
    """True when *token* is a numeric tail that belongs on *prev* (same row, touching)."""
    frag = token.text.strip()
    if abs(token.cy - prev.cy) > gap_limit or not (-2.0 <= token.x - prev.right <= gap_limit):
        return False
    # ".90" / ",90" / "." right after a digit  ->  "12" + ".90" = "12.90"
    punct_led = frag in {".", ","} or (
        2 <= len(frag) <= 3 and frag[0] in {".", ","} and frag[1:].isdigit()
    )
    if punct_led and prev.text[-1:].isdigit():
        return True
    # trailing-separator token completed by 1-2 digits  ->  "12." + "90" = "12.90"
    stem = prev.text[:-1]
    return (
        prev.text[-1:] in {".", ","}
        and stem.replace(".", "").replace(",", "").isdigit()
        and 1 <= len(frag) <= 2
        and frag.isdigit()
    )


# --- CLI -------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    ap = argparse.ArgumentParser(prog="python -m src.ocr",
                                 description="OCR a receipt and print reconstructed lines")
    ap.add_argument("image")
    ap.add_argument("--engine", choices=["easyocr", "tesseract"], default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", action="store_true", help="skip preprocessing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.engine:
        cfg = cfg.model_copy(update={"ocr": cfg.ocr.model_copy(update={"engine": args.engine})})

    img = imread_unicode(args.image)
    if img is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1
    if not args.raw:
        img = preprocess(img, cfg).image

    result = get_engine(cfg).run(img)
    print(f"# engine={result.engine}  lines={len(result.lines)}  "
          f"tokens={len(result.tokens)}  mean_conf={result.mean_conf:.3f}")
    for line in result.lines:
        print(f"{line.conf:.2f}  {line.text}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
