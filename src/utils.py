"""Shared helpers: logging, timing, money parsing, unicode-safe image IO.

See Prompt.md - Phase 1. Nothing here depends on the rest of ``src`` so every
other module can import it freely.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# --- logging ---------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler attached once.

    Level comes from the ``CC_LOG_LEVEL`` environment variable (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("CC_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def ensure_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which crashes on the box-drawing
    characters EasyOCR prints in its download progress bar. Safe to call more
    than once and on non-Windows platforms.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover - stream not reconfigurable
                pass


# --- timing ---------------------------------------------------------------

class Timer:
    """Context manager measuring wall-clock time.

    >>> with Timer() as t:
    ...     ...
    >>> t()      # elapsed milliseconds
    >>> t.ms     # same value as a property

    Calling the instance while still inside the ``with`` block returns the
    time elapsed so far.
    """

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._end = time.perf_counter()
        return False

    def __call__(self) -> float:
        """Elapsed milliseconds (live if the block has not exited yet)."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0

    @property
    def ms(self) -> float:
        return self()


#: Backwards-compatible alias; ``with timer() as t:`` still works.
timer = Timer


# --- money parsing ------------------------------------------------------

_MONEY_RE = re.compile(
    r"""(?ix)
    (?:rm|myr|usd|\$)?\s*
    (?P<num>
        (?:\d{1,3}(?:[.,]\d{3})+|\d+)[.,]\d{2}(?!\d)   # 1,234.50 / 1.234,50 / 12.50
      | \d{1,3}(?:[.,]\d{3})+(?!\d)                    # 1,234 / 1.234.567 (thousands)
      | \d+                                            # 1234 / 5
    )
    \s*-?
    """
)


def _normalize_number(num: str) -> str:
    """Collapse thousands separators and settle on ``.`` as the decimal point."""
    has_comma, has_dot = "," in num, "." in num
    if has_comma and has_dot:
        # The right-most separator is the decimal point.
        if num.rfind(",") > num.rfind("."):
            return num.replace(".", "").replace(",", ".")
        return num.replace(",", "")
    if has_comma:
        # A single trailing group of exactly two digits -> decimal comma.
        if num.count(",") == 1 and len(num.rsplit(",", 1)[1]) == 2:
            return num.replace(",", ".")
        return num.replace(",", "")
    if num.count(".") > 1:
        # 1.234.567 -> European thousands, no decimal part.
        return num.replace(".", "")
    return num


def parse_money(text: str | None) -> float | None:
    """Parse a currency-ish string to ``float``; return ``None`` on failure.

    Handles a leading ``RM``/``MYR``/``$``, thousands separators, US (``1,234.50``)
    and European (``1.234,50``) decimals, a trailing ``-`` refund marker, and a
    missing cents part (``RM 5`` -> ``5.0``). A bare ``1.234`` is read as a
    decimal, not European thousands.
    """
    if not text:
        return None
    match = _MONEY_RE.search(text)
    if not match:
        return None
    try:
        return round(float(_normalize_number(match.group("num"))), 2)
    except ValueError:  # pragma: no cover - regex should guarantee a number
        return None


# --- currency detection -----------------------------------------------

_CURRENCY_WORDS = {
    "RM": "MYR", "MYR": "MYR", "RINGGIT": "MYR",
    "USD": "USD", "US$": "USD",
    "SGD": "SGD", "S$": "SGD",
    "EUR": "EUR", "GBP": "GBP", "AUD": "AUD", "JPY": "JPY",
    "IDR": "IDR", "RP": "IDR", "RUPIAH": "IDR",
    "THB": "THB", "BAHT": "THB", "INR": "INR",
}
_CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "¥": "JPY"}


def detect_currency(text: str | None) -> str | None:
    """Best-guess ISO currency code from free OCR text, or ``None`` if unclear.

    A bare ``$`` only votes for USD when no stronger word-level signal is
    present (it is also the Malaysian/Singapore ``S$`` tail, etc.).
    """
    if not text:
        return None
    upper = text.upper()
    votes: Counter[str] = Counter()
    for word, code in _CURRENCY_WORDS.items():
        hits = len(re.findall(rf"(?<![A-Z]){re.escape(word)}(?![A-Z])", upper))
        if hits:
            votes[code] += hits
    for symbol, code in _CURRENCY_SYMBOLS.items():
        hits = text.count(symbol)
        if hits:
            votes[code] += hits
    if not votes:
        return "USD" if "$" in text else None
    return votes.most_common(1)[0][0]


# --- image IO ----------------------------------------------------------

def imread_unicode(
    path: str | Path, flags: int = cv2.IMREAD_COLOR
) -> np.ndarray | None:
    """``cv2.imread`` replacement that tolerates non-ASCII Windows paths.

    Returns ``None`` for a missing, empty, or undecodable file (Phase 9's
    ``unreadable_image`` path relies on this).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:  # pragma: no cover - is_file() already screens most of this
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    """``cv2.imwrite`` replacement for non-ASCII paths. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(p))
    return True
