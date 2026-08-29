"""Shared helpers: logging, timing, money parsing, unicode-safe image IO.

See Prompt.md - Phase 1.
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(label: str = "block"):
    """Context manager that yields a callable returning elapsed ms."""
    start = time.perf_counter()
    try:
        yield lambda: (time.perf_counter() - start) * 1000.0
    finally:
        pass


# --- money parsing -------------------------------------------------------------

_MONEY_RE = re.compile(
    r"""(?ix)
    (?:rm|myr|\$)?\s*
    (?P<num>
        \d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})   # 1,234.50 / 1.234,50
        | \d+[.,]\d{2}                        # 1234.50
        | \d+                                 # 1234
    )
    \s*(?:-)?
    """
)


def parse_money(text: str | None) -> float | None:
    """Parse a currency-ish string to float. Returns ``None`` on failure.

    TODO(Phase 1): finish European-decimal handling and add the unit tests
    listed in the prompt (comma decimals, trailing '-', missing cents).
    """
    if not text:
        return None
    match = _MONEY_RE.search(text)
    if not match:
        return None
    num = match.group("num")
    if "," in num and "." in num:
        # assume the last separator is the decimal point
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif num.count(",") == 1 and len(num.split(",")[-1]) == 2:
        num = num.replace(",", ".")
    else:
        num = num.replace(",", "")
    try:
        return round(float(num), 2)
    except ValueError:
        return None


# --- image IO ----------------------------------------------------------------

def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread replacement that tolerates non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
