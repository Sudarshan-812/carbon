"""Shared pytest fixtures. See Prompt.md - Phase 10."""
from __future__ import annotations

import pytest

from src.config import load_config
from src.ocr import Line, OcrResult, Token


@pytest.fixture
def cfg():
    """The real, validated config from ``config.yaml``."""
    return load_config()


@pytest.fixture
def out_dir(tmp_path):
    """A throwaway ``outputs/`` tree (``json/`` + ``debug/`` pre-created)."""
    root = tmp_path / "outputs"
    (root / "json").mkdir(parents=True)
    (root / "debug").mkdir(parents=True)
    return root


@pytest.fixture
def make_ocr():
    """Build a fake :class:`OcrResult` from ``[(text, conf), ...]`` lines.

    One :class:`Token` per whitespace-separated word, laid out top-to-bottom
    with a crude left-to-right x advance so extraction's column logic has real
    geometry to work with. ``mean_conf`` is the mean of the line confidences.
    """

    def _factory(lines: list[tuple[str, float]], width: float = 400.0) -> OcrResult:
        toks: list[Token] = []
        built: list[Line] = []
        y = 0.0
        for text, conf in lines:
            x = 0.0
            line_tokens: list[Token] = []
            for word in text.split():
                w = max(10.0, len(word) * 8.0)
                t = Token(text=word, conf=conf, box=(x, y, w, 18.0))
                toks.append(t)
                line_tokens.append(t)
                x += w + 6.0
            built.append(Line(text=text, conf=conf,
                              box=(0.0, y, max(x, width), 18.0), tokens=line_tokens))
            y += 24.0
        mean = sum(c for _, c in lines) / len(lines) if lines else 0.0
        return OcrResult(tokens=toks, lines=built,
                         full_text="\n".join(t for t, _ in lines),
                         mean_conf=mean, engine="fake")

    return _factory
