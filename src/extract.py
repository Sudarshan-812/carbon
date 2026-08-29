"""Key information extraction: store_name, date, items[], total_amount.

See Prompt.md - Phase 4. This module ONLY selects raw values and records how
it found them (provenance + signals). Confidence numbers are computed later in
``src/confidence.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .ocr import OcrResult
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class FieldRaw:
    """Raw pick for one field plus everything the confidence module needs."""

    value: str | None = None
    tokens_conf: list[float] = field(default_factory=list)  # OCR conf of value tokens
    rule: str = "none"           # which heuristic fired (e.g. "tier_a_keyword")
    line_index: int | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)  # e.g. dayfirst_assumed


@dataclass
class RawExtraction:
    store_name: FieldRaw = field(default_factory=FieldRaw)
    date: FieldRaw = field(default_factory=FieldRaw)
    total_amount: FieldRaw = field(default_factory=FieldRaw)
    items: list[tuple[FieldRaw, FieldRaw]] = field(default_factory=list)  # (name, price)
    body_span: tuple[int, int] | None = None


def extract_fields(ocr: OcrResult, cfg: Config) -> RawExtraction:
    """Run all four extractors over a reconstructed OCR result. TODO(Phase 4)."""
    raw = RawExtraction()
    raw.store_name = extract_store_name(ocr, cfg)
    raw.date = extract_date(ocr, cfg)
    raw.total_amount = extract_total(ocr, cfg)
    raw.items = extract_items(ocr, cfg)
    return raw


def extract_store_name(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Top-of-receipt scoring: caps, length, company suffix, not a reject word."""
    raise NotImplementedError("Phase 4")


def extract_date(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Regex family + keyword proximity + dateutil(dayfirst=True) -> ISO date."""
    raise NotImplementedError("Phase 4")


def extract_total(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Tiered total-keyword ranking with subtotal/tax/change exclusions."""
    raise NotImplementedError("Phase 4")


def extract_items(ocr: OcrResult, cfg: Config) -> list[tuple[FieldRaw, FieldRaw]]:
    """Body-region line parsing: name column vs right-aligned price column."""
    raise NotImplementedError("Phase 4")
