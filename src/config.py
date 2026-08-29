"""Load and validate ``config.yaml`` into a typed, immutable object.

See Prompt.md - Phase 1. Every other module accepts a :class:`Config` produced
by :func:`load_config` rather than reading YAML or environment variables
directly.  The model tree mirrors ``config.yaml`` exactly and forbids unknown
keys, so a typo in the YAML fails loudly at startup instead of silently being
ignored.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class _Base(BaseModel):
    """Immutable and typo-proof: unknown keys raise, instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- paths ------------------------------------------------------------------

class Paths(_Base):
    input_dir: str
    output_dir: str
    json_dir: str
    debug_dir: str


# --- ocr ------------------------------------------------------------------

class LineGrouping(_Base):
    y_tolerance_factor: float
    right_column_frac: float


class OcrConfig(_Base):
    engine: str
    languages: list[str]
    gpu: bool
    tesseract_cmd: str | None
    line_grouping: LineGrouping

    @field_validator("engine")
    @classmethod
    def _known_engine(cls, v: str) -> str:
        if v.lower() not in {"easyocr", "tesseract"}:
            raise ValueError(f"ocr.engine must be 'easyocr' or 'tesseract', got {v!r}")
        return v.lower()


# --- preprocess ----------------------------------------------------------

class PreprocessConfig(_Base):
    enabled: bool
    to_grayscale: bool
    upscale: bool
    min_side: int
    max_side: int
    denoise: bool
    denoise_method: str
    denoise_h: float
    illumination_correction: bool
    clahe: bool
    clahe_clip: float
    clahe_tile: int
    deskew: bool
    deskew_max_deg: float
    deskew_min_deg: float
    orientation_fix: bool

    @field_validator("denoise_method")
    @classmethod
    def _known_denoise(cls, v: str) -> str:
        if v.lower() not in {"fastnl", "bilateral"}:
            raise ValueError(
                f"preprocess.denoise_method must be 'fastnl' or 'bilateral', got {v!r}"
            )
        return v.lower()

    @model_validator(mode="after")
    def _sane_sides(self) -> PreprocessConfig:
        if self.min_side > self.max_side:
            raise ValueError("preprocess.min_side must be <= preprocess.max_side")
        return self


# --- extract ------------------------------------------------------------

class StoreNameRules(_Base):
    n_header_lines: int
    company_suffixes: list[str]
    reject_contains: list[str]
    vendor_merge_ratio: int


class DateRules(_Base):
    keywords: list[str]
    min_year: int


class TotalRules(_Base):
    tier_a: list[str]
    tier_b: list[str]
    exclude: list[str]
    fallback_bottom_frac: float


class ItemRules(_Base):
    max_items: int
    separator_chars: str
    drop_contains: list[str]


class ExtractConfig(_Base):
    store_name: StoreNameRules
    date: DateRules
    total: TotalRules
    items: ItemRules


# --- confidence --------------------------------------------------------

class FieldWeights(_Base):
    """Signal weights for one field. Must sum to 1.0."""

    ocr: float = 0.0
    pattern: float = 0.0
    heuristic: float = 0.0
    cross_check: float = 0.0

    @model_validator(mode="after")
    def _sums_to_one(self) -> FieldWeights:
        total = self.ocr + self.pattern + self.heuristic + self.cross_check
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"confidence weights must sum to 1.0, got {total:.6f}")
        return self


class ConfidenceWeights(_Base):
    store_name: FieldWeights
    date: FieldWeights
    total_amount: FieldWeights
    item_name: FieldWeights
    item_price: FieldWeights


class CrossCheck(_Base):
    abs_tolerance: float
    rel_tolerance: float


class ConfidenceFlags(_Base):
    low_mean_ocr_conf: float
    mismatch_cross_check: float


class ConfidenceConfig(_Base):
    low_conf_threshold: float
    weights: ConfidenceWeights
    cross_check: CrossCheck
    flags: ConfidenceFlags


# --- summary / run ----------------------------------------------------

class SummaryConfig(_Base):
    currency: str
    min_total_conf: float


class RunConfig(_Base):
    seed: int
    workers: int
    pipeline_version: str


# --- root -------------------------------------------------------------

class Config(_Base):
    paths: Paths
    ocr: OcrConfig
    preprocess: PreprocessConfig
    extract: ExtractConfig
    confidence: ConfidenceConfig
    summary: SummaryConfig
    run: RunConfig


def load_config(path: str | Path | None = None) -> Config:
    """Read ``config.yaml`` (or *path*) and return a validated :class:`Config`.

    Raises ``FileNotFoundError`` if the file is missing and
    ``pydantic.ValidationError`` on any unknown key or type mismatch.
    """
    cfg_path = Path(path) if path else _DEFAULT_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
