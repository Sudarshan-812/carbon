"""Tests for the engine-comparison helpers. See Prompt.md - Phase 12."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import compare_engines as ce


def test_cfg_for_overrides_engine_and_preprocess(cfg):
    variant = ce._cfg_for(cfg, "tesseract", preprocess_on=False)
    assert variant.ocr.engine == "tesseract"
    assert variant.preprocess.enabled is False
    assert cfg.ocr.engine == "easyocr"  # original untouched


def test_pooled_accuracy_ignores_missing_fields():
    run = ce.EngineRun(label="x", acc_store=0.8, acc_date=None, acc_total=0.6)
    assert run.pooled_accuracy == 0.7
    assert ce.EngineRun(label="y").pooled_accuracy is None


def test_resolve_finds_uppercase_extension(tmp_path):
    (tmp_path / "6.JPG").write_bytes(b"x")
    found = ce._resolve("6", tmp_path)
    assert found is not None and found.stem == "6" and found.suffix.lower() == ".jpg"
    assert ce._resolve("missing", tmp_path) is None


def test_takeaway_is_three_sentences_single_engine():
    only = ce.EngineRun(label="easyocr", mean_ocr_conf=0.71, sec_per_image=9.0)
    off = ce.EngineRun(label="easyocr (no preprocess)", mean_ocr_conf=0.63, sec_per_image=6.0)
    text = ce._takeaway([only], only, off, "easyocr", labelled=False)
    assert text.count(". ") + text.count("._") >= 2  # ~3 sentences
    assert "not installed" in text
    assert "blank until eval/labels.csv" in text


def test_table_row_formats_none_accuracy_as_na():
    row = ce._table([ce.EngineRun(label="easyocr", n=3, sec_per_image=9.1,
                                  mean_ocr_conf=0.7, mean_field_conf=0.6)])
    assert "| easyocr | 3 | 9.1 | 0.700 | 0.600 | n/a | n/a | n/a | n/a | 0 |" in row
