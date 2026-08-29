"""Edge-case handling. See Prompt.md - Phase 9.

Covers, with tiny synthetic inputs:
  1 unreadable image       2 empty OCR result      3 partial receipt (no total)
  4 rotated image retry    5 non-RM currency       6 duplicate images
  7 item-count truncation  8 barcode / noise lines
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.confidence import score
from src.extract import extract_fields, extract_items, extract_store_name
from src.ocr import OcrResult, Token
from src.pipeline import _maybe_fix_orientation, run_one
from src.schema import FieldConf, ReceiptExtraction
from src.utils import detect_currency


def _lines(mean_conf: float, *texts: str) -> list[tuple[str, float]]:
    return [(t, mean_conf) for t in texts]


# --- 1. unreadable image ------------------------------------

def test_unreadable_image_gives_valid_json_with_flag(tmp_path, cfg):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg at all")
    result = run_one(bad, cfg)
    assert result.flags == ["unreadable_image"]
    assert result.to_output_dict()["plain"]["store_name"] is None
    assert result.store_name.confidence == 0.0


# --- 2. empty OCR result --------------------------------

def test_zero_tokens_flags_ocr_empty_and_nulls_everything(cfg, monkeypatch):
    empty = OcrResult(tokens=[], lines=[], full_text="", mean_conf=0.0, engine="fake")

    class _Blank:
        def run(self, _img):
            return empty

    monkeypatch.setattr("src.pipeline.get_engine", lambda _cfg: _Blank())
    monkeypatch.setattr("src.pipeline.imread_unicode",
                        lambda *_a, **_k: np.zeros((40, 40), np.uint8))
    result = run_one("whatever.png", cfg)
    assert "ocr_empty" in result.flags
    assert result.store_name.value is None and result.total_amount.value is None
    assert result.items == []


# --- 3. partial receipt (no total block) ----------------

def test_partial_receipt_total_via_fallback(make_ocr, cfg):
    ocr = make_ocr(_lines(0.9, "MINI MART", "APPLE 2.00", "BANANA 3.00", "5.00"))
    raw = extract_fields(ocr, cfg)
    assert raw.total_amount.rule == "fallback_max"
    assert len(raw.items) >= 1
    assert "total_from_fallback" in score(raw, ocr, cfg).flags


# --- 4. rotated image -> retry ---------------------------

def test_orientation_retry_picks_best_rotation(cfg):
    good = OcrResult(tokens=[Token("X", 0.9, (0, 0, 1, 1))], lines=[], full_text="X",
                     mean_conf=0.92, engine="fake")
    bad = OcrResult(tokens=[Token("?", 0.1, (0, 0, 1, 1))], lines=[], full_text="?",
                    mean_conf=0.10, engine="fake")

    class _Rotto:
        def run(self, img):
            # 90/270 rotations of a 20x40 image are 40x20 (taller than wide -> "good")
            return good if img.shape[0] > img.shape[1] else bad

    ocr, turn = _maybe_fix_orientation(_Rotto(), np.zeros((20, 40), np.uint8), bad, cfg)
    assert turn in (90, 270)
    assert ocr.mean_conf == pytest.approx(0.92)


def test_orientation_retry_noop_when_conf_ok(cfg):
    ok = OcrResult(tokens=[], lines=[], full_text="", mean_conf=0.80, engine="fake")

    class _Boom:
        def run(self, _img):
            raise AssertionError("should not re-OCR")

    result, turn = _maybe_fix_orientation(_Boom(), np.zeros((10, 10), np.uint8), ok, cfg)
    assert turn == 0 and result is ok


# --- 5. non-RM currency --------------------------------

def test_detect_currency():
    assert detect_currency("TOTAL USD 5.00\nCASH $10.00") == "USD"
    assert detect_currency("JUMLAH RM 42.90  RINGGIT") == "MYR"
    assert detect_currency("no money words here") is None
    assert detect_currency("PRICE $ 3.20") == "USD"


def test_run_one_records_detected_currency(make_ocr, cfg, monkeypatch):
    ocr = make_ocr(_lines(0.7, "SOME SHOP", "TOTAL USD 9.99"))

    monkeypatch.setattr("src.pipeline.get_engine",
                        lambda _cfg: type("E", (), {"run": lambda self, _i: ocr})())
    monkeypatch.setattr("src.pipeline.imread_unicode",
                        lambda *_a, **_k: np.zeros((40, 40), np.uint8))
    assert run_one("x.png", cfg).meta["currency_detected"] == "USD"


# --- 6. duplicate images ------------------------------

def test_batch_deduplicates_identical_images(tmp_path, monkeypatch):
    from src import cli

    receipts = tmp_path / "r"
    receipts.mkdir()
    (receipts / "a.jpg").write_bytes(b"IDENTICAL-BYTES")
    (receipts / "b.jpg").write_bytes(b"IDENTICAL-BYTES")
    (receipts / "c.jpg").write_bytes(b"different")

    calls: list[str] = []

    def fake_run_one(path, _cfg, **_kw):
        calls.append(path.stem)
        r = ReceiptExtraction(meta={"image_id": path.stem})
        r.total_amount = FieldConf(value="5.00", confidence=0.9)
        return r

    monkeypatch.setattr("src.pipeline.run_one", fake_run_one)
    out = tmp_path / "out"
    cli.main(["batch", "--input", str(receipts), "--output", str(out)])

    assert len(calls) == 2  # a + c processed, b skipped as a duplicate
    assert {p.name for p in (out / "json").glob("*.json")} == {"a.json", "b.json", "c.json"}
    dup = json.loads((out / "json" / "b.json").read_text(encoding="utf-8"))
    assert dup["meta"]["duplicate_of"] == "a"
    assert "duplicates: **1**" in (out / "run_report.md").read_text(encoding="utf-8")


# --- 7. item-count truncation -----------------------

def test_items_truncated_flag(make_ocr, cfg):
    tiny = cfg.model_copy(update={"extract": cfg.extract.model_copy(
        update={"items": cfg.extract.items.model_copy(update={"max_items": 3})})})
    ocr = make_ocr(_lines(0.9, "SHOP SDN BHD", "DATE 01/01/2018",
                          "ITEM A 1.00", "ITEM B 2.00", "ITEM C 3.00",
                          "ITEM D 4.00", "ITEM E 5.00", "TOTAL 15.00"))
    raw = extract_fields(ocr, tiny)
    assert len(raw.items) == 3
    assert raw.items_truncated is True
    assert "items_truncated" in score(raw, ocr, tiny).flags


# --- 8. barcode / noise lines ----------------------

def test_barcode_and_noise_lines_are_filtered(make_ocr, cfg):
    ocr = make_ocr(_lines(0.85, "REAL STORE SDN BHD", "DATE 02/02/2018",
                          "8 88012 34567 8", "XXXXXXXXXXXX", "NASI LEMAK 4.50",
                          "============", "TOTAL 4.50"))
    assert [n.value for n, _ in extract_items(ocr, cfg)] == ["NASI LEMAK"]
    assert extract_store_name(ocr, cfg).value == "REAL STORE SDN BHD"
