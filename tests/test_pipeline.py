"""Pipeline orchestration tests. See Prompt.md - Phase 6.

The fast tests cover the error paths and JSON shape without touching an OCR
engine. The full end-to-end run is marked ``slow`` (``pytest -m slow``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import _empty_result, run_one, write_receipt_json
from src.schema import FieldConf, Item, ReceiptExtraction

# --- error paths -----------------------------------------------

def test_run_one_unreadable_image_flag(tmp_path, cfg):
    junk = tmp_path / "not_an_image.jpg"
    junk.write_bytes(b"\x00\x01 definitely not a JPEG \xff")
    result = run_one(junk, cfg)
    assert result.flags == ["unreadable_image"]
    assert result.meta["image_id"] == "not_an_image"
    assert result.store_name.value is None
    assert result.total_amount.confidence == 0.0


def test_run_one_missing_file_flag(tmp_path, cfg):
    result = run_one(tmp_path / "nope.png", cfg)
    assert result.flags == ["unreadable_image"]


def test_run_one_catches_internal_error(tmp_path, cfg, monkeypatch):
    (tmp_path / "x.png").write_bytes(b"junk")

    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    # imread succeeds on a real array; force a failure further down the pipeline
    monkeypatch.setattr("src.pipeline.imread_unicode", lambda *_a, **_k: np.zeros((8, 8), np.uint8))
    monkeypatch.setattr("src.pipeline.preprocess", boom)
    result = run_one(tmp_path / "x.png", cfg)
    assert result.flags == ["pipeline_error:RuntimeError"]
    assert result.meta["error"] == "kaboom"
    assert "traceback" in result.meta
    assert result.meta["image_id"] == "x"


def test_empty_result_is_schema_valid(cfg):
    result = _empty_result("abc", cfg, ["unreadable_image"])
    assert result.to_output_dict()["plain"] == {
        "store_name": None, "date": None, "items": [], "total_amount": None,
    }


# --- JSON output ---------------------------------------------

def _sample_result() -> ReceiptExtraction:
    r = ReceiptExtraction(meta={"image_id": "R001"})
    r.store_name = FieldConf(value="ACME SDN BHD", confidence=0.91)
    r.date = FieldConf(value="2018-03-12", confidence=0.8)
    r.total_amount = FieldConf(value="9.54", confidence=0.95)
    r.items = [Item(name=FieldConf(value="MILK", confidence=0.9),
                    price=FieldConf(value="5.94", confidence=0.88),
                    meta={"qty": 2})]
    r.flags = ["date_order_ambiguous"]
    r.low_confidence_fields = []
    return r


def test_write_receipt_json_shape_and_plain_key(tmp_path):
    path = write_receipt_json(_sample_result(), tmp_path)
    assert path.name == "R001.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert set(data) >= {"store_name", "date", "items", "total_amount",
                         "low_confidence_fields", "flags", "meta", "plain"}
    assert data["store_name"] == {"value": "ACME SDN BHD", "confidence": 0.91}
    assert data["items"][0] == {
        "name": {"value": "MILK", "confidence": 0.9},
        "price": {"value": "5.94", "confidence": 0.88},
        "qty": 2,
    }
    # embedded plain form matches the PDF schema exactly
    assert data["plain"] == {
        "store_name": "ACME SDN BHD",
        "date": "2018-03-12",
        "items": [{"name": "MILK", "price": "5.94"}],
        "total_amount": "9.54",
    }
    assert data["meta"]["image_id"] == "R001"


def test_write_receipt_json_is_deterministic(tmp_path):
    a = write_receipt_json(_sample_result(), tmp_path / "a").read_text(encoding="utf-8")
    b = write_receipt_json(_sample_result(), tmp_path / "b").read_text(encoding="utf-8")
    assert a == b


# --- full pipeline (slow) ---------------------------------

@pytest.mark.slow
def test_run_one_full_pipeline_on_real_receipt(cfg):
    images = sorted(Path(cfg.paths.input_dir).glob("*.jpg"))
    if not images:
        pytest.skip("no receipt images present")

    first = run_one(images[0], cfg)
    assert first.meta["image_id"] == images[0].stem
    assert first.meta["engine"] == cfg.ocr.engine
    assert first.meta["elapsed_ms"] >= 0
    assert first.meta["pipeline_version"] == cfg.run.pipeline_version
    assert first.meta["n_tokens"] >= 0
    json.dumps(first.to_output_dict())  # must serialise

    again = run_one(images[0], cfg)
    assert again.to_plain_dict() == first.to_plain_dict()  # deterministic
