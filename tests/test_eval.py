"""Tests for the evaluation harness comparison + bucket helpers. Prompt.md - Phase 11."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import evaluate as ev


def test_match_store_exact_and_fuzzy():
    assert ev._match_store("KEDAI RUNCIT ABC", "kedai runcit abc") == (True, True)
    assert ev._match_store("TESCO STORES (M) SDN BHD", "TESCO STORES") == (False, True)
    assert ev._match_store("SPAR", "COLD STORAGE") == (False, False)
    assert ev._match_store(None, "ANYTHING") == (False, False)


def test_match_total_exact_and_tolerance():
    assert ev._match_total("54.20", "54.20") == (True, True)
    assert ev._match_total("54.23", "54.20") == (False, True)     # within 0.05
    assert ev._match_total("55.00", "54.20") == (False, False)
    assert ev._match_total(None, "54.20") == (False, False)
    assert ev._match_total("RM 1,234.50", "1234.50") == (True, True)
    assert ev._match_total("49.90", "49.9") == (True, True)      # 1-dp hand-typed label
    assert ev._match_total("72.00", "72") == (True, True)        # bare-int hand-typed label


def test_match_date_accepts_iso_and_hand_typed_formats():
    # model always emits ISO; labels may be hand-typed day-first
    assert ev._match_date("2018-03-05", "2018-03-05") is True
    assert ev._match_date("2018-03-05", "05-03-2018") is True
    assert ev._match_date("2018-03-05", "05/03/2018") is True
    assert ev._match_date("2018-03-06", "05-03-2018") is False
    assert ev._match_date(None, "05-03-2018") is False


def test_bucket_edges():
    assert ev._bucket(0.0) == (0.0, 0.5)
    assert ev._bucket(0.49) == (0.0, 0.5)
    assert ev._bucket(0.5) == (0.5, 0.7)
    assert ev._bucket(0.7) == (0.7, 0.9)
    assert ev._bucket(0.95) == (0.9, 1.01)
    assert ev._bucket(1.0) == (0.9, 1.01)


def test_field_stats_row():
    s = ev.FieldStats()
    assert s.row("x") == "| n/a | 0 | n/a | n/a |"
    s.add(exact=True, loose=True)
    s.add(exact=False, loose=True)
    row = s.row("fuzzy >= 90")
    assert "1/2 (50%)" in row and "2/2 (100%)" in row


def test_evaluate_end_to_end(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    (json_dir / "r1.json").write_text(json.dumps({
        "store_name": {"value": "MY STORE SDN BHD", "confidence": 0.95},
        "date": {"value": "2018-03-12", "confidence": 0.9},
        "total_amount": {"value": "9.54", "confidence": 0.8},
        "items": [{"name": {}, "price": {}}, {"name": {}, "price": {}}],
        "meta": {"image_id": "r1"},
    }), encoding="utf-8")

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "image_id,store_name,date,total_amount,n_items\n"
        "# comment line ignored\n"
        "r1,MY STORE,2018-03-12,9.55,2\n"
        "r2,GHOST,2018-01-01,1.00,1\n",
        encoding="utf-8",
    )

    result = ev.evaluate(json_dir, labels)
    assert result["matched"] == 1
    assert result["labelled_missing_pred"] == 1          # r2 has no prediction
    assert result["stats"]["store_name"].loose == 1      # fuzzy match
    assert result["stats"]["store_name"].exact == 0
    assert result["stats"]["date"].exact == 1
    assert result["stats"]["total_amount"].exact == 0    # 9.54 vs 9.55
    assert result["stats"]["total_amount"].loose == 1    # within 0.05
    assert result["stats"]["n_items"].exact == 1
    assert result["coverage"]["store_name"] == (1, 1)
