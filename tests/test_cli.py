"""CLI + batch-runner tests. See Prompt.md - Phase 8.

``run_one`` is monkeypatched so these never touch a real OCR engine.
"""
from __future__ import annotations

import json

import pytest

from src import cli
from src.schema import FieldConf, Item, ReceiptExtraction


def _fake_result(image_id: str, *, error: bool = False, total: str = "10.00",
                 conf: float = 0.9) -> ReceiptExtraction:
    meta = {"image_id": image_id, "engine": "fake", "mean_ocr_confidence": 0.8,
            "n_lines": 5, "n_tokens": 12, "elapsed_ms": 1.0, "pipeline_version": "0.1.0"}
    if error:
        return ReceiptExtraction(flags=["pipeline_error:RuntimeError"], meta=meta,
                                 low_confidence_fields=["store_name", "date", "total_amount"])
    r = ReceiptExtraction(meta=meta)
    r.store_name = FieldConf(value=f"STORE {image_id[:1].upper()}", confidence=conf)
    r.date = FieldConf(value="2018-03-01", confidence=0.8)
    r.total_amount = FieldConf(value=total, confidence=conf)
    r.items = [Item(name=FieldConf(value="THING", confidence=0.6),
                    price=FieldConf(value="10.00", confidence=0.5))]
    r.low_confidence_fields = ["items[0].name", "items[0].price"]
    return r


@pytest.fixture
def receipts_dir(tmp_path):
    d = tmp_path / "receipts"
    d.mkdir()
    for name in ("b.jpg", "a.jpg", "c.PNG", "notes.txt"):
        (d / name).write_bytes(b"x")
    return d


# --- helpers ------------------------------------------------

def test_iter_images_sorted_filtered_limited(receipts_dir):
    from src.cli import _iter_images

    assert [p.name for p in _iter_images(receipts_dir, None)] == ["a.jpg", "b.jpg", "c.PNG"]
    assert [p.name for p in _iter_images(receipts_dir, 2)] == ["a.jpg", "b.jpg"]


def test_config_hash_changes_with_engine(cfg):
    other = cfg.model_copy(update={"ocr": cfg.ocr.model_copy(update={"engine": "tesseract"})})
    assert cli._config_hash(cfg) == cli._config_hash(cfg)
    assert cli._config_hash(cfg) != cli._config_hash(other)


def test_apply_overrides_engine_and_paths(cfg, tmp_path):
    ns = __import__("argparse").Namespace(
        engine="tesseract", input="in_dir", output=str(tmp_path / "out"), json_dir=None)
    merged = cli._apply_overrides(cfg, ns)
    assert merged.ocr.engine == "tesseract"
    assert merged.paths.input_dir == "in_dir"
    assert merged.paths.json_dir == str(tmp_path / "out" / "json")


# --- batch --------------------------------------------------

def test_batch_writes_json_summary_and_report(receipts_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.run_one",
                        lambda path, cfg, **_kw: _fake_result(path.stem))
    out = tmp_path / "out"
    rc = cli.main(["batch", "--input", str(receipts_dir), "--output", str(out)])
    assert rc == 0

    assert {p.name for p in (out / "json").glob("*.json")} == {"a.json", "b.json", "c.json"}
    assert (out / "expense_summary.json").is_file()
    assert (out / "expense_summary.csv").is_file()

    report = (out / "run_report.md").read_text(encoding="utf-8")
    assert "# Run report" in report
    assert "Config hash:" in report
    assert "Mean confidence per field" in report
    assert "items[].price" in report          # granular low-conf roll-up
    assert "## Financial summary" in report     # appended by write_summary


def test_batch_limit(receipts_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.run_one",
                        lambda path, cfg, **_kw: _fake_result(path.stem))
    out = tmp_path / "out"
    cli.main(["batch", "--input", str(receipts_dir), "--output", str(out), "--limit", "1"])
    assert {p.name for p in (out / "json").glob("*.json")} == {"a.json"}


def test_batch_exit_nonzero_when_too_many_errors(receipts_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.run_one",
                        lambda path, cfg, **_kw: _fake_result(path.stem, error=True))
    rc = cli.main(["batch", "--input", str(receipts_dir), "--output", str(tmp_path / "o")])
    assert rc == 1


def test_batch_missing_input_dir(tmp_path):
    assert cli.main(["batch", "--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]) == 2


# --- one / summary --------------------------------------

def test_one_command_prints_json(receipts_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.pipeline.run_one",
                        lambda path, cfg, **_kw: _fake_result("single"))
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["one", str(receipts_dir / "a.jpg")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["plain"]["store_name"] == "STORE S"


def test_summary_command(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.run_one",
                        lambda path, cfg, **_kw: _fake_result(path.stem))
    receipts = tmp_path / "r"
    receipts.mkdir()
    (receipts / "a.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    cli.main(["batch", "--input", str(receipts), "--output", str(out)])

    rc = cli.main(["summary", "--json-dir", str(out / "json"), "--output", str(out)])
    assert rc == 0
    data = json.loads((out / "expense_summary.json").read_text(encoding="utf-8"))
    assert data["num_transactions"] == 1
    assert data["total_spend"] == 10.00
