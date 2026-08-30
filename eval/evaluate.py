"""Evaluate pipeline output against a hand-labelled subset. See Prompt.md - Phase 11.

    python eval/evaluate.py --json-dir outputs/json --labels eval/labels.csv

We ship no ground truth, so ``eval/labels.csv`` is filled in by hand (30 rows
sampled across the dataset). Blank cells are skipped per field, so the harness
is useful even when only a few rows are done.

Reports:
  * per-field accuracy on the labelled subset
      store_name  - exact + fuzzy (rapidfuzz token_set_ratio >= 90)
      date        - exact ISO match
      total       - exact to 2dp + within 0.05
      n_items     - exact + off-by-one
  * coverage: % of every receipt in ``--json-dir`` with a non-null value
  * calibration: accuracy per confidence bucket (pooled store/date/total),
    written as a table and a bar chart ``eval/calibration.png``
"""
from __future__ import annotations

# ruff: noqa: I001  (a sys.path shim has to sit between stdlib and local imports)
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dateutil import parser as dtparser
from rapidfuzz import fuzz
from src.utils import parse_money

FUZZY_THRESHOLD = 90.0
TOTAL_TOLERANCE = 0.05
CONF_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
FIELDS_WITH_CONF = ("store_name", "date", "total_amount")


# --- loading -------------------------------------------------

def _load_predictions(json_dir: Path) -> dict[str, dict]:
    preds: dict[str, dict] = {}
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        preds[str(data.get("meta", {}).get("image_id", path.stem))] = data
    return preds


def _load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    return [r for r in rows if r.get("image_id")]


# --- comparisons ----------------------------------------

def _norm(text: str | None) -> str:
    return " ".join((text or "").upper().split())


def _match_store(pred: str | None, label: str) -> tuple[bool, bool]:
    p, g = _norm(pred), _norm(label)
    if not p:
        return False, False
    return p == g, fuzz.token_set_ratio(p, g) >= FUZZY_THRESHOLD


def _label_num(text: str) -> float | None:
    """Parse a hand-typed number cell. Labels are clean (``49.9``, ``72``,
    ``1,234.50``) so a plain float parse beats the OCR-oriented ``parse_money``,
    which truncates a single decimal place (``49.9`` -> ``49.0``)."""
    cleaned = re.sub(r"[^\d.\-]", "", (text or "").replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _match_total(pred: str | None, label: str) -> tuple[bool, bool]:
    pv, gv = parse_money(pred), _label_num(label)
    if pv is None or gv is None:
        return False, False
    return round(pv, 2) == round(gv, 2), abs(pv - gv) <= TOTAL_TOLERANCE


def _iso(text: str | None) -> str:
    """Normalise a date cell to ISO ``YYYY-MM-DD``. The pipeline always emits
    ISO; hand-typed labels may instead be ``DD-MM-YYYY`` / ``DD/MM/YYYY``."""
    text = (text or "").strip()
    if not text or re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    try:
        return dtparser.parse(text, dayfirst=True).date().isoformat()
    except (ValueError, OverflowError):
        return text


def _match_date(pred: str | None, label: str) -> bool:
    p, g = _iso(pred), _iso(label)
    return bool(p) and p == g


# --- stats containers ----------------------------------

@dataclass
class FieldStats:
    n: int = 0
    exact: int = 0
    loose: int = 0

    def add(self, exact: bool, loose: bool) -> None:
        self.n += 1
        self.exact += int(exact)
        self.loose += int(loose)

    def row(self, loose_label: str) -> str:
        if not self.n:
            return "| n/a | 0 | n/a | n/a |"
        return (f"| {self.exact}/{self.n} ({self.exact / self.n:.0%}) | {self.n} "
                f"| {loose_label} | {self.loose}/{self.n} ({self.loose / self.n:.0%}) |")


def _bucket(conf: float) -> tuple[float, float]:
    for lo, hi in CONF_BUCKETS:
        if lo <= conf < hi:
            return lo, hi
    return CONF_BUCKETS[-1]


# --- core ---------------------------------------------

def evaluate(json_dir: Path, labels_path: Path) -> dict:
    preds = _load_predictions(json_dir)
    labels = _load_labels(labels_path)

    stats = {f: FieldStats() for f in ("store_name", "date", "total_amount", "n_items")}
    calibration: dict[tuple[float, float], list[bool]] = {b: [] for b in CONF_BUCKETS}
    labelled_missing_pred = 0
    matched = 0

    for row in labels:
        data = preds.get(row["image_id"])
        if data is None:
            if any(row.get(k) for k in ("store_name", "date", "total_amount", "n_items")):
                labelled_missing_pred += 1
            continue
        matched += 1

        if row.get("store_name"):
            exact, fuzzy = _match_store(data["store_name"]["value"], row["store_name"])
            stats["store_name"].add(exact, fuzzy)
            calibration[_bucket(data["store_name"]["confidence"])].append(fuzzy)

        if row.get("date"):
            exact = _match_date(data["date"]["value"], row["date"])
            stats["date"].add(exact, exact)
            calibration[_bucket(data["date"]["confidence"])].append(exact)

        if row.get("total_amount"):
            exact, within = _match_total(data["total_amount"]["value"], row["total_amount"])
            stats["total_amount"].add(exact, within)
            calibration[_bucket(data["total_amount"]["confidence"])].append(within)

        if row.get("n_items"):
            pred_n, gold_n = len(data.get("items", [])), int(row["n_items"])
            stats["n_items"].add(pred_n == gold_n, abs(pred_n - gold_n) <= 1)

    return {
        "preds": preds,
        "n_labels": len(labels),
        "matched": matched,
        "labelled_missing_pred": labelled_missing_pred,
        "stats": stats,
        "calibration": calibration,
        "coverage": _coverage(preds),
    }


def _coverage(preds: dict[str, dict]) -> dict[str, tuple[int, int]]:
    n = len(preds)
    def count(fn) -> tuple[int, int]:
        return sum(1 for d in preds.values() if fn(d)), n
    return {
        "store_name": count(lambda d: d["store_name"]["value"]),
        "date": count(lambda d: d["date"]["value"]),
        "total_amount": count(lambda d: d["total_amount"]["value"]),
        "items (>=1)": count(lambda d: d.get("items")),
    }


# --- output ------------------------------------------

def _plot_calibration(calibration: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{lo:.1f}-{min(hi, 1.0):.1f}" for lo, hi in CONF_BUCKETS]
    accs = [sum(v) / len(v) if v else 0.0 for v in calibration.values()]
    counts = [len(v) for v in calibration.values()]

    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(labels, accs, color="#4C78A8", width=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={c}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("predicted confidence")
    ax.set_title("Confidence calibration: store(fuzzy) / date / total(+/-0.05)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _render_report(result: dict, json_dir: Path, chart_path: Path) -> str:
    stats: dict[str, FieldStats] = result["stats"]
    cov = result["coverage"]
    lines = [
        "# Evaluation report", "",
        f"- Predictions: **{len(result['preds'])}** JSON files in `{json_dir}`",
        (f"- Labels: **{result['n_labels']}** rows in `eval/labels.csv` "
         f"(**{result['matched']}** matched a prediction, "
         f"{result['labelled_missing_pred']} labelled rows have no prediction yet)"),
        "",
        "## Coverage (all predicted receipts)", "",
        "| Field | non-null | % |", "| --- | ---: | ---: |",
        *(f"| {k} | {c}/{n} | {(c / n if n else 0):.1%} |" for k, (c, n) in cov.items()),
        "",
        "## Accuracy on the labelled subset", "",
        "| Field | exact | n | loose metric | loose |",
        "| --- | ---: | ---: | --- | ---: |",
        f"| store_name {stats['store_name'].row('fuzzy ≥ 90')}",
        f"| date {stats['date'].row('(same as exact)')}",
        f"| total_amount {stats['total_amount'].row('within 0.05')}",
        f"| n_items {stats['n_items'].row('off-by-one ≤ 1')}",
        "",
        "## Confidence calibration (pooled store / date / total)", "",
        "| Confidence bucket | n | accuracy |", "| --- | ---: | ---: |",
    ]
    for (lo, hi), hits in result["calibration"].items():
        acc = f"{sum(hits) / len(hits):.0%}" if hits else "n/a"
        lines.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {len(hits)} | {acc} |")
    lines += ["", f"![calibration]({chart_path.name})", ""]

    weak = [f for f, s in stats.items() if s.n and s.loose / s.n < 0.7]
    if weak:
        lines += [(f"> Below 0.7 (loose): **{', '.join(weak)}**, "
                   "see Prompt.md Phase 11 follow-up before changing rules."), ""]
    elif sum(s.n for s in stats.values()) == 0:
        lines += [("> _No label cells filled yet: populate `eval/labels.csv` and "
                   "re-run for accuracy + calibration._"), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-dir", default="outputs/json", type=Path)
    ap.add_argument("--labels", default="eval/labels.csv", type=Path)
    ap.add_argument("--out", default="eval/eval_report.md", type=Path)
    ap.add_argument("--chart", default="eval/calibration.png", type=Path)
    args = ap.parse_args()

    if not args.json_dir.is_dir():
        print(f"no prediction dir: {args.json_dir}", file=sys.stderr)
        return 2

    result = evaluate(args.json_dir, args.labels)
    _plot_calibration(result["calibration"], args.chart)
    report = _render_report(result, args.json_dir, args.chart)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
