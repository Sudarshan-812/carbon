"""Key information extraction: store_name, date, items[], total_amount.

See Prompt.md - Phase 4. This module ONLY selects raw values and records how it
found them (provenance + signals). Confidence numbers are computed later in
``src/confidence.py``. Every keyword list lives in ``config.yaml`` so the rules
stay tunable without touching code.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from dateutil import parser as dateparser
from rapidfuzz import fuzz

from .config import Config
from .ocr import Line, OcrResult, Token
from .utils import get_logger, parse_money

log = get_logger(__name__)


@dataclass
class FieldRaw:
    """Raw pick for one field plus everything the confidence module needs."""

    value: str | None = None
    tokens_conf: list[float] = field(default_factory=list)  # OCR conf of value tokens
    rule: str = "missing"        # which heuristic fired (e.g. "tier_a_keyword")
    line_index: int | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawExtraction:
    store_name: FieldRaw = field(default_factory=FieldRaw)
    date: FieldRaw = field(default_factory=FieldRaw)
    total_amount: FieldRaw = field(default_factory=FieldRaw)
    items: list[tuple[FieldRaw, FieldRaw]] = field(default_factory=list)  # (name, price)
    body_span: tuple[int, int] | None = None


class VendorRegistry:
    """Run-wide list of seen store names, fuzzy-merged so variants collapse.

    ``"TESCO STORES (M)"`` and ``"TESCO"`` map to whichever was seen first.
    """

    def __init__(self, ratio: float = 90.0) -> None:
        self.ratio = ratio
        self._names: list[str] = []

    def canonical(self, name: str) -> tuple[str, bool]:
        """Return ``(canonical_name, matched_existing)``."""
        if not name:
            return name, False
        for known in self._names:
            if fuzz.token_set_ratio(name.upper(), known.upper()) >= self.ratio:
                return known, True
        self._names.append(name)
        return name, False


# --- money helpers ----------------------------------------------------

_TOKEN_MONEY_RE = re.compile(
    r"""(?ix) ^ \(? \s* (?:rm|myr)? \s*
        (?P<n> \d{1,3}(?:[.,]\d{3})*[.,]\d{2} | \d+[.,]\d{2} )
        \s* \)? \s* -? $
    """
)


def _price_of(text: str) -> float | None:
    """Parse a token that is *entirely* a money value (needs a 2-digit cents part)."""
    match = _TOKEN_MONEY_RE.match(text.strip())
    if not match:
        return None
    value = parse_money(match.group("n"))
    return None if value is None else abs(value)


def _money_tokens(line: Line) -> list[tuple[Token, float]]:
    """Every token on *line* that is a bare money value, left-to-right."""
    out = []
    for token in line.tokens:
        value = _price_of(token.text)
        if value is not None:
            out.append((token, value))
    return out


# --- store name ------------------------------------------------------

def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def _normalize_store(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t.,:;-_*|")


def _score_store_line(line: Line, position: int, cfg: Config) -> float:
    rules = cfg.extract.store_name
    text = line.text.strip()
    upper = text.upper()
    letters = [c for c in text if c.isalpha()]
    digits = sum(c.isdigit() for c in text)

    score = 1.5 * _caps_ratio(text)
    score += 1.0 if 4 <= len(text) <= 40 else -0.5
    score += line.conf
    score += max(0.0, 0.6 - 0.15 * position)          # gentle prior toward the top
    if any(sfx.upper() in upper for sfx in rules.company_suffixes):
        score += 2.0
    if not letters:
        score -= 3.0
    elif digits / max(len(text), 1) > 0.4:
        score -= 1.0
    if any(bad.upper() in upper for bad in rules.reject_contains):
        score -= 2.5
    return score


def extract_store_name(
    ocr: OcrResult, cfg: Config, vendors: VendorRegistry | None = None
) -> FieldRaw:
    """Score the first N header lines; pick the best; canonicalise via *vendors*."""
    rules = cfg.extract.store_name
    header: list[tuple[int, Line]] = []
    for i, line in enumerate(ocr.lines):
        if line.text.strip():
            header.append((i, line))
        if len(header) >= rules.n_header_lines:
            break
    if not header:
        return FieldRaw(rule="missing")

    ranked = sorted(
        ((_score_store_line(ln, pos, cfg), i, ln) for pos, (i, ln) in enumerate(header)),
        key=lambda t: t[0],
        reverse=True,
    )
    best_score, best_i, best_line = ranked[0]
    value = _normalize_store(best_line.text)

    matched = False
    if vendors is not None and value:
        value, matched = vendors.canonical(value)

    upper = best_line.text.upper()
    has_suffix = any(sfx.upper() in upper for sfx in rules.company_suffixes)
    if matched:
        rule = "vendor_match"
    elif has_suffix:
        rule = "company_suffix"
    elif best_score >= 1.5:
        rule = "top_scored"
    else:
        rule = "weak"

    return FieldRaw(
        value=value or None,
        tokens_conf=[t.conf for t in best_line.tokens] or [best_line.conf],
        rule=rule,
        line_index=best_i,
        alternatives=[
            {"value": _normalize_store(ln.text), "line_index": i, "score": round(sc, 2)}
            for sc, i, ln in ranked[:3]
        ],
        signals={"score": round(best_score, 2), "has_company_suffix": has_suffix,
                 "vendor_matched": matched},
    )


# --- date ----------------------------------------------------------

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE_PATTERNS = [
    # (regex, kind): "dmy" day-first numeric, "ymd" ISO-ish, "text" spelled month
    (re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b"), "dmy"),
    (re.compile(r"\b(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})\s*[-/ ]?\s*({_MONTHS})[a-z]*\.?\s*[-/ ]?\s*(\d{{2,4}})\b", re.IGNORECASE), "text"),
    (re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}})\s*,?\s*(\d{{2,4}})\b", re.IGNORECASE), "text"),
]


def _valid_iso(day: dt.date, cfg: Config) -> str | None:
    if day.year < cfg.extract.date.min_year or day > dt.date.today():  # noqa: DTZ011
        return None
    return day.isoformat()


def _parse_iso(raw: str, kind: str, cfg: Config, groups: tuple[str, ...]) -> str | None:
    if kind == "ymd":  # unambiguous - build it directly, dateutil mis-handles dayfirst here
        try:
            return _valid_iso(dt.date(int(groups[0]), int(groups[1]), int(groups[2])), cfg)
        except ValueError:
            return None
    try:
        parsed = dateparser.parse(raw, dayfirst=True, default=dt.datetime(2000, 1, 1))  # noqa: DTZ001
    except (ValueError, OverflowError, TypeError):
        return None
    return _valid_iso(parsed.date(), cfg)


def extract_date(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Regex family + keyword proximity + ``dateutil(dayfirst=True)`` -> ISO date."""
    keywords = [k.upper() for k in cfg.extract.date.keywords]
    candidates: list[dict[str, Any]] = []

    for i, line in enumerate(ocr.lines):
        text = line.text
        context = " ".join(
            ocr.lines[j].text for j in (i - 1, i, i + 1) if 0 <= j < len(ocr.lines)
        ).upper()
        near_keyword = any(k in context for k in keywords)
        for regex, kind in _DATE_PATTERNS:
            for match in regex.finditer(text):
                iso = _parse_iso(match.group(0), kind, cfg, match.groups())
                if iso is None:
                    continue
                ambiguous = False
                if kind == "dmy":
                    a, b = int(match.group(1)), int(match.group(2))
                    ambiguous = a <= 12 and b <= 12 and a != b
                candidates.append({
                    "iso": iso, "line_index": i, "raw": match.group(0).strip(),
                    "ambiguous": ambiguous, "near_keyword": near_keyword,
                })

    if not candidates:
        return FieldRaw(rule="missing")

    candidates.sort(key=lambda c: (not c["near_keyword"], c["ambiguous"], c["line_index"]))
    chosen = candidates[0]
    line = ocr.lines[chosen["line_index"]]

    distinct: list[str] = []
    for c in candidates:
        if c["iso"] not in distinct:
            distinct.append(c["iso"])

    return FieldRaw(
        value=chosen["iso"],
        tokens_conf=[t.conf for t in line.tokens] or [line.conf],
        rule="keyword_line" if chosen["near_keyword"] else "regex_only",
        line_index=chosen["line_index"],
        alternatives=[{"value": iso} for iso in distinct],
        signals={"dayfirst_assumed": chosen["ambiguous"], "raw": chosen["raw"],
                 "n_distinct": len(distinct)},
    )


# --- total amount --------------------------------------------------

def _tier_of(upper: str, cfg: Config) -> str | None:
    if any(k.upper() in upper for k in cfg.extract.total.tier_a):
        return "A"
    if any(k.upper() in upper for k in cfg.extract.total.tier_b):
        return "B"
    return None


def _excluded_total_line(upper: str, tier: str, cfg: Config) -> bool:
    for phrase in cfg.extract.total.exclude:
        if phrase.upper() not in upper:
            continue
        # tier-A phrasing ("TOTAL INCLUSIVE OF GST") legitimately contains GST/TAX.
        if phrase.upper() in {"GST", "TAX"} and tier == "A":
            continue
        return True
    return False


def extract_total(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Tiered total-keyword ranking with subtotal/tax/change exclusions."""
    candidates: list[dict[str, Any]] = []
    for i, line in enumerate(ocr.lines):
        upper = line.text.upper()
        tier = _tier_of(upper, cfg)
        if tier is None or _excluded_total_line(upper, tier, cfg):
            continue
        monies = _money_tokens(line)
        used_index, token = i, None
        if monies:
            token, value = monies[-1]
        else:  # keyword line has no number - look 1-2 lines down
            value = None
            for j in range(i + 1, min(i + 3, len(ocr.lines))):
                ahead = _money_tokens(ocr.lines[j])
                if ahead:
                    token, value = ahead[-1]
                    used_index = j
                    break
        if value is None:
            continue
        candidates.append({
            "value": value, "tier": tier, "line_index": i,
            "value_index": used_index, "conf": token.conf if token else line.conf,
        })

    pool = [c for c in candidates if c["tier"] == "A"] or \
           [c for c in candidates if c["tier"] == "B"]

    if pool:
        chosen = max(pool, key=lambda c: c["line_index"])  # totals sit near the bottom
        tier_a_values = {round(c["value"], 2) for c in candidates if c["tier"] == "A"}
        return FieldRaw(
            value=f"{chosen['value']:.2f}",
            tokens_conf=[chosen["conf"]],
            rule=f"tier_{chosen['tier'].lower()}_keyword",
            line_index=chosen["value_index"],
            alternatives=[
                {"value": f"{c['value']:.2f}", "tier": c["tier"], "line_index": c["line_index"]}
                for c in candidates
            ],
            signals={"tier": chosen["tier"], "from_fallback": False,
                     "conflicting": len(tier_a_values) >= 2,
                     "tier_a_values": sorted(tier_a_values)},
        )

    return _fallback_total(ocr, cfg)


def _fallback_total(ocr: OcrResult, cfg: Config) -> FieldRaw:
    """Largest money value in the bottom fraction of the receipt."""
    n = len(ocr.lines)
    start = int(n * (1.0 - cfg.extract.total.fallback_bottom_frac))
    found: list[tuple[float, float, int]] = []
    for i in range(start, n):
        for token, value in _money_tokens(ocr.lines[i]):
            found.append((value, token.conf, i))
    if not found:
        return FieldRaw(rule="missing")
    value, conf, idx = max(found, key=lambda t: t[0])
    return FieldRaw(
        value=f"{value:.2f}",
        tokens_conf=[conf],
        rule="fallback_max",
        line_index=idx,
        alternatives=[{"value": f"{v:.2f}", "line_index": i} for v, _, i in found],
        signals={"tier": None, "from_fallback": True, "conflicting": False},
    )


# --- items + prices ----------------------------------------------

_QTY_LEAD_RE = re.compile(r"(?i)^\s*(\d{1,3})\s*[x@*]?\s+")
_QTY_TRAIL_RE = re.compile(r"(?i)\s*\d+(?:\.\d+)?\s*[x@*]\s*$")
_QTY_ANY_RE = re.compile(r"(?i)\b(\d{1,3})\s*[x@*]\s*\d")
_CODE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-/]{4,}$")


def _is_separator(text: str, sep_chars: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    hits = sum(1 for c in stripped if c in sep_chars or c.isspace())
    return hits / len(stripped) >= 0.8


def _is_code(text: str) -> bool:
    return bool(_CODE_RE.match(text)) and sum(c.isdigit() for c in text) >= 3


def _find_qty(text: str) -> int | None:
    match = _QTY_ANY_RE.search(text) or re.match(r"\s*(\d{1,3})\s+[A-Za-z]", text)
    return int(match.group(1)) if match else None


def _body_span(ocr: OcrResult, cfg: Config) -> tuple[int, int]:
    """[start, end) line range holding the itemised body."""
    lines = ocr.lines
    n = len(lines)
    sep_chars = cfg.extract.items.separator_chars
    date_keywords = [k.upper() for k in cfg.extract.date.keywords]

    start = 0
    for i, line in enumerate(lines[: max(3, n // 2)]):
        upper = line.text.upper()
        if _is_separator(line.text, sep_chars):
            start = i + 1
        elif any(k in upper for k in date_keywords) or "TEL" in upper or "GST ID" in upper:
            start = max(start, i + 1)

    end = n
    for i in range(start, n):
        upper = lines[i].text.upper()
        if "SUBTOTAL" in upper or "SUB TOTAL" in upper:
            end = i
            break
        if _tier_of(upper, cfg) is not None and _money_tokens(lines[i]):
            end = i
            break
    return start, min(end if end > start else n, n)


def _split_name_price(line: Line, cfg: Config) -> tuple[str, tuple[Token, float] | None, int | None]:
    """Return (name, (price_token, value) | None, qty) for one body line."""
    monies = _money_tokens(line)
    price: tuple[Token, float] | None = None
    if monies:
        right_frac = cfg.ocr.line_grouping.right_column_frac
        cut = line.box[0] + line.box[2] * (1.0 - right_frac)
        right_side = [m for m in monies if m[0].cx >= cut]
        price = (right_side or monies)[-1]

    price_token = price[0] if price else None
    name_tokens = [
        t for t in line.tokens
        if t is not price_token and _price_of(t.text) is None and not _is_code(t.text)
    ]
    name = " ".join(t.text for t in name_tokens)
    name = _QTY_TRAIL_RE.sub("", _QTY_LEAD_RE.sub("", name))
    name = re.sub(r"\s+", " ", name).strip(" \t.,:;-_*/|")
    return name, price, _find_qty(line.text)


def extract_items(ocr: OcrResult, cfg: Config) -> list[tuple[FieldRaw, FieldRaw]]:
    """Body-region line parsing: name column vs right-aligned price column."""
    rules = cfg.extract.items
    drop = [d.upper() for d in rules.drop_contains]
    start, end = _body_span(ocr, cfg)
    items: list[tuple[FieldRaw, FieldRaw]] = []

    i = start
    while i < end and len(items) < rules.max_items:
        line = ocr.lines[i]
        text = line.text.strip()
        upper = text.upper()
        if not text or _is_separator(text, rules.separator_chars) or any(d in upper for d in drop):
            i += 1
            continue

        name, price, qty = _split_name_price(line, cfg)
        consumed = 0
        if price is None and name and i + 1 < end:  # price on the following line
            ahead = _money_tokens(ocr.lines[i + 1])
            if len(ahead) == 1 and len(ocr.lines[i + 1].text.split()) <= 2:
                price = ahead[0]
                consumed = 1

        if price is None or len(name) < 3 or sum(c.isalpha() for c in name) < 2:
            i += 1 + consumed
            continue

        price_token, value = price
        items.append((
            FieldRaw(
                value=name,
                tokens_conf=[t.conf for t in line.tokens
                             if t is not price_token and _price_of(t.text) is None]
                or [line.conf],
                rule="body_line",
                line_index=i,
                signals={"qty": qty},
            ),
            FieldRaw(
                value=f"{value:.2f}",
                tokens_conf=[price_token.conf],
                rule="right_column" if consumed == 0 else "next_line",
                line_index=i + consumed,
                signals={"qty": qty},
            ),
        ))
        i += 1 + consumed

    if len(items) >= rules.max_items:
        log.debug("items truncated at max_items=%d", rules.max_items)
    return items


# --- entry point --------------------------------------------------

def extract_fields(
    ocr: OcrResult, cfg: Config, vendors: VendorRegistry | None = None
) -> RawExtraction:
    """Run all four extractors over a reconstructed OCR result."""
    raw = RawExtraction()
    raw.store_name = extract_store_name(ocr, cfg, vendors)
    raw.date = extract_date(ocr, cfg)
    raw.total_amount = extract_total(ocr, cfg)
    raw.items = extract_items(ocr, cfg)
    raw.body_span = _body_span(ocr, cfg)
    return raw
