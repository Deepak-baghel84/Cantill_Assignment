from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    """Normalize a value for matching while retaining useful letters/digits."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    if text in {"", "NAN", "NONE", "NULL", "NAT"}:
        return ""
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_supplier(value: Any) -> str:
    """Normalize supplier names and remove common legal suffixes."""
    text = clean_text(value)
    if not text:
        return ""
    suffixes = {
        "PTY", "LTD", "LIMITED", "PROPRIETARY", "INC", "INCORPORATED",
        "LLC", "PLC", "SA", "CC", "CORP", "CORPORATION", "COMPANY", "CO",
    }
    tokens = [token for token in text.split() if token not in suffixes]
    return " ".join(tokens)


def parse_number(value: Any) -> float | None:
    """Parse common finance number formats, returning None for invalid values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        number = float(value)
        return number if pd.notna(number) else None

    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text or text.upper() in {"NAN", "NONE", "NULL", "-"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace("'", "")

    # When both separators exist, the last separator is treated as decimal.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        if len(tail) in (1, 2):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    text = re.sub(r"[^0-9.\-+]", "", text)
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if negative:
        number = -abs(number)
    return number if pd.notna(number) else None


def parse_vat_rate(value: Any, default_rate: float = 0.15) -> float:
    """Return VAT as a fraction (15% -> 0.15); blank values use the default."""
    if value is None or str(value).strip() == "":
        return float(default_rate)
    text = str(value).strip().replace("%", "")
    parsed = parse_number(text)
    if parsed is None:
        return float(default_rate)
    rate = float(parsed)
    if rate > 1:
        rate /= 100.0
    if rate < 0 or rate > 1:
        raise ValueError(f"VAT rate is outside the supported range: {value!r}")
    return rate


def extract_po_and_item(value: Any) -> tuple[str, str]:
    """Extract an 8–12 digit PO and optional item from free-form PO references."""
    text = "" if value is None else str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL"}:
        return "", ""
    # Prefer a long PO-like integer; decimal suffixes on item numbers are accepted.
    match = re.search(r"(?<!\d)(\d{8,12})(?:\s*/\s*(\d+(?:\.\d+)?))?", text)
    if not match:
        digits = re.sub(r"\D", "", text)
        return (digits, "") if 8 <= len(digits) <= 12 else ("", "")
    po = match.group(1)
    item = match.group(2) or ""
    if item:
        try:
            item = str(int(float(item)))
        except ValueError:
            item = item.strip()
    return po, item


def normalize_po(value: Any) -> str:
    po, _ = extract_po_and_item(value)
    if po:
        return po
    text = str(value or "").strip()
    return re.sub(r"\.0$", "", text) if text else ""


def stable_key(prefix: str, value: Any) -> str:
    """Create deterministic opaque IDs; raw supplier/item names are not exposed."""
    normalized = clean_text(value)
    if not normalized:
        normalized = "UNKNOWN"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed ISO and day-month-name dates into pandas datetime values."""
    series = values.astype("string").fillna("").str.strip()
    series = series.mask(series.eq(""))
    try:
        # pandas >= 2 supports mixed-format parsing.
        return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    except (TypeError, ValueError):
        first = pd.to_datetime(series, dayfirst=True, errors="coerce")
        second = pd.to_datetime(series, dayfirst=False, errors="coerce")
        return first.fillna(second)
