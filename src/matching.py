from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .utils import clean_text, extract_po_and_item, normalize_po, normalize_supplier, parse_dates


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _amount_similarity(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    relative_error = abs(a - b) / scale
    return max(0.0, 1.0 - min(relative_error / 0.05, 1.0))


def match_tool_records(
    erp: pd.DataFrame,
    invoice: pd.DataFrame,
    erp_net_amounts: list[float | None],
    invoice_net_amounts: list[float | None],
    config: dict[str, Any] | None = None,
    eligible_invoice_positions: set[int] | None = None,
) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one invoice-to-ERP matching. Returns invoice_idx -> (erp_idx, score)."""
    config = config or {}
    po_threshold = float(config.get("po_match_threshold", 0.70))
    no_po_threshold = float(config.get("no_po_match_threshold", 0.86))
    supplier_threshold = float(config.get("supplier_name_threshold", 0.90))
    max_candidates = max(20, int(config.get("max_candidates_per_invoice", 150)))

    if eligible_invoice_positions is None:
        eligible_invoice_positions = set(range(len(invoice)))

    erp_po_index: dict[str, list[int]] = defaultdict(list)
    supplier_index: dict[str, list[int]] = defaultdict(list)
    amount_index: dict[int, list[int]] = defaultdict(list)

    erp_suppliers: list[str] = []
    erp_descriptions: list[str] = []
    erp_items: list[str] = []
    erp_pos: list[str] = []

    for i, (_, row) in enumerate(erp.iterrows()):
        po = normalize_po(row.get("po_number", ""))
        supplier = normalize_supplier(row.get("supplier_name", ""))
        description = clean_text(" ".join([
            str(row.get("material_short_text", "")),
            str(row.get("service_short_text", "")),
        ]))
        item = str(row.get("po_item", "")).strip()
        if item.endswith(".0"):
            item = item[:-2]
        erp_po_index[po].append(i) if po else None
        supplier_index[supplier].append(i) if supplier else None
        amount = erp_net_amounts[i]
        if amount is not None:
            amount_index[round(float(amount))].append(i)
        erp_suppliers.append(supplier)
        erp_descriptions.append(description)
        erp_items.append(item)
        erp_pos.append(po)

    erp_dates = parse_dates(erp.get("posting_date", pd.Series([""] * len(erp))))
    inv_dates = parse_dates(invoice.get("invoice_date", pd.Series([""] * len(invoice))))

    matches: dict[int, tuple[int, float]] = {}
    used_erp: set[int] = set()

    # Process rows with the strongest PO evidence first, retaining original output IDs.
    positions = sorted(eligible_invoice_positions, key=lambda p: (not bool(extract_po_and_item(invoice.iloc[p].get("po_reference", ""))[0]), p))
    for inv_idx in positions:
        row = invoice.iloc[inv_idx]
        po, po_item = extract_po_and_item(row.get("po_reference", ""))
        supplier = normalize_supplier(row.get("supplier_name_as_captured", ""))
        description = clean_text(row.get("line_text", ""))
        amount = invoice_net_amounts[inv_idx]

        candidate_set: set[int] = set()
        if po:
            candidate_set.update(erp_po_index.get(po, []))
        if not candidate_set:
            if supplier:
                candidate_set.update(supplier_index.get(supplier, []))
                # Include limited neighboring aliases with close supplier names.
                if len(candidate_set) < max_candidates:
                    for known, indices in supplier_index.items():
                        if known != supplier and _ratio(supplier, known) >= supplier_threshold:
                            candidate_set.update(indices)
                            if len(candidate_set) >= max_candidates * 3:
                                break
            if amount is not None:
                rounded = round(float(amount))
                for bucket in (rounded - 1, rounded, rounded + 1):
                    candidate_set.update(amount_index.get(bucket, []))

        candidate_set.difference_update(used_erp)
        if not candidate_set:
            continue
        candidates = sorted(candidate_set)
        if len(candidates) > max_candidates:
            # Prefer candidates sharing supplier; if none, retain a deterministic prefix.
            candidates.sort(key=lambda idx: (
                -_ratio(supplier, erp_suppliers[idx]),
                abs((erp_net_amounts[idx] or 0.0) - (amount or 0.0)),
                idx,
            ))
            candidates = candidates[:max_candidates]

        best_idx: int | None = None
        best_score = -1.0
        for erp_idx in candidates:
            po_score = 1.0 if po and erp_pos[erp_idx] == po else 0.0
            supplier_score = _ratio(supplier, erp_suppliers[erp_idx])
            amount_score = _amount_similarity(amount, erp_net_amounts[erp_idx])
            description_score = _ratio(description, erp_descriptions[erp_idx])
            item_score = 1.0 if po_item and erp_items[erp_idx] and po_item == erp_items[erp_idx] else 0.0

            if po:
                score = (
                    0.43 * po_score + 0.12 * item_score +
                    0.20 * supplier_score + 0.15 * amount_score + 0.10 * description_score
                )
                threshold = po_threshold
            else:
                score = 0.38 * supplier_score + 0.32 * amount_score + 0.30 * description_score
                threshold = no_po_threshold

            # Date closeness is a small tie-breaker; missing/broken dates do not add evidence.
            inv_date = inv_dates.iloc[inv_idx]
            erp_date = erp_dates.iloc[erp_idx]
            if pd.notna(inv_date) and pd.notna(erp_date):
                delta_days = abs((inv_date - erp_date).days)
                if delta_days <= 30:
                    score += 0.02 * (1.0 - delta_days / 30.0)

            if score > best_score:
                best_score, best_idx = score, erp_idx

        if best_idx is not None:
            # PO matches require enough supplier evidence where a supplier name exists.
            supplier_ok = not supplier or _ratio(supplier, erp_suppliers[best_idx]) >= supplier_threshold
            if best_score >= threshold and (not po or supplier_ok):
                matches[inv_idx] = (best_idx, float(min(best_score, 1.0)))
                used_erp.add(best_idx)

    return matches
