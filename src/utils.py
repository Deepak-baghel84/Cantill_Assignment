from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .utils import clean_text, normalize_supplier, stable_key


def labels_by_line(labels: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    """Return gold labels keyed by the ERP line_id (labels use ERP:<line_id>)."""
    result: dict[str, dict[str, str]] = {}
    if labels is None or labels.empty:
        return result
    for _, row in labels.iterrows():
        record_id = str(row.get("record_id", "")).strip()
        if record_id.startswith("ERP:"):
            line_id = record_id[4:]
        else:
            line_id = record_id
        if not line_id:
            continue
        result[line_id] = {
            key: str(row.get(key, "")).strip()
            for key in ("category_l1", "category_l2", "category_l3", "supplier_key", "item_key")
        }
    return result


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


class SupplierResolver:
    """Resolve supplier names to supplied golden keys or deterministic fallback keys."""

    def __init__(self, erp: pd.DataFrame, labels_map: dict[str, dict[str, str]], fuzzy_threshold: float = 0.92):
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.row_keys: list[str] = []
        self.name_to_keys: dict[str, set[str]] = defaultdict(set)
        self.canonical_names: list[str] = []
        self.canonical_keys: list[str] = []
        seen: dict[str, str] = {}

        for _, row in erp.iterrows():
            line_id = str(row.get("line_id", "")).strip()
            name = normalize_supplier(row.get("supplier_name", ""))
            key = labels_map.get(line_id, {}).get("supplier_key", "").strip()
            if not key:
                key = stable_key("SG", name or row.get("vendor_id", "UNKNOWN"))
            self.row_keys.append(key)
            if name:
                self.name_to_keys[name].add(key)
                if name not in seen:
                    seen[name] = key

        self.canonical_names = list(seen.keys())
        self.canonical_keys = [seen[name] for name in self.canonical_names]

    def key_for_erp_index(self, index: int) -> str:
        return self.row_keys[index]

    def resolve_invoice(self, supplier_name: Any) -> str:
        normalized = normalize_supplier(supplier_name)
        if not normalized:
            return stable_key("SG", "UNKNOWN SUPPLIER")
        exact = self.name_to_keys.get(normalized)
        if exact:
            return sorted(exact)[0]

        best_score = 0.0
        best_key = ""
        # Supplier cardinality is normally far smaller than invoice-line cardinality.
        for candidate, key in zip(self.canonical_names, self.canonical_keys):
            score = _similarity(normalized, candidate)
            if score > best_score:
                best_score, best_key = score, key
        if best_score >= self.fuzzy_threshold and best_key:
            return best_key
        return stable_key("SG", normalized)


class ItemResolver:
    """Resolve material/service descriptions using golden keys and token blocking."""

    def __init__(
        self,
        erp: pd.DataFrame,
        labels_map: dict[str, dict[str, str]],
        fuzzy_threshold: float = 0.93,
        max_token_candidates: int = 250,
    ):
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.max_token_candidates = max(10, int(max_token_candidates))
        self.row_keys: list[str] = []
        self.signatures: list[str] = []
        self.token_index: dict[str, set[int]] = defaultdict(set)
        exact_keys: dict[str, set[str]] = defaultdict(set)

        for index, (_, row) in enumerate(erp.iterrows()):
            line_id = str(row.get("line_id", "")).strip()
            material_number = clean_text(row.get("material_number", ""))
            description = clean_text(" ".join([
                str(row.get("material_short_text", "")),
                str(row.get("service_short_text", "")),
                str(row.get("material_type", "")),
            ]))
            signature = material_number or description
            key = labels_map.get(line_id, {}).get("item_key", "").strip()
            if not key:
                key = stable_key("IG", signature or f"ROW {index}")
            self.row_keys.append(key)
            self.signatures.append(signature)
            if signature:
                exact_keys[signature].add(key)
                for token in set(signature.split()):
                    if len(token) >= 2:
                        self.token_index[token].add(index)

        self.exact_keys = dict(exact_keys)

    def key_for_erp_index(self, index: int) -> str:
        return self.row_keys[index]

    def resolve_invoice(self, description: Any) -> str:
        normalized = clean_text(description)
        if not normalized:
            return stable_key("IG", "UNKNOWN ITEM")
        exact = self.exact_keys.get(normalized)
        if exact:
            return sorted(exact)[0]

        tokens = [token for token in set(normalized.split()) if len(token) >= 2]
        counts: dict[int, int] = defaultdict(int)
        for token in tokens:
            for index in self.token_index.get(token, ()):
                counts[index] += 1
        candidates = sorted(counts, key=lambda idx: (-counts[idx], idx))[: self.max_token_candidates]

        best_score = 0.0
        best_key = ""
        for index in candidates:
            candidate = self.signatures[index]
            score = _similarity(normalized, candidate)
            if score > best_score:
                best_score, best_key = score, self.row_keys[index]
        if best_score >= self.fuzzy_threshold and best_key:
            return best_key
        return stable_key("IG", normalized)
