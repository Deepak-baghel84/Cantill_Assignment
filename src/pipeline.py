from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .classifier import (
    ZeroShotTaxonomyClassifier,
    SupervisedTaxonomyClassifier,
    compose_erp_text,
    compose_invoice_text,
    evaluate_supervised_holdout,
    evaluate_zero_shot,
    get_taxonomy_columns,
)
from .entities import ItemResolver, SupplierResolver, labels_by_line
from .matching import match_tool_records
from .utils import (
    clean_text,
    extract_po_and_item,
    normalize_supplier,
    parse_dates,
    parse_number,
    parse_vat_rate,
    stable_key,
)

LOGGER = logging.getLogger("cantill_spend")

ERP_REQUIRED = {
    "line_id", "source_system", "spend_type", "po_number", "po_item",
    "receipt_document", "receipt_item", "posting_date", "material_number",
    "material_short_text", "service_short_text", "material_type", "quantity",
    "uom", "po_uom", "value_zar", "unit_cost_zar", "document_currency",
    "vendor_id", "supplier_name", "site_code", "site_name", "cost_centre",
    "opex_capex", "contract_status", "contract_number",
}
INVOICE_REQUIRED = {
    "tool_record_id", "invoice_number", "invoice_date", "supplier_name_as_captured",
    "po_reference", "line_text", "quantity", "gross_amount", "vat_rate",
    "currency", "status",
}
LABEL_REQUIRED = {"record_id", "category_l1", "category_l2", "category_l3", "supplier_key", "item_key"}
TAXONOMY_REQUIRED = {"category_l1", "category_l2", "category_l3"}
DEFAULT_NON_PROCUREMENT_PATTERNS = [
    r"\bVAT\b", r"\bVALUE ADDED TAX\b", r"\bSALES TAX\b", r"\bTAXES\b",
    r"\bINTERCOMPANY\b", r"\bINTER COMPANY\b", r"\bROYALT(?:Y|IES)\b",
    r"\bPARTNER BILL(?:ING|INGS)\b", r"\bREVENUE\b",
    r"\bFINANC(?:E|ING) COSTS?\b", r"\bINTEREST CHARGES?\b",
    r"\bBANK CHARGES?\b", r"\bINCOME TAX\b",
]


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config, config_path.parent.parent


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}\nPlace the raw task files in the project's files/ folder.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, encoding="utf-8-sig")
    return frame.fillna("").reset_index(drop=True)


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{label} schema mismatch. Missing columns: {missing}.\n"
            f"Received columns: {frame.columns.tolist()}"
        )


def _invoice_net_amount(row: pd.Series, default_vat: float, reporting_currency: str, allow_blank_currency: bool) -> float | None:
    currency = clean_text(row.get("currency", ""))
    expected = clean_text(reporting_currency)
    if currency and currency != expected:
        raise ValueError(
            f"Company A invoice currency '{currency}' is not '{expected}'. "
            "No Company A FX-rate file was included in the supplied schema; add an approved FX source rather than guessing."
        )
    if not currency and not allow_blank_currency:
        raise ValueError("An invoice row has a blank currency and blank currencies are disallowed by config.")
    gross = parse_number(row.get("gross_amount", ""))
    if gross is None:
        return None
    vat = parse_vat_rate(row.get("vat_rate", ""), default_vat)
    return gross / (1.0 + vat)


def _exclusion_reason(
    row: pd.Series,
    patterns: list[re.Pattern[str]],
    is_invoice: bool,
) -> str:
    if is_invoice:
        status = clean_text(row.get("status", ""))
        if "CANCEL" in status:
            return "CANCELLED"
        if status not in {"APPROVED", "PAID"}:
            return "NOT_APPROVED"

    fields = (
        "spend_type", "material_short_text", "service_short_text", "material_type",
        "line_text",
    )
    searchable = " ".join(str(row.get(field, "")) for field in fields)
    if any(pattern.search(searchable) for pattern in patterns):
        return "NON_PROCUREMENT"
    return ""


def _invoice_signature(row: pd.Series, net_amount: float | None) -> tuple[str, ...]:
    invoice_number = clean_text(row.get("invoice_number", ""))
    supplier = normalize_supplier(row.get("supplier_name_as_captured", ""))
    po, item = extract_po_and_item(row.get("po_reference", ""))
    description = clean_text(row.get("line_text", ""))
    amount = "" if net_amount is None else f"{net_amount:.2f}"
    return invoice_number, supplier, po, item, description, amount


def _path_to_output(path: tuple[str, ...], max_levels: int = 4) -> dict[str, str]:
    return {f"l{i + 1}": path[i] if i < len(path) else "" for i in range(max_levels)}


def _make_target_paths(erp: pd.DataFrame, labels_map: dict[str, dict[str, str]]) -> list[tuple[str, ...]]:
    targets = []
    for _, row in erp.iterrows():
        label = labels_map.get(str(row.get("line_id", "")).strip(), {})
        path = tuple(label.get(f"category_l{i}", "") for i in (1, 2, 3))
        path = tuple(x for x in path if x)
        targets.append(path)
    return targets


def _dedupe_counts(output: pd.DataFrame) -> dict[str, int]:
    counts = Counter(str(v) for v in output.loc[output["include"] == 0, "exclusion_reason"].tolist() if str(v))
    return dict(sorted(counts.items()))


def run_pipeline(config_path: str | Path, force_zero_shot: bool = False) -> Path:
    config, project_root = load_config(config_path)
    paths = config.get("paths", {})
    required_paths = ["erp", "invoice_tool", "taxonomy"]
    missing_cfg = [name for name in required_paths if not paths.get(name)]
    if missing_cfg:
        raise ValueError(f"Missing path settings in YAML: {missing_cfg}")

    erp_path = resolve_path(project_root, paths["erp"])
    invoice_path = resolve_path(project_root, paths["invoice_tool"])
    taxonomy_path = resolve_path(project_root, paths["taxonomy"])
    labels_path = resolve_path(project_root, paths["labels"]) if paths.get("labels") else None
    output_path = resolve_path(project_root, config.get("output", "outputs/candidate__A.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading Company A inputs")
    erp = read_csv(erp_path)
    invoice = read_csv(invoice_path)
    taxonomy = read_csv(taxonomy_path)
    validate_columns(erp, ERP_REQUIRED, "ERP goods-receipt file")
    validate_columns(invoice, INVOICE_REQUIRED, "Invoice-tool file")
    validate_columns(taxonomy, TAXONOMY_REQUIRED, "Taxonomy file")
    if erp["line_id"].duplicated().any():
        raise ValueError("ERP line_id contains duplicates; the required output record_id would not be unique.")
    if invoice["tool_record_id"].duplicated().any():
        raise ValueError("Invoice tool_record_id contains duplicates; the required output record_id would not be unique.")

    labels: pd.DataFrame | None = None
    if labels_path and labels_path.exists():
        labels = read_csv(labels_path)
        validate_columns(labels, LABEL_REQUIRED, "Gold-label file")
        if labels["record_id"].duplicated().any():
            raise ValueError("labels_A.csv contains duplicate record_id values.")
    elif labels_path:
        LOGGER.warning("Gold-label file not found at %s; zero-label classification will be used.", labels_path)

    labels_map = labels_by_line(labels)
    taxonomy_columns = get_taxonomy_columns(taxonomy)
    taxonomy_paths = set(tuple(str(row.get(c, "")).strip() for c in taxonomy_columns if str(row.get(c, "")).strip()) for _, row in taxonomy.iterrows())
    target_paths = _make_target_paths(erp, labels_map)
    valid_target_positions = [i for i, path in enumerate(target_paths) if path in taxonomy_paths]
    label_coverage = len(valid_target_positions) / len(erp) if len(erp) else 0.0

    LOGGER.info("Loaded %s ERP rows, %s invoice rows, %s taxonomy paths", len(erp), len(invoice), len(taxonomy))
    LOGGER.info("Gold-label coverage on ERP rows: %.2f%%", 100 * label_coverage)

    reporting_currency = str(config.get("reporting_currency", "ZAR")).upper()
    default_vat = float(config.get("default_vat_rate", 0.15))
    validation_cfg = config.get("validation", {})
    fail_invalid_amount = bool(validation_cfg.get("fail_on_invalid_amount", True))
    allow_blank_currency = bool(validation_cfg.get("allow_blank_currency_as_reporting_currency", True))

    # Per the brief and the `value_zar` field, ERP values are already expressed in ZAR.
    # `document_currency` is retained as source metadata and is not used to reconvert value_zar.
    erp_net_amounts = [parse_number(v) for v in erp["value_zar"].tolist()]
    invoice_net_amounts = [
        _invoice_net_amount(row, default_vat, reporting_currency, allow_blank_currency)
        for _, row in invoice.iterrows()
    ]

    # Entity resolvers preserve the task's golden supplier_key/item_key for ERP records.
    supplier_cfg = config.get("supplier_resolution", {})
    item_cfg = config.get("item_resolution", {})
    # In zero-shot mode, gold labels are not used for supplier/item resolution either.
    # They remain available only for post-prediction evaluation of category accuracy.
    resolver_labels_map = {} if force_zero_shot else labels_map
    supplier_resolver = SupplierResolver(
        erp, resolver_labels_map, fuzzy_threshold=float(supplier_cfg.get("fuzzy_threshold", 0.92))
    )
    item_resolver = ItemResolver(
        erp, resolver_labels_map,
        fuzzy_threshold=float(item_cfg.get("fuzzy_threshold", 0.93)),
        max_token_candidates=int(item_cfg.get("max_token_candidates", 250)),
    )
    erp_supplier_keys = [supplier_resolver.key_for_erp_index(i) for i in range(len(erp))]
    erp_item_keys = [item_resolver.key_for_erp_index(i) for i in range(len(erp))]

    # Always compute taxonomy-only predictions; these support the required zero-label evaluation.
    classifier_cfg = config.get("classification", {})
    zero_shot_classifier = ZeroShotTaxonomyClassifier(
        taxonomy,
        category_keywords=classifier_cfg.get("category_keywords", {}),
        embedding_model=classifier_cfg.get("embedding_model") or None,
    )
    erp_texts = [compose_erp_text(row) for _, row in erp.iterrows()]
    invoice_texts = [compose_invoice_text(row) for _, row in invoice.iterrows()]
    zero_shot_erp_predictions = zero_shot_classifier.predict(erp_texts)
    zero_shot_metrics = evaluate_zero_shot(zero_shot_erp_predictions, target_paths) if labels is not None else {
        "status": "not_available", "reason": "No gold labels were supplied for evaluation."
    }

    use_supervised = bool(classifier_cfg.get("use_supervised_labels", True)) and not force_zero_shot and labels is not None
    supervised_metrics: dict[str, Any] = {
        "status": "not_run", "reason": "Supervised mode disabled or gold labels unavailable."
    }
    if use_supervised:
        supervised_metrics = evaluate_supervised_holdout(
            taxonomy, erp_texts, target_paths,
            test_size=float(classifier_cfg.get("test_size", 0.20)),
            random_state=int(classifier_cfg.get("random_state", 42)),
        )
        supervised_classifier = SupervisedTaxonomyClassifier(
            taxonomy, random_state=int(classifier_cfg.get("random_state", 42))
        )
        supervised_classifier.fit(erp_texts, target_paths)
        erp_predictions = supervised_classifier.predict(erp_texts)
        invoice_predictions = supervised_classifier.predict(invoice_texts)
        classification_mode = "supervised_from_A_gold_labels"
    else:
        erp_predictions = zero_shot_erp_predictions
        invoice_predictions = zero_shot_classifier.predict(invoice_texts)
        classification_mode = "zero_shot_taxonomy_only"

    pattern_texts = config.get("non_procurement_patterns", DEFAULT_NON_PROCUREMENT_PATTERNS)
    patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in pattern_texts]

    # Excluded invoices must not consume a one-to-one ERP match slot.
    matchable_invoice_positions = {
        pos for pos, (_, row) in enumerate(invoice.iterrows())
        if not _exclusion_reason(row, patterns, is_invoice=True)
    }
    matches = match_tool_records(
        erp, invoice, erp_net_amounts, invoice_net_amounts,
        config.get("matching", {}), eligible_invoice_positions=matchable_invoice_positions,
    )
    match_scores = [score for _, score in matches.values()]
    LOGGER.info("Matched %s eligible invoice-tool records to ERP receipts", len(matches))

    erp_reasons: list[str] = []
    for _, row in erp.iterrows():
        erp_reasons.append(_exclusion_reason(row, patterns, is_invoice=False))

    output_rows: list[dict[str, Any]] = []
    invalid_amount_records: list[str] = []
    erp_exclusion_counts = Counter()

    # ERP receipt record is authoritative whenever it matches a tool invoice.
    for pos, (_, row) in enumerate(erp.iterrows()):
        reason = erp_reasons[pos]
        amount = erp_net_amounts[pos]
        if amount is None and not reason:
            invalid_amount_records.append(f"ERP:{row.get('line_id', '')}")
            if fail_invalid_amount:
                continue
        include = 0 if reason else 1
        if reason:
            erp_exclusion_counts[reason] += 1
        path, confidence = erp_predictions[pos]
        output_rows.append({
            "record_id": f"ERP:{row.get('line_id', '')}",
            "include": include,
            "exclusion_reason": reason,
            "supplier_key": erp_supplier_keys[pos],
            "item_key": erp_item_keys[pos],
            "spend": round(float(amount or 0.0), 2) if include else 0.0,
            **_path_to_output(path),
            "confidence": round(float(confidence), 4),
        })

    seen_invoice_signatures: set[tuple[str, ...]] = set()
    invoice_exclusion_counts = Counter()
    for pos, (_, row) in enumerate(invoice.iterrows()):
        reason = _exclusion_reason(row, patterns, is_invoice=True)
        net_amount = invoice_net_amounts[pos]
        signature = _invoice_signature(row, net_amount)

        if not reason:
            if signature in seen_invoice_signatures:
                reason = "DUPLICATE"
            else:
                seen_invoice_signatures.add(signature)

        matched_erp_pos: int | None = matches[pos][0] if pos in matches else None
        if not reason and matched_erp_pos is not None:
            # If ERP has already been excluded as non-procurement, mirror that reason.
            reason = erp_reasons[matched_erp_pos] or "DUPLICATE"

        if matched_erp_pos is not None:
            supplier_key = erp_supplier_keys[matched_erp_pos]
            item_key = erp_item_keys[matched_erp_pos]
            path, confidence = erp_predictions[matched_erp_pos]
        else:
            supplier_key = supplier_resolver.resolve_invoice(row.get("supplier_name_as_captured", ""))
            item_key = item_resolver.resolve_invoice(row.get("line_text", ""))
            path, confidence = invoice_predictions[pos]

        if net_amount is None and not reason:
            invalid_amount_records.append(f"TOOL:{row.get('tool_record_id', '')}")
            if fail_invalid_amount:
                continue
        include = 0 if reason else 1
        if reason:
            invoice_exclusion_counts[reason] += 1
        output_rows.append({
            "record_id": f"TOOL:{row.get('tool_record_id', '')}",
            "include": include,
            "exclusion_reason": reason,
            "supplier_key": supplier_key,
            "item_key": item_key,
            "spend": round(float(net_amount or 0.0), 2) if include else 0.0,
            **_path_to_output(path),
            "confidence": round(float(confidence), 4),
        })

    if invalid_amount_records and fail_invalid_amount:
        preview = ", ".join(invalid_amount_records[:10])
        raise ValueError(
            f"Found {len(invalid_amount_records)} included records with unparsable spend amounts. "
            f"Examples: {preview}. Fix source data or document a defensible policy; no output was written."
        )

    output = pd.DataFrame(output_rows, columns=[
        "record_id", "include", "exclusion_reason", "supplier_key", "item_key",
        "spend", "l1", "l2", "l3", "l4", "confidence",
    ])
    expected_rows = len(erp) + len(invoice)
    if len(output) != expected_rows:
        raise AssertionError(f"Output has {len(output)} rows; expected {expected_rows}. Check invalid amount handling.")
    if output["record_id"].duplicated().any():
        raise AssertionError("Output contains duplicate record_id values.")
    if not output["include"].isin([0, 1]).all():
        raise AssertionError("include must contain only 0 or 1.")
    if ((output["confidence"] < 0) | (output["confidence"] > 1)).any():
        raise AssertionError("confidence must be between 0 and 1.")
    if (output.loc[output["include"] == 0, "spend"] != 0).any():
        raise AssertionError("Excluded records must have spend=0.")

    # Every emitted nonblank category path must be present in the runtime taxonomy.
    valid_paths = set(tuple(str(row.get(f"category_l{i}", "")).strip() for i in range(1, 4)) for _, row in taxonomy.iterrows())
    for _, row in output.iterrows():
        path = tuple(str(row.get(f"l{i}", "")).strip() for i in range(1, 4))
        path = tuple(v for v in path if v)
        if path and path not in valid_paths:
            raise AssertionError(f"Predicted category path not in taxonomy: {path}")

    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    erp_dates = parse_dates(erp["posting_date"])
    invoice_dates = parse_dates(invoice["invoice_date"])
    erp_date_missing = erp["posting_date"].astype(str).str.strip().eq("")
    invoice_date_missing = invoice["invoice_date"].astype(str).str.strip().eq("")
    erp_date_invalid = (~erp_date_missing) & erp_dates.isna()
    invoice_date_invalid = (~invoice_date_missing) & invoice_dates.isna()
    missing_vat = invoice["vat_rate"].astype(str).str.strip().eq("")

    summary = {
        "company": "A",
        "erp_input_file": str(erp_path),
        "invoice_input_file": str(invoice_path),
        "labels_file": str(labels_path) if labels_path else None,
        "taxonomy_file": str(taxonomy_path),
        "erp_rows": len(erp),
        "invoice_rows": len(invoice),
        "output_rows": len(output),
        "gold_label_rows": len(labels) if labels is not None else 0,
        "gold_label_coverage_on_erp": round(label_coverage, 4),
        "matched_invoice_to_erp": len(matches),
        "matching_score_mean": round(sum(match_scores) / len(match_scores), 4) if match_scores else None,
        "matching_score_min": round(min(match_scores), 4) if match_scores else None,
        "erp_excluded_by_reason": dict(sorted(erp_exclusion_counts.items())),
        "invoice_excluded_by_reason": dict(sorted(invoice_exclusion_counts.items())),
        "excluded_by_reason_total": _dedupe_counts(output),
        "included_rows": int((output["include"] == 1).sum()),
        "excluded_rows": int((output["include"] == 0).sum()),
        "included_spend_zar": round(float(output["spend"].sum()), 2),
        "classification_mode": classification_mode,
        "zero_shot_backend": zero_shot_classifier.embedding_backend,
        "supervised_holdout_evaluation": supervised_metrics,
        "zero_shot_taxonomy_only_evaluation": zero_shot_metrics,
        "average_classification_confidence": round(float(output.loc[output["include"] == 1, "confidence"].mean()), 4)
            if (output["include"] == 1).any() else 0.0,
        "confidence_note": "Supervised values are classifier probability scores; zero-shot values are cosine similarities. Neither is guaranteed calibrated.",
        "erp_missing_posting_date_count": int(erp_date_missing.sum()),
        "erp_unparseable_nonblank_posting_date_count": int(erp_date_invalid.sum()),
        "invoice_missing_date_count": int(invoice_date_missing.sum()),
        "invoice_unparseable_nonblank_date_count": int(invoice_date_invalid.sum()),
        "invoice_missing_vat_rate_count_defaulted_to_15_percent": int(missing_vat.sum()),
        "invoice_unmatched_to_erp_count": int(len(invoice) - len(matches)),
        "invalid_amount_records": invalid_amount_records,
        "assumptions": [
            "ERP value_zar is treated as net ZAR spend.",
            "Invoice gross_amount is divided by (1 + VAT); blank VAT defaults to 15%.",
            "Only Approved/Paid invoice rows are eligible; cancelled lines are CANCELLED.",
            "A confident one-to-one invoice-to-ERP match causes the invoice row to be excluded as DUPLICATE.",
            "The ERP record is retained as the canonical record when it is included.",
            "Supplier_key and item_key from labels_A.csv are preserved for ERP records.",
            "Company A category paths are read from taxonomy_A.csv at runtime.",
        ],
    }
    summary_path = output_path.with_name(output_path.stem + "__run_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote output: %s", output_path)
    LOGGER.info("Included spend: %s %s", summary["included_spend_zar"], reporting_currency)
    LOGGER.info("Classification mode: %s", classification_mode)
    return output_path
