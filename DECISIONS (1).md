# DECISIONS.md — Company A

## Confirmed task rules

- Company A reports spend in ZAR.
- ERP goods-receipt values are net of tax.
- Invoice-tool amounts are gross; remove VAT using the supplied rate. If VAT is blank, assume 15%.
- When the same purchase appears in both systems, retain the ERP record and exclude the invoice-tool row as `DUPLICATE`.
- An invoice-tool export repeated within the tool file is counted once.
- Only approved or paid invoices count.
- Exclusion codes are `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, and `DUPLICATE`.
- Classification must use the category tree read from a file at runtime.

## Corrected input schemas shared by the candidate

- `erp_goods_receipts.csv`: 37,029 rows and 26 columns. This is the ERP goods-receipt source, containing `line_id`, `po_number`, `po_item`, `receipt_document`, `material_number`, `value_zar`, `vendor_id`, `supplier_name`, descriptions, and procurement metadata.
- `invoice_tool_export.csv`: 28,520 rows and 11 columns. This is the invoice-tool source, containing `tool_record_id`, invoice number/date, captured supplier, PO reference, line text, gross amount, VAT rate, currency, and status.
- `labels_A.csv`: 37,029 rows and 6 columns: `record_id`, `category_l1`, `category_l2`, `category_l3`, `supplier_key`, `item_key`. ERP label IDs use the `ERP:<line_id>` convention.
- `taxonomy_A.csv`: 40 taxonomy paths with `category_l1`, `category_l2`, `category_l3`.

## Implementation decisions / assumptions

1. **Canonical source:** ERP receipt rows are retained when a confident invoice-tool match exists; the invoice row is assigned `DUPLICATE` if the ERP row is included.
2. **Matching:** Normalize numeric PO values and invoice PO strings. Score candidates using PO, PO item, supplier, VAT-exclusive amount, description and date. Enforce one-to-one matching to avoid assigning many invoice rows to the same receipt. This is an assumption; verify against actual data.
3. **Invoice duplicate signature:** Use normalized invoice number, supplier, PO/item, description and net amount. The first eligible occurrence is retained; later exact signature repeats are `DUPLICATE`.
4. **Supplier identity:** Preserve the supplied golden supplier key for every ERP row. For invoice records, inherit the key from an ERP match or use conservative fuzzy matching against ERP supplier aliases. Otherwise create a stable normalized key.
5. **Item identity:** Preserve the supplied golden item key for every ERP row. For invoice records, inherit from an ERP match, use a fuzzy description match against ERP descriptions, or create a stable description-based key.
6. **Status:** `APPROVED` and `PAID` are accepted. Statuses containing `CANCEL` map to `CANCELLED`; other statuses map to `NOT_APPROVED`.
7. **Non-procurement:** Use configurable conservative keyword patterns for taxes, intercompany activity, royalties, partner billings, revenue, financing/interest and bank charges. Review samples for false positives/negatives.
8. **Dates:** Parse using mixed date formats with day-first handling. Missing or invalid dates are counted in the run summary; they are not automatically excluded because the task does not define a date-related exclusion code.
9. **Currency:** ERP `value_zar` is treated as already expressed in ZAR, as indicated by the task field and brief; `document_currency` is retained as metadata and is not used to reconvert it. Invoice rows must be ZAR (blank may be treated as ZAR if enabled in config). If a non-ZAR invoice appears, the pipeline stops because no Company A FX-rate source is specified in the supplied files.
10. **Amounts:** An included record with an unparsable amount causes the run to stop. The task defines no malformed-amount reason code, so the pipeline does not silently assign a new one.
11. **Classification:** The default A run uses supervised training from `labels_A.csv`, with a held-out evaluation before final refit. `--zero-shot` disables labelled training and label-based entity keys; it classifies from the runtime taxonomy alone. Labels are consulted only after prediction to measure zero-shot accuracy. An optional local SentenceTransformer backend can be configured; TF-IDF remains the offline fallback.
12. **Confidence:** Supervised confidence is the model's maximum predicted class probability; zero-shot confidence is cosine similarity. Neither is guaranteed to be calibrated.

## Required audit before freezing

- Review at least 50 high-confidence and 50 borderline invoice-to-ERP matches.
- Review unmatched invoice rows with approved/paid status.
- Review all `NON_PROCUREMENT` rows and a sample of included ERP/invoice records.
- Verify that gross/net invoice amounts align with receipt values after VAT removal.
- Compare supervised held-out classification metrics with zero-shot taxonomy-only metrics.
- Measure full pipeline runtime on the actual files and retain the generated run summary.
