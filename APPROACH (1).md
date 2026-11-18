# APPROACH.md — Company A

## Pipeline design

The pipeline is driven by `configs/company_a.yaml`. It validates the four input schemas, parses ERP and invoice amounts, normalizes dates and purchase-order references, matches the two source systems, assigns exclusions, resolves entity keys, predicts taxonomy paths, and writes one row per source record. Taxonomy category names are loaded from the taxonomy CSV at runtime and are not embedded in Python code.

## Matching and double-counting

ERP goods receipts are treated as the canonical record when an invoice-tool line refers to the same purchase. Candidate matching uses normalized PO number as the primary signal, then PO item, supplier-name similarity, net amount similarity, description similarity, and date proximity. Candidate matches are greedily assigned one-to-one; a confident matched invoice-tool row is marked `DUPLICATE`, while the ERP row remains included unless its own content triggers an exclusion. Invoice-tool exports repeated within the tool file are separately deduplicated using invoice number, supplier, PO, item, line description, and VAT-exclusive amount.

This is a heuristic matching method. Match thresholds must be reviewed using actual matched/unmatched samples; one-to-one matching assumes an invoice-tool line corresponds to at most one ERP receipt line.

## Spend and exclusions

ERP `value_zar` is treated as net ZAR. Invoice-tool `gross_amount` is converted to net by dividing by `(1 + VAT rate)`; a blank VAT rate uses 15%. Only `Approved` and `Paid` invoice rows are eligible. Cancelled rows use `CANCELLED`; other ineligible statuses use `NOT_APPROVED`. Configurable text patterns identify likely taxes, intercompany charges, royalties, partner billings, revenue, financing and similar non-procurement records as `NON_PROCUREMENT`.

## Supplier and item resolution

For ERP rows, the provided `supplier_key` and `item_key` in `labels_A.csv` are preserved. Invoice rows that match ERP inherit those keys. Unmatched invoice supplier names are normalized and fuzzy-matched to ERP supplier aliases; unmatched item descriptions are compared against ERP material/service descriptions. Deterministic fallback keys are generated for unresolved entities. These invoice-side resolutions are heuristics; inspect their grouping quality before submission.

## Classification

The main Company A run uses a supervised TF-IDF feature model (word n-grams plus character n-grams) and logistic regression trained on the supplied ERP category labels. It first reports a held-out 80/20 evaluation and then refits on all valid labels for final output. The `--zero-shot` mode bypasses labelled training and label-based supplier/item keys. It compares line text with full taxonomy paths using word/character TF-IDF similarity, or an optional locally cached SentenceTransformer model if configured. The A labels are consulted only after predictions to calculate evaluation metrics. This provides a separate taxonomy-only test and is the fallback path for a company with no labelled examples. The taxonomy is always loaded at runtime.

## Accuracy, runtime and cost

- **Supervised accuracy:** Generated in `__run_summary.json` by a held-out split when the pipeline is run; do not fill in a number until it has been measured on the actual CSVs.
- **Zero-label accuracy:** Also generated in the run summary by predicting from the taxonomy alone and comparing predictions to A labels only after inference.
- **Runtime:** Must be measured on the machine used for submission with the actual 65,549 input records.
- **LLM/API cost:** 0. The current implementation is local and uses scikit-learn; it makes no paid API calls.

## Known limitations

The zero-shot TF-IDF baseline may struggle with short, abbreviated or semantically distant descriptions. An optional local sentence-embedding model or domain synonym dictionary may improve this, but must be evaluated and included in the frozen build if adopted. Matching, fuzzy supplier/item resolution, and non-procurement rules remain heuristic and should be audited against the actual data.
