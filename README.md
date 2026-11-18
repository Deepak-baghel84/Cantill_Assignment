# Cantill.ai — Company A Spend Data Pipeline

A configurable Python pipeline for Company A's SAP goods-receipt export and invoice-tool export. It produces one output row per input record, avoids double-counting matched invoice rows, cleans spend amounts, resolves supplier/item identities, and classifies spend into the runtime taxonomy.

> **Confidentiality:** The task data is confidential. Keep the CSV files under `files/` locally and do not commit them to GitHub. `.gitignore` already excludes them.

## 1. Project layout

```text
Cantill_A_Project/
├── configs/
│   └── company_a.yaml
├── files/                       # Put the four private CSVs here; never commit them
├── outputs/
├── src/
│   ├── classifier.py
│   ├── entities.py
│   ├── matching.py
│   ├── pipeline.py
│   └── utils.py
├── tests/
├── main.py
├── requirements.txt
├── DECISIONS.md
└── APPROACH.md
```

## 2. Expected input files

Place these exact files in the `files/` directory:

- `erp_goods_receipts.csv` — 37,029 ERP rows; fields include `line_id`, `po_number`, `material_number`, `value_zar`, `vendor_id`, and `supplier_name`.
- `invoice_tool_export.csv` — 28,520 invoice-tool rows; fields include `tool_record_id`, `invoice_number`, `po_reference`, `gross_amount`, `vat_rate`, `currency`, and `status`.
- `labels_A.csv` — 37,029 gold-label rows; fields: `record_id`, `category_l1`, `category_l2`, `category_l3`, `supplier_key`, `item_key`.
- `taxonomy_A.csv` — runtime taxonomy; fields: `category_l1`, `category_l2`, `category_l3`.

The YAML maps these four inputs. If the filenames or fields change, update `configs/company_a.yaml` and the schema validation in `src/pipeline.py` as needed.

## 3. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the pipeline

From the project root:

```bash
python main.py --config configs/company_a.yaml
```

The default output is:

```text
outputs/Deepak_Baghel__A.csv
```

A run summary is also written to `outputs/Deepak_Baghel__A__run_summary.json`. Change the output filename in `configs/company_a.yaml` to match your exact submission naming preference.

### Run the required zero-label test

This mode does **not** train on `labels_A.csv`; it classifies using only the taxonomy file. The labels file is consulted only after predictions to calculate evaluation metrics.

```bash
python main.py --config configs/company_a.yaml --zero-shot
```

The same command without `--zero-shot` uses the gold labels to train a supervised classifier for the main Company A output. It first performs a held-out split evaluation and then refits on all usable A labels for final predictions.

In `--zero-shot` mode, `labels_A.csv` is not used for category predictions or supplier/item key resolution. Its category labels are consulted only after prediction to calculate evaluation metrics. This simulates hiding the labels from the tool while retaining a scoring reference.

### Optional semantic zero-shot model

The zero-shot default is an offline TF-IDF taxonomy matcher. For better semantic matching of abbreviated descriptions and an unseen taxonomy, the project can optionally use a locally cached SentenceTransformer model:

```powershell
python -m pip install -r requirements-embeddings.txt
```

Download/cache a suitable model before the blind test (for example, `sentence-transformers/all-MiniLM-L6-v2`), then set `classification.embedding_model` in the YAML to its local model directory. The tool attempts to load it and falls back to TF-IDF if loading fails. Do not depend on downloading model weights during the timed blind test.

## 5. Main processing logic

1. Validate the input schemas and unique record IDs.
2. Treat ERP `value_zar` as net ZAR spend; `document_currency` is not used to reconvert that already-ZAR field.
3. Remove VAT from invoice-tool gross amounts: `net = gross / (1 + VAT rate)`. Blank VAT defaults to 15%.
4. Parse PO references in forms such as `PO 4510109330` and `4510085877/10.0`.
5. Match invoice rows to ERP receipt rows using PO, PO item, supplier similarity, net amount, description, and date proximity. Matching is greedy and one-to-one so one receipt cannot absorb several unrelated invoice lines.
6. Keep the ERP receipt as the canonical record when it is included; exclude its matched invoice-tool row as `DUPLICATE`.
7. Exclude cancelled invoices, non-approved/non-paid invoices, non-procurement records, and duplicate invoice exports using the task's reason codes.
8. Preserve the supplied `supplier_key` and `item_key` for ERP rows. Resolve invoice suppliers/items against ERP aliases and then generate deterministic fallback keys for unmatched entities.
9. Train a supervised TF-IDF + logistic-regression classifier from A labels for the main A run. Also implement a taxonomy-only TF-IDF baseline for the zero-label requirement and unseen trees.
10. Validate that every input record has exactly one output row and that excluded records have zero spend.

## 6. Output columns

`record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence`

- ERP IDs are prefixed `ERP:`.
- Invoice-tool IDs are prefixed `TOOL:`.
- `include` is `1` or `0`.
- Excluded records have `spend=0` and one of the task exclusion codes.
- Company A has three taxonomy levels, so `l4` is blank.

## 7. Important limitations / checks before freezing

- Matching thresholds are starting heuristics. Review sampled matched and unmatched rows; tune the YAML thresholds against actual data.
- Supplier/item keys for ERP are taken from the supplied gold keys. Invoice rows inherit keys from a confident ERP match or use fuzzy/normalized fallback resolution.
- The zero-shot classifier is a lexical similarity baseline. Its confidence is cosine similarity, not a calibrated probability. The supervised classifier's `predict_proba` score is also not guaranteed calibrated.
- Do not report a result until the pipeline has been run on the actual CSVs. The project was authored from the schemas and sample rows shared in the conversation; the underlying CSV contents were not available in this workspace for a full-data execution.
- If Company A contains non-ZAR invoice amounts, the pipeline stops because no Company A FX-rate file was included in the supplied schema. Do not guess the FX rate.
- If an included row has an unparsable amount, the pipeline stops rather than silently dropping it or inventing an exclusion reason code.

## 8. Submission hygiene

- Keep raw task data out of Git.
- Commit code, YAML config, README, `APPROACH.md`, `DECISIONS.md`, tests, and the required final output CSV.
- Do not commit API keys or `.env` files.
