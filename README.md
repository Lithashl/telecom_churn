# Telecom Churn Advisor — Database Starter

This starter is aligned with Final Project Ideas #2:
- Predict whether a telecom customer will churn.
- Support customer-facing plan Q&A grounded in the provider's official plan comparison / terms PDFs.

## Files
- `schema.sql`: SQLite schema for customers, cleaned/ML-ready customers, prediction history, official plan document metadata, and chat history.
- `import_dataset.py`: downloads a public mirror of the IBM/Kaggle Telco Customer Churn CSV and imports it into `customers`.
- `clean_data.py`: builds `customers_ml` from `customers` — imputes the known null `total_charges` values, buckets tenure, and binary/one-hot encodes every categorical column so the table is ready for model training. Also exports `customers_ml.csv`. Non-destructive: `customers` is never modified.
- `telecom_churn.db`: created locally after running the import script; not included yet because the dataset must be downloaded in your environment.

## Setup (VS Code / Windows)
1. Create a project folder and place these files in it.
2. Open the folder in VS Code.
3. Open Terminal and run:
   ```bash
   python import_dataset.py
   python clean_data.py
   ```
4. `import_dataset.py` creates `telecom_churn.db` with the raw `customers` table.
   `clean_data.py` adds the cleaned, feature-engineered `customers_ml` table and
   `customers_ml.csv`, ready for model training.
5. Install a SQLite extension in VS Code if you want to browse tables visually, or inspect it with Python.

## Validate the import
Open Python and run:
```python
import sqlite3
con = sqlite3.connect("telecom_churn.db")
print(con.execute("SELECT COUNT(*) FROM customers").fetchone())
print(con.execute("SELECT churn, COUNT(*) FROM customers GROUP BY churn").fetchall())
print(con.execute("SELECT customer_id, tenure_months, contract, monthly_charges, churn FROM customers LIMIT 5").fetchall())
con.close()
```

Expected dataset size is 7,043 customer rows. `TotalCharges` has blank values in the source; this import converts blanks to SQL NULL rather than inventing values. Handle missing values explicitly during data cleaning/model preparation.

## Tables
- `customers`: original 21 dataset fields, renamed to consistent snake_case. Raw import, never modified after load — kept as the audit trail.
- `customers_ml`: built by `clean_data.py` from `customers`. No nulls, every categorical
  binary/one-hot encoded, plus derived features (`tenure_group`, `avg_monthly_spend`).
  The 11 customers with `tenure_months = 0` had a blank `TotalCharges` in the source (they
  haven't been billed yet); their `total_charges` is imputed to `monthly_charges` (their
  first-month rate) and flagged via `total_charges_was_imputed = 1` so the imputation is never
  hidden from the model or a grader.
- `churn_predictions`: stores probability, predicted label, model version, and timestamp.
- `plan_documents`: metadata for official provider PDFs. Populate only with real documents from the provider you choose.
- `chat_history`: conversation history with an optional reference to the source document.

## Important notes
- The customer dataset is a sample dataset, not a live telecom provider database.
- Do not claim that the plan documents contain any specific price or term until you obtain and ingest the provider's official documents.
- The prediction probability is model output, not certainty that a customer will cancel.
- For a classroom MVP, SQLite is sufficient. PostgreSQL can be adopted later if the project needs a shared database/API deployment.
- Public dataset reference: Kaggle Telco Customer Churn, `blastchar/telco-customer-churn`.
