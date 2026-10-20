# Telecom Churn Advisor — Database Starter

This starter is aligned with Final Project Ideas #2:
- Predict whether a telecom customer will churn.
- Support customer-facing plan Q&A grounded in the provider's official plan comparison / terms PDFs.

## Files
- `schema.sql`: PostgreSQL schema for customers, cleaned/ML-ready customers, prediction history, official plan document metadata, and chat history.
- `validate_data.py`: read-only data-quality audit of the raw CSV, run before importing anything. Checks for "hidden" missing values (whitespace/`"N/A"`/`"null"` that `pandas.isnull()` would miss), exact duplicate rows, cross-column logical consistency (e.g. `InternetService=No` vs its 6 add-on columns), range sanity, and `customerID` integrity. Does not touch the database.
- `import_dataset.py`: downloads a public mirror of the IBM/Kaggle Telco Customer Churn CSV and imports it into `customers`.
- `clean_data.py`: builds `customers_ml` from `customers` — imputes the known null `total_charges` values, buckets tenure, and binary/one-hot encodes every categorical column so the table is ready for model training. Also exports `customers_ml.csv`. Non-destructive: `customers` is never modified.

Database: **PostgreSQL**, run locally (open it in DBeaver to browse tables — DBeaver is just a
GUI client, the Postgres server itself must be running, see Setup below).

## Setup (PostgreSQL, local)
1. Make sure a local PostgreSQL server is installed and running.
   - macOS (Homebrew): `brew install postgresql@16 && brew services start postgresql@16`
2. Create the database (name must match `PGDATABASE` below, default `telecom_churn`):
   ```bash
   createdb telecom_churn
   ```
   (or create it via DBeaver: New Database Connection → right-click → Create Database)
3. Install the Python driver:
   ```bash
   python3 -m pip install --user psycopg2-binary
   ```
4. Run:
   ```bash
   python3 import_dataset.py   # downloads the CSV if not already present
   python3 validate_data.py
   python3 clean_data.py
   ```
   `import_dataset.py` downloads the CSV (if needed), creates all tables (from `schema.sql`),
   and loads the raw `customers` table. `validate_data.py` prints a data-quality report
   against the raw CSV (8 checks — see Files below); exits non-zero only if it finds
   something *not* already accounted for. `clean_data.py` adds the cleaned,
   feature-engineered `customers_ml` table and exports `customers_ml.csv`, ready for model
   training.
5. Open the `telecom_churn` database in DBeaver to browse tables visually.

### Connection settings
Both scripts default to `localhost:5432`, database `telecom_churn`, with your OS username as
the Postgres user and no password — the standard local Homebrew Postgres setup. If your own
DBeaver connection uses different settings, override with environment variables before
running the scripts:
```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=telecom_churn
export PGUSER=your_postgres_user
export PGPASSWORD=your_password
```

## Validate the import
```bash
psql telecom_churn -c "SELECT COUNT(*) FROM customers;"
psql telecom_churn -c "SELECT churn, COUNT(*) FROM customers GROUP BY churn;"
psql telecom_churn -c "SELECT customer_id, tenure_months, contract, monthly_charges, churn FROM customers LIMIT 5;"
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

## Data validation findings
Run `python3 validate_data.py` to reproduce this report. Summary (8/8 checks pass):
- **Hidden missing values**: `TotalCharges` has 11 rows containing a single whitespace
  character `' '` — invisible to `pandas.isnull()`, but caught here by an explicit
  strip-and-check. No other column has any hidden-missing token (`"N/A"`, `"null"`, `"-"`,
  etc.). Handled by `import_dataset.py` (stripped to NULL) and `clean_data.py` (imputed +
  flagged via `total_charges_was_imputed`).
- No exact duplicate rows, no stray whitespace padding elsewhere, no logical
  inconsistencies between `PhoneService`/`MultipleLines` or `InternetService`/its 6
  add-on columns, `tenure==0` lines up 1:1 with the blank `TotalCharges` rows, all
  numeric ranges are sane, and every `customerID` is unique and correctly formatted.

## Important notes
- The customer dataset is a sample dataset, not a live telecom provider database.
- Do not claim that the plan documents contain any specific price or term until you obtain and ingest the provider's official documents.
- The prediction probability is model output, not certainty that a customer will cancel.
- Running PostgreSQL locally is enough for development; a shared/cloud Postgres instance can be adopted later if the project needs multiple teammates or a deployed API to read the same database.
- Public dataset reference: Kaggle Telco Customer Churn, `blastchar/telco-customer-churn`.
