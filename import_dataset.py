from pathlib import Path
import csv
import getpass
import os
import urllib.request

import psycopg2

# Dataset: IBM/Kaggle Telco Customer Churn, 7,043 rows.
# This public mirror is provided for convenience; verify dataset provenance for your submission.
DATA_URL = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_FILE = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
SCHEMA_FILE = Path("schema.sql")

# Local PostgreSQL connection. Override with env vars if your DBeaver connection differs.
DB_PARAMS = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
    dbname=os.environ.get("PGDATABASE", "telecom_churn"),
    user=os.environ.get("PGUSER", getpass.getuser()),
    password=os.environ.get("PGPASSWORD", ""),
)

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

def download_dataset():
    if DATA_FILE.exists():
        print(f"Using existing dataset: {DATA_FILE}")
        return
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"Downloaded: {DATA_FILE}")

def convert(value, column):
    value = value.strip() if isinstance(value, str) else value
    if value == "":
        return None
    if column in ("SeniorCitizen", "tenure"):
        return int(value)
    if column in ("MonthlyCharges", "TotalCharges"):
        return float(value)
    return value

def main():
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError("schema.sql must be in the same folder as this script.")
    download_dataset()

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))

    source_columns = list(COLUMN_MAP.keys())
    target_columns = [COLUMN_MAP[c] for c in source_columns]
    placeholders = ",".join(["%s"] * len(target_columns))
    columns_sql = ",".join(target_columns)
    update_sql = ",".join(f"{c} = EXCLUDED.{c}" for c in target_columns if c != "customer_id")
    sql = (
        f"INSERT INTO customers ({columns_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT (customer_id) DO UPDATE SET {update_sql}"
    )

    inserted = 0
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = [convert(row.get(c, ""), c) for c in source_columns]
            cur.execute(sql, values)
            inserted += 1

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM customers")
    count = cur.fetchone()[0]
    cur.execute("SELECT churn, COUNT(*) FROM customers GROUP BY churn ORDER BY churn")
    churn_counts = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Import complete. Processed rows: {inserted}; unique customers in DB: {count}")
    print("Churn distribution:", churn_counts)
    print(f"Database: {DB_PARAMS['dbname']} on {DB_PARAMS['host']}:{DB_PARAMS['port']}")

if __name__ == "__main__":
    main()
