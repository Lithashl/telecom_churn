from pathlib import Path
import csv
import sqlite3
import urllib.request

# Dataset: IBM/Kaggle Telco Customer Churn, 7,043 rows.
# This public mirror is provided for convenience; verify dataset provenance for your submission.
DATA_URL = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_FILE = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
DB_FILE = Path("telecom_churn.db")
SCHEMA_FILE = Path("schema.sql")

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

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))

    source_columns = list(COLUMN_MAP.keys())
    target_columns = [COLUMN_MAP[c] for c in source_columns]
    placeholders = ",".join(["?"] * len(target_columns))
    columns_sql = ",".join(target_columns)
    sql = f"INSERT OR REPLACE INTO customers ({columns_sql}) VALUES ({placeholders})"

    inserted = 0
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = [convert(row.get(c, ""), c) for c in source_columns]
            conn.execute(sql, values)
            inserted += 1

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    churn_counts = conn.execute(
        "SELECT churn, COUNT(*) FROM customers GROUP BY churn ORDER BY churn"
    ).fetchall()
    conn.close()

    print(f"Import complete. Processed rows: {inserted}; unique customers in DB: {count}")
    print("Churn distribution:", churn_counts)
    print(f"Database created at: {DB_FILE.resolve()}")

if __name__ == "__main__":
    main()
