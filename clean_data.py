from pathlib import Path
import csv
import sqlite3

# Builds `customers_ml` from the raw `customers` table: handles the one known data-quality
# issue (null total_charges for brand-new customers) and encodes categoricals so the table
# is ready for model training. `customers` itself is never modified or deleted from.
DB_FILE = Path("telecom_churn.db")
SCHEMA_FILE = Path("schema.sql")
CSV_EXPORT_FILE = Path("customers_ml.csv")

YES_NO_FIELDS = {
    "partner": "has_partner",
    "dependents": "has_dependents",
    "phone_service": "has_phone_service",
    "paperless_billing": "has_paperless_billing",
    "churn": "churn",
}

ONE_HOT_FIELDS = {
    "multiple_lines": {
        "No": "multiple_lines_no",
        "Yes": "multiple_lines_yes",
        "No phone service": "multiple_lines_no_phone_service",
    },
    "internet_service": {
        "DSL": "internet_service_dsl",
        "Fiber optic": "internet_service_fiber_optic",
        "No": "internet_service_no",
    },
    "online_security": {
        "No": "online_security_no",
        "Yes": "online_security_yes",
        "No internet service": "online_security_no_internet_service",
    },
    "online_backup": {
        "No": "online_backup_no",
        "Yes": "online_backup_yes",
        "No internet service": "online_backup_no_internet_service",
    },
    "device_protection": {
        "No": "device_protection_no",
        "Yes": "device_protection_yes",
        "No internet service": "device_protection_no_internet_service",
    },
    "tech_support": {
        "No": "tech_support_no",
        "Yes": "tech_support_yes",
        "No internet service": "tech_support_no_internet_service",
    },
    "streaming_tv": {
        "No": "streaming_tv_no",
        "Yes": "streaming_tv_yes",
        "No internet service": "streaming_tv_no_internet_service",
    },
    "streaming_movies": {
        "No": "streaming_movies_no",
        "Yes": "streaming_movies_yes",
        "No internet service": "streaming_movies_no_internet_service",
    },
    "contract": {
        "Month-to-month": "contract_month_to_month",
        "One year": "contract_one_year",
        "Two year": "contract_two_year",
    },
    "payment_method": {
        "Electronic check": "payment_method_electronic_check",
        "Mailed check": "payment_method_mailed_check",
        "Bank transfer (automatic)": "payment_method_bank_transfer_automatic",
        "Credit card (automatic)": "payment_method_credit_card_automatic",
    },
}

# Upper bound of each tenure bucket, in months.
TENURE_BUCKETS = [(12, "0-12"), (24, "13-24"), (48, "25-48"), (72, "49-72")]


def tenure_group(tenure_months):
    for upper_bound, label in TENURE_BUCKETS:
        if tenure_months <= upper_bound:
            return label
    return TENURE_BUCKETS[-1][1]


def one_hot_row(field, value):
    columns = ONE_HOT_FIELDS[field]
    if value not in columns:
        raise ValueError(f"Unexpected value {value!r} for field {field!r}")
    return {col: (1 if col == columns[value] else 0) for col in columns.values()}


def build_ml_row(row):
    total_charges_was_imputed = row["total_charges"] is None
    total_charges = row["monthly_charges"] if total_charges_was_imputed else row["total_charges"]
    avg_monthly_spend = (
        row["monthly_charges"] if row["tenure_months"] == 0 else total_charges / row["tenure_months"]
    )

    record = {
        "customer_id": row["customer_id"],
        "tenure_months": row["tenure_months"],
        "tenure_group": tenure_group(row["tenure_months"]),
        "monthly_charges": row["monthly_charges"],
        "total_charges": total_charges,
        "total_charges_was_imputed": int(total_charges_was_imputed),
        "avg_monthly_spend": avg_monthly_spend,
        "senior_citizen": row["senior_citizen"],
        "gender_male": 1 if row["gender"] == "Male" else 0,
    }
    for field, target in YES_NO_FIELDS.items():
        record[target] = 1 if row[field] == "Yes" else 0

    for field in ONE_HOT_FIELDS:
        record.update(one_hot_row(field, row[field]))

    return record


def main():
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError("schema.sql must be in the same folder as this script.")
    if not DB_FILE.exists():
        raise FileNotFoundError("telecom_churn.db not found. Run import_dataset.py first.")

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))

    source_rows = conn.execute("SELECT * FROM customers").fetchall()
    ml_rows = [build_ml_row(row) for row in source_rows]

    columns = list(ml_rows[0].keys())
    placeholders = ",".join(["?"] * len(columns))
    columns_sql = ",".join(columns)
    sql = f"INSERT OR REPLACE INTO customers_ml ({columns_sql}) VALUES ({placeholders})"
    conn.executemany(sql, [[r[c] for c in columns] for r in ml_rows])
    conn.commit()

    # Validation report.
    ml_count = conn.execute("SELECT COUNT(*) FROM customers_ml").fetchone()[0]
    null_columns = [c for c in columns if conn.execute(
        f"SELECT COUNT(*) FROM customers_ml WHERE {c} IS NULL"
    ).fetchone()[0] > 0]
    imputed_count = conn.execute(
        "SELECT COUNT(*) FROM customers_ml WHERE total_charges_was_imputed = 1"
    ).fetchone()[0]
    churn_counts = [tuple(r) for r in conn.execute(
        "SELECT churn, COUNT(*) FROM customers_ml GROUP BY churn ORDER BY churn"
    ).fetchall()]

    with CSV_EXPORT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in ml_rows:
            writer.writerow([r[c] for c in columns])

    conn.close()

    print(f"customers rows: {len(source_rows)}; customers_ml rows: {ml_count}")
    print(f"Columns with any NULL in customers_ml: {null_columns or 'none'}")
    print(f"Rows with imputed total_charges (tenure=0): {imputed_count}")
    print("Churn distribution (0=No, 1=Yes):", churn_counts)
    print(f"CSV export written to: {CSV_EXPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()
