# Libraries
from pathlib import Path

import joblib
import pandas as pd

MODEL_FILE = Path("churn_model.joblib")
DATA_FILE = Path("customers_ml.csv")


def predict_customer(customer_id):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "churn_model.joblib tidak ditemukan. "
            "Jalankan train_model.py terlebih dahulu."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "customers_ml.csv tidak ditemukan."
        )

    # Load the model
    saved = joblib.load(MODEL_FILE)

    model = saved["model"]
    threshold = saved["threshold"]
    feature_columns = saved["feature_columns"]

    # Read customer ID
    df = pd.read_csv(DATA_FILE, dtype={"customer_id": str})

    if "customer_id" not in df.columns:
        raise ValueError(
            "Kolom customer_id tidak ditemukan."
        )

    # Search customer ID
    customer = df[
        df["customer_id"].str.strip() == str(customer_id).strip()
    ]

    if customer.empty:
        print(f"Pelanggan dengan ID {customer_id} tidak ditemukan.")
        return

    if len(customer) > 1:
        raise ValueError(
            f"ID {customer_id} ditemukan lebih dari satu kali."
        )

    # Features
    X = customer[feature_columns].copy()

    if X.isnull().any().any():
        raise ValueError(
            "Data pelanggan memiliki nilai kosong."
        )

    # Calculating churn probability
    probability = model.predict_proba(X)[0, 1]

    # Treshold
    prediction = int(probability >= threshold)

    # Results
    print("\n=== HASIL PREDIKSI CHURN ===")
    print(f"Customer ID       : {customer_id}")
    print(f"Probabilitas churn: {probability:.2%}")
    print(f"Threshold         : {threshold:.2%}")

    if prediction == 1:
        print("Prediksi          : BERISIKO CHURN")
    else:
        print("Prediksi          : TIDAK DIKLASIFIKASIKAN CHURN")

    return {
        "customer_id": customer_id,
        "churn_probability": float(probability),
        "threshold": float(threshold),
        "prediction": prediction,
    }


if __name__ == "__main__":
    customer_id = input(
        "Masukkan customer_id yang ingin diprediksi: "
    ).strip()

    if not customer_id:
        print("Customer ID tidak boleh kosong.")
    else:
        predict_customer(customer_id)