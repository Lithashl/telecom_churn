
# Libraries
from pathlib import Path

import joblib
import pandas as pd

MODEL_FILE = Path("churn_models.joblib")
DATA_FILE = Path("customers_ml.csv")


def predict_customer(customer_id, model_choice="ensemble"):
    # Check files
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "churn_models.joblib tidak ditemukan. "
            "Jalankan revised_model.py terlebih dahulu."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "customers_ml.csv tidak ditemukan."
        )

    # Load model package
    saved = joblib.load(MODEL_FILE)

    # Read customer data
    df = pd.read_csv(DATA_FILE, dtype={"customer_id": str})

    if "customer_id" not in df.columns:
        raise ValueError("Kolom customer_id tidak ditemukan.")

    # Search customer ID
    customer = df[
        df["customer_id"].str.strip() == str(customer_id).strip()
    ]

    if customer.empty:
        print(f"Pelanggan dengan ID {customer_id} tidak ditemukan.")
        return None

    if len(customer) > 1:
        raise ValueError(
            f"ID {customer_id} ditemukan lebih dari satu kali."
        )

    # Get feature columns
    feature_columns = saved["feature_columns"]

    missing_columns = [
        col for col in feature_columns if col not in customer.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom fitur tidak ditemukan: {missing_columns}"
        )

    X = customer[feature_columns].copy()

    if X.isnull().any().any():
        raise ValueError("Data pelanggan memiliki nilai kosong.")

    # Select model and threshold
    if model_choice == "logistic_regression":
        model = saved["logistic_regression"]
        threshold = saved["threshold_lr"]

    elif model_choice == "random_forest":
        model = saved["random_forest"]
        threshold = saved["threshold_rf"]

    elif model_choice == "ensemble":
        lr_model = saved["logistic_regression"]
        rf_model = saved["random_forest"]
        threshold = saved["threshold_ensemble"]

    else:
        raise ValueError(
            "Pilihan harus logistic_regression, random_forest, "
            "atau ensemble."
        )

    # Predict churn probability
    if model_choice == "ensemble":
        lr_probability = lr_model.predict_proba(X)[0, 1]
        rf_probability = rf_model.predict_proba(X)[0, 1]

        # Average both probabilities
        probability = (lr_probability + rf_probability) / 2

    else:
        probability = model.predict_proba(X)[0, 1]

    # Classify prediction using the selected threshold
    prediction = int(probability >= threshold)

    # Display results
    print("\n=== HASIL PREDIKSI CHURN ===")
    print(f"Customer ID        : {customer_id}")
    print(f"Model              : {model_choice}")
    print(f"Probabilitas churn : {probability:.2%}")
    print(f"Threshold          : {threshold:.2%}")

    if prediction == 1:
        print("Prediksi           : BERISIKO CHURN")
    else:
        print("Prediksi           : TIDAK DIKLASIFIKASIKAN CHURN")

    return {
        "customer_id": customer_id,
        "model": model_choice,
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
        print("\nPilih model:")
        print("1. Logistic Regression + SMOTE")
        print("2. Random Forest + SMOTE")
        print("3. Ensemble (rata-rata probabilitas kedua model)")

        choice = input("Masukkan pilihan (1/2/3): ").strip()

        model_mapping = {
            "1": "logistic_regression",
            "2": "random_forest",
            "3": "ensemble",
        }

        if choice not in model_mapping:
            print("Pilihan tidak valid.")

        else:
            predict_customer(
                customer_id,
                model_choice=model_mapping[choice],
            )