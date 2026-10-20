#  LOGISTIC REGRES

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

DATA_FILE = Path("customers_ml.csv")
MODEL_FILE = Path("churn_model.joblib")
CALIBRATION_PLOT = Path("calibration_curve.png")

# Asumsi biaya (contohh)
FP_COST = 1
FN_COST = 3

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("customers_ml.csv tidak ditemukan.")

    df = pd.read_csv(DATA_FILE)

    if "churn" not in df.columns:
        raise ValueError("Kolom churn tidak ditemukan.")

# Target tdk boleh jdi fitur prediksi (kemungkinan bisa data leakage?).
# customer_id adlh identifier, bkn fitur prediksi.
# tenure_group tdk dipakai krna tenure_months sdh ada.
    excluded = ["churn", "customer_id", "tenure_group"]
    feature_columns = [c for c in df.columns if c not in excluded]

    X = df[feature_columns].copy()
    y = df["churn"]

    if X.isnull().any().any() or y.isnull().any():
        raise ValueError("Masih ada nilai kosong dalam dataset.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Fitur belum numerik: {non_numeric}")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target churn harus memiliki kelas 0 dan 1.")

    # 60% training, 20% validation, 20% testing.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp
    )

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42
        )),
    ])

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )
    model.fit(X_train, y_train)

    # Pilih threshold pda validation set, bkn test set.
    val_prob = model.predict_proba(X_val)[:, 1]
    thresholds = [i / 100 for i in range(5, 96)]

    best_threshold = 0.50
    best_cost = float("inf")

    for threshold in thresholds:
        val_pred = (val_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, val_pred, labels=[0, 1]
        ).ravel()

        total_cost = FP_COST * fp + FN_COST * fn

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    print("=== DATA SPLIT ===")
    print(f"Training   : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Testing    : {len(X_test)}")

    print("\n=== THRESHOLD & COST ===")
    print(f"Assumed FP cost : {FP_COST}")
    print(f"Assumed FN cost : {FN_COST}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Validation cost   : {best_cost}")

    print("\n=== TEST PERFORMANCE ===")
    print(f"Accuracy : {accuracy_score(y_test, test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{average_precision_score(y_test, test_prob):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test, test_pred,
        labels=[0, 1],
        target_names=["Tidak Churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Estimated test cost under assumed costs: {FP_COST * fp + FN_COST * fn}")

# Evaluasi kalibrasi
    brier = brier_score_loss(y_test, test_prob)
    print(f"\nBrier score: {brier:.4f}")

    actual_rate, predicted_rate = calibration_curve(
        y_test, test_prob, n_bins=10, strategy="uniform"
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        predicted_rate, actual_rate, marker="o",
        label="Model calibration"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Rata-rata probabilitas prediksi")
    plt.ylabel("Proporsi churn aktual")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT)
    plt.close()

    joblib.dump({
        "model": model,
        "threshold": best_threshold,
        "feature_columns": feature_columns,
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,
    }, MODEL_FILE)

    print(f"\nModel dan threshold disimpan: {MODEL_FILE.resolve()}")
    print(f"Calibration plot disimpan: {CALIBRATION_PLOT.resolve()}")

if __name__ == "__main__":
    main()
