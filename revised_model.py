from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# %%
# import dataset
DATA_FILE = Path("customers_ml.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {DATA_FILE.resolve()}"
    )

df = pd.read_csv(DATA_FILE)

# print("ukuran dataset:", df.shape)
# df.head()

# %%
# Split fitur prediksi dan target

excluded = ["churn", "customer_id", "tenure_group"]
feature_columns = [
    col for col in df.columns
    if col not in excluded
]

X = df[feature_columns].copy()

# y = target yang ingin diprediksi
y = df["churn"]

# print("jumlah fitur:", len(feature_columns))
# print("daftar fitur:")
# print(feature_columns)

# print("\nDistribusi target churn:")
# print(y.value_counts())

# %%
# Split data menjadi training, validation, dan testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# print(f"Training   : {len(X_train)} data ({len(X_train)/len(df):.0%})")
# print(f"Validation : {len(X_val)} data ({len(X_val)/len(df):.0%})")
# print(f"Testing    : {len(X_test)} data ({len(X_test)/len(df):.0%})")

# %%
# 5. Logistic regression + SMOTE

base_model = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(
        random_state=42,
        sampling_strategy="auto",
        k_neighbors=5
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    )),
])

model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=5
)

model.fit(X_train, y_train)

# %%
# Cek imbalance data
# smote_check = SMOTE(random_state=42)

# X_train_scaled = StandardScaler().fit_transform(X_train)
# X_resampled, y_resampled = smote_check.fit_resample(
#     X_train_scaled, y_train
# )

# print("Sebelum SMOTE:")
# print(y_train.value_counts())

# print("\nSesudah SMOTE:")
# print(y_resampled.value_counts())

# %%
# 7. Memilih threshold menggunakan validation set

FP_COST = 1
FN_COST = 3

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

# print("Threshold terbaik:", best_threshold)
# print("Total biaya kesalahan pada validation set:", best_cost)
# print("Asumsi biaya false positive:", FP_COST)
# print("Asumsi biaya false negative:", FN_COST)

# %%
# 8 evaluasi model logistic
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

# print("Accuracy :", accuracy_score(y_test, test_pred))
# print("Precision:", precision_score(y_test, test_pred, zero_division=0))
# print("Recall   :", recall_score(y_test, test_pred, zero_division=0))
# print("F1-score :", f1_score(y_test, test_pred, zero_division=0))
# print("ROC-AUC  :", roc_auc_score(y_test, test_prob))
# print("PR-AUC   :", average_precision_score(y_test, test_prob))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, test_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, test_pred, zero_division=0))

# %%
# Random forest (added model) + SMOTE
rf_base_model = ImbPipeline([
    ("smote", SMOTE(
        random_state=42,
        sampling_strategy="auto",
        k_neighbors=5
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model = CalibratedClassifierCV(
    estimator=rf_base_model,
    method="sigmoid",
    cv=5
)

rf_model.fit(X_train, y_train)

# %%
rf_val_prob = rf_model.predict_proba(X_val)[:, 1]

best_threshold_rf = None
best_cost_rf = float("inf")

# FP_COST = 1
# FN_COST = 3

for threshold in np.arange(0.05, 0.96, 0.01):
    rf_val_pred = (rf_val_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, rf_val_pred, labels=[0, 1]
    ).ravel()

    total_cost = FP_COST * fp + FN_COST * fn

    if total_cost < best_cost_rf:
        best_cost_rf = total_cost
        best_threshold_rf = threshold

# print("Threshold terbaik RF:", best_threshold_rf)
# print("Biaya validation RF:", best_cost_rf)

# %%
rf_test_prob = rf_model.predict_proba(X_test)[:, 1]

rf_test_pred = (rf_test_prob >= best_threshold_rf).astype(int)

# Evaluasi RF
# print("Threshold:", best_threshold_rf)
# print("Accuracy:", accuracy_score(y_test, rf_test_pred))
# print("Precision:", precision_score(y_test, rf_test_pred, zero_division=0))
# print("Recall:", recall_score(y_test, rf_test_pred, zero_division=0))
# print("F1-score:", f1_score(y_test, rf_test_pred, zero_division=0))
# print("ROC-AUC:", roc_auc_score(y_test, rf_test_prob))
# print("PR-AUC:", average_precision_score(y_test, rf_test_prob))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, rf_test_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, rf_test_pred, zero_division=0))

# %%
#  Ensemble (gabungan)

lr_val_prob = model.predict_proba(X_val)[:, 1]
rf_val_prob = rf_model.predict_proba(X_val)[:, 1]

ensemble_val_prob = (lr_val_prob + rf_val_prob) / 2

# print("Contoh probabilitas ensemble:")
# print(ensemble_val_prob[:10])

# %%
best_threshold_ensemble = None
best_cost_ensemble = float("inf")

for threshold in np.arange(0.05, 0.96, 0.01):
    ensemble_val_pred = (
        ensemble_val_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        ensemble_val_pred,
        labels=[0, 1]
    ).ravel()

    total_cost = FP_COST * fp + FN_COST * fn

    if total_cost < best_cost_ensemble:
        best_cost_ensemble = total_cost
        best_threshold_ensemble = threshold

# print("Threshold terbaik Ensemble:", best_threshold_ensemble)
# print("Biaya validation Ensemble:", best_cost_ensemble)

# %%
lr_test_prob = model.predict_proba(X_test)[:, 1]
rf_test_prob = rf_model.predict_proba(X_test)[:, 1]

ensemble_test_prob = (lr_test_prob + rf_test_prob) / 2

ensemble_test_pred = (ensemble_test_prob >= best_threshold_ensemble).astype(int)

# print("Threshold:", best_threshold_ensemble)
# print("Accuracy:", accuracy_score(y_test, ensemble_test_pred))
# print("Precision:", precision_score(y_test, ensemble_test_pred, zero_division=0))
# print("Recall:", recall_score(y_test, ensemble_test_pred, zero_division=0))
# print("F1-score:", f1_score(y_test, ensemble_test_pred, zero_division=0))
# print("ROC-AUC:", roc_auc_score(y_test, ensemble_test_prob))
# print("PR-AUC:", average_precision_score(y_test, ensemble_test_prob))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, ensemble_test_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, ensemble_test_pred, zero_division=0))

# %%
model_package = {
    "logistic_regression": model,
    "random_forest": rf_model,

    # Thresholdnya
    "threshold_lr": best_threshold,
    "threshold_rf": best_threshold_rf,
    "threshold_ensemble": best_threshold_ensemble,

    "feature_columns": list(X_train.columns),

    # Aturan biaya kesalahan
    "fp_cost": FP_COST,
    "fn_cost": FN_COST
}

joblib.dump(model_package, "churn_model.joblib")


# - Untuk visualisasi
evaluation_data = {
    "y_test": y_test,

    "test_pred": test_pred,
    "test_prob": test_prob,

    "rf_test_pred": rf_test_pred,
    "rf_test_prob": rf_test_prob,

    "ensemble_test_pred": ensemble_test_pred,
    "ensemble_test_prob": ensemble_test_prob,
}

joblib.dump(evaluation_data, "evaluation_results.joblib")