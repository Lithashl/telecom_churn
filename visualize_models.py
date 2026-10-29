# Visualisasi
import joblib
import pandas as pd
import matplotlib.pyplot as plt
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
    ConfusionMatrixDisplay,)

#  CALL
evaluation_data = joblib.load("evaluation_results.joblib")

y_test = evaluation_data["y_test"]

test_pred = evaluation_data["test_pred"]
test_prob = evaluation_data["test_prob"]

rf_test_pred = evaluation_data["rf_test_pred"]
rf_test_prob = evaluation_data["rf_test_prob"]

ensemble_test_pred = evaluation_data["ensemble_test_pred"]
ensemble_test_prob = evaluation_data["ensemble_test_prob"]

#  Plot
results = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1-score": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "PR-AUC": average_precision_score(y_test, test_prob),
    },
    "Random Forest": {
        "Accuracy": accuracy_score(y_test, rf_test_pred),
        "Precision": precision_score(y_test, rf_test_pred, zero_division=0),
        "Recall": recall_score(y_test, rf_test_pred, zero_division=0),
        "F1-score": f1_score(y_test, rf_test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, rf_test_prob),
        "PR-AUC": average_precision_score(y_test, rf_test_prob),
    },
    "Ensemble": {
        "Accuracy": accuracy_score(y_test, ensemble_test_pred),
        "Precision": precision_score(y_test, ensemble_test_pred, zero_division=0),
        "Recall": recall_score(y_test, ensemble_test_pred, zero_division=0),
        "F1-score": f1_score(y_test, ensemble_test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, ensemble_test_prob),
        "PR-AUC": average_precision_score(y_test, ensemble_test_prob),
    }
}

metrics_df = pd.DataFrame(results).T

ax = metrics_df.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    width=0.8
)

ax.set_title("Comparison of Churn Prediction Models")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("model_metrics_comparison.png", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

predictions = [
    ("Logistic Regression", test_pred),
    ("Random Forest", rf_test_pred),
    ("Ensemble", ensemble_test_pred)
]

for ax, (name, pred) in zip(axes, predictions):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        labels=[0, 1],
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()

probabilities = [
    ("Logistic Regression", test_prob),
    ("Random Forest", rf_test_prob),
    ("Ensemble", ensemble_test_prob)
]

plt.figure(figsize=(8, 6))

for name, prob in probabilities:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))

for name, prob in probabilities:
    precision_values, recall_values, _ = (
        precision_recall_curve(y_test, prob)
    )
    ap = average_precision_score(y_test, prob)

    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP = {ap:.3f})"
    )

baseline = y_test.mean()
plt.axhline(
    baseline,
    linestyle="--",
    label=f"Baseline = {baseline:.3f}"
)

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("precision_recall_comparison.png", dpi=300)
plt.show()

# metrics_df.to_csv("model_metrics_comparison.csv")

# print("\nHasil perbandingan:")
# print(metrics_df.round(4))
# print("\nSemua grafik dan tabel berhasil dibuat.")