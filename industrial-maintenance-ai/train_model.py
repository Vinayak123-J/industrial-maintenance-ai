from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Remove identifiers and columns that should not be used
# UDI is an identifier; Product ID is a unique product label.
# TWF, HDF, PWF, OSF and RNF describe failure modes and can
# reveal the target directly, so exclude them from the inputs.
drop_columns = [
    "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"
]

df = df.drop(columns=drop_columns, errors="ignore")

target = "Machine failure"

X = df.drop(columns=[target])
y = df[target]

# Convert categorical machine Type (L, M, H) into numeric columns
X = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print("Encoded feature columns:", X.columns.tolist())

# Split data, preserving the failure/no-failure ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

# Save trained model and feature names
model_path = MODEL_DIR / "failure_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(
        {"model": model, "features": list(X.columns)},
        f
    )

print("\nModel saved to:", model_path)

# Save confusion matrix image
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Failure", "Failure"]
)

display.plot(values_format="d")
plt.title("Machine Failure - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png")
plt.close()

print("Confusion matrix saved.")

# Save feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    REPORTS_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))

print("\nTraining and evaluation completed!")