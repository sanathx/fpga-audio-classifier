import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"C:\Users\sandb\Desktop\fpga-audio-classifier\data\sanath_features.csv"

MODEL_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier\data\models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "sanath_mlp.joblib"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "sanath_scaler.joblib"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("SANATH KEYWORD SPOTTER - MLP TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape : {df.shape}")

# Label column:
# 0 = Not Sanath
# 1 = Sanath

y = df["label"].values

# Keep only numeric feature columns.
# The CSV also contains a filename column.
feature_columns = df.select_dtypes(include=[np.number]).columns

feature_columns = feature_columns.drop("label")

X = df[feature_columns].values.astype(np.float32)

print(f"Samples       : {len(X)}")
print(f"Features      : {X.shape[1]}")

print("\nClass distribution:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nSplit:")
print(f"Training   : {len(X_train)}")
print(f"Validation : {len(X_val)}")


# ============================================================
# FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    scaler,
    SCALER_FILE
)

print(f"Scaler saved:")
print(SCALER_FILE)


# ============================================================
# BUILD MLP
# ============================================================

print("\nBuilding neural network...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",

    solver="adam",

    alpha=0.0005,

    batch_size=16,

    learning_rate_init=0.001,

    max_iter=300,

    early_stopping=True,

    validation_fraction=0.20,

    n_iter_no_change=15,

    random_state=RANDOM_STATE,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("=" * 60)

model.fit(
    X_train,
    y_train
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved:")
print(MODEL_FILE)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_val
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    predictions
)

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)


print("\n")
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "Not Sanath",
            "Sanath"
        ],
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_val,
        predictions
    )
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING INFORMATION")
print("=" * 60)

print(
    f"Iterations completed : {model.n_iter_}"
)

print(
    f"Best validation score: "
    f"{model.best_validation_score_:.4f}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("SANATH MODEL TRAINING COMPLETE")
print("=" * 60)

print(f"Model : {MODEL_FILE}")
print(f"Scaler: {SCALER_FILE}")

print("\nIMPORTANT:")
print(
    "This validation result is only a first model result."
)

print(
    "We will test the model on a completely new "
    "Sanath recording before judging real-world performance."
)