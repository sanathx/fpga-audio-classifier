from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data" / "sanath_features.csv"
MODEL_DIR = ROOT / "data" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "sanath_mlp_v2.joblib"
SCALER_FILE = MODEL_DIR / "sanath_scaler_v2.joblib"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("=" * 60)
print("SANATH KEYWORD MODEL V2")
print("=" * 60)

print(f"Dataset shape : {df.shape}")

# Everything except metadata is a feature
feature_columns = [
    c for c in df.columns
    if c.startswith("feature_")
]

X = df[feature_columns].values.astype(np.float32)
y = df["label"].values.astype(np.int32)

print(f"Features      : {X.shape[1]}")


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

# Session 03 is completely held out.
validation_sessions = [
    "sanath_session_03",
    "speech_session_03",
]

val_mask = df["session"].isin(validation_sessions)

train_mask = ~val_mask

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

print("\n" + "=" * 60)
print("SESSION-AWARE SPLIT")
print("=" * 60)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")

print("\nTraining distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nValidation distribution:")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
)

print("\nValidation sessions:")
print(
    df.loc[val_mask, "session"]
    .value_counts()
)


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# ============================================================
# MODEL
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",

    alpha=0.001,

    batch_size=16,

    learning_rate_init=0.0005,

    max_iter=500,

    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=25,

    random_state=42,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

y_pred = model.predict(X_val_scaled)

accuracy = accuracy_score(
    y_val,
    y_pred
)

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("V2 VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Not Sanath",
            "Sanath"
        ],
        zero_division=0
    )
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_val,
        y_pred
    )
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)


# ============================================================
# FINAL INFO
# ============================================================

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model  : {MODEL_FILE}")
print(f"Scaler : {SCALER_FILE}")

print(f"\nIterations completed : {model.n_iter_}")

if hasattr(model, "best_validation_score_"):
    print(
        f"Internal validation score: "
        f"{model.best_validation_score_:.4f}"
    )
    