from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data" / "sound_features.csv"
MODEL_DIR = ROOT / "data" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "sound_mlp_v1.joblib"
SCALER_FILE = MODEL_DIR / "sound_scaler_v1.joblib"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("=" * 60)
print("GENERAL SOUND CLASSIFIER V1")
print("=" * 60)

feature_columns = [
    c for c in df.columns
    if c.startswith("feature_")
]

X = df[feature_columns].values.astype(np.float32)
y = df["class"].values

print(f"Dataset shape : {df.shape}")
print(f"Features      : {len(feature_columns)}")

print("\nClass distribution:")
print(df["class"].value_counts())


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

validation_sessions = [
    "doorbell_session_02",
    "knock_session_02",
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
print(pd.Series(y_train).value_counts())

print("\nValidation distribution:")
print(pd.Series(y_val).value_counts())


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


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")


print("\nClassification report:")

print(
    classification_report(
        y_val,
        y_pred,
        labels=[
            "background",
            "doorbell",
            "knock"
        ],
        zero_division=0
    )
)


print("\nConfusion matrix:")

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[
        "background",
        "doorbell",
        "knock"
    ]
)

print(cm)

print("\nRows = actual")
print("Columns = predicted")

print("\nLabels:")
print("[background, doorbell, knock]")


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


print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model  : {MODEL_FILE}")
print(f"Scaler : {SCALER_FILE}")

print(f"\nIterations : {model.n_iter_}")

if hasattr(model, "best_validation_score_"):
    print(
        f"Internal validation score : "
        f"{model.best_validation_score_:.4f}"
    )