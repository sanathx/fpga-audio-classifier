import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sanath_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_mlp_v6.joblib"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_scaler_v6.joblib"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("Loading features...")

df = pd.read_csv(FEATURE_FILE)

X = df.drop(columns=["label", "session", "filename"], errors="ignore")
y = df["label"]

print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")

print("\nOriginal class distribution:")
print(y.value_counts())


# ============================================================
# OVERSAMPLE POSITIVE CLASS
# ============================================================

df_train = df.copy()

positive = df_train[df_train["label"] == 1]
negative = df_train[df_train["label"] == 0]

print(f"\nNegative samples: {len(negative)}")
print(f"Positive samples: {len(positive)}")

positive_oversampled = resample(
    positive,
    replace=True,
    n_samples=len(negative),
    random_state=42
)

train_balanced = pd.concat(
    [negative, positive_oversampled],
    ignore_index=True
)

train_balanced = train_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_balanced.drop(
    columns=["label", "session", "filename"],
    errors="ignore"
)

y_train = train_balanced["label"]

print("\nBalanced training distribution:")
print(y_train.value_counts())


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Sanath V6...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.0005,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_FILE),
    exist_ok=True
)

joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("SANATH V6 TRAINING COMPLETE")
print("========================================")

print(f"Iterations : {model.n_iter_}")
print(f"Final loss : {model.loss_:.6f}")

print("\nModel:")
print(MODEL_FILE)

print("\nScaler:")
print(SCALER_FILE)

print("\nIMPORTANT:")
print("Session 03 is now included in training.")
print("The untouched sanath_test_01.wav will be used as the benchmark.")