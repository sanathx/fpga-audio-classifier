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


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "data" / "sanath_features.csv"
)

MODEL_DIR = (
    ROOT / "data" / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = (
    MODEL_DIR / "sanath_mlp_v3.joblib"
)

SCALER_FILE = (
    MODEL_DIR / "sanath_scaler_v3.joblib"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("LOADING SANATH DATASET")
print("=" * 60)

df = pd.read_csv(FEATURE_FILE)

feature_columns = [
    column
    for column in df.columns
    if column.startswith("feature_")
]

X = df[feature_columns].values
y = df["label"].values

print(f"Total samples : {len(df)}")
print(f"Features      : {len(feature_columns)}")


# ============================================================
# SESSION-AWARE SPLIT
# ============================================================

# Training:
#   Sanath Session 01
#   Sanath Session 02
#   Background
#   Speech
#
# Held-out validation:
#   Sanath Session 03
#   Speech Session 03

train_sessions = {
    "sanath_session_01",
    "sanath_session_02",
    "background",
    "speech",
}

val_sessions = {
    "sanath_session_03",
    "speech_session_03",
}


train_mask = df["session"].isin(
    train_sessions
)

val_mask = df["session"].isin(
    val_sessions
)


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]


print("\nTraining sessions:")
print(
    df.loc[train_mask, "session"]
    .value_counts()
)

print("\nValidation sessions:")
print(
    df.loc[val_mask, "session"]
    .value_counts()
)

print("\nTraining class distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nValidation class distribution:")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
)


# ============================================================
# OVERSAMPLE POSITIVE CLASS
# ============================================================

print("\n" + "=" * 60)
print("BALANCING TRAINING DATA")
print("=" * 60)

positive_indices = np.where(
    y_train == 1
)[0]

negative_indices = np.where(
    y_train == 0
)[0]

rng = np.random.default_rng(42)

if len(positive_indices) < len(negative_indices):

    extra_positive = rng.choice(
        positive_indices,
        size=len(negative_indices) - len(positive_indices),
        replace=True,
    )

    balanced_indices = np.concatenate(
        [
            negative_indices,
            positive_indices,
            extra_positive,
        ]
    )

else:

    balanced_indices = np.arange(
        len(y_train)
    )


rng.shuffle(
    balanced_indices
)

X_train_balanced = X_train[
    balanced_indices
]

y_train_balanced = y_train[
    balanced_indices
]


print(
    f"Original training samples : "
    f"{len(y_train)}"
)

print(
    f"Balanced training samples : "
    f"{len(y_train_balanced)}"
)

print("\nBalanced class distribution:")

print(
    pd.Series(y_train_balanced)
    .value_counts()
    .sort_index()
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "=" * 60)
print("SCALING FEATURES")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_val_scaled = scaler.transform(
    X_val
)


# ============================================================
# TRAIN MLP
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SANATH MLP V3")
print("=" * 60)

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
    verbose=True,
)


model.fit(
    X_train_scaled,
    y_train_balanced
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("SESSION 03 VALIDATION")
print("=" * 60)

y_pred = model.predict(
    X_val_scaled
)

y_probability = model.predict_proba(
    X_val_scaled
)[:, 1]


accuracy = accuracy_score(
    y_val,
    y_pred
)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Not Sanath",
            "Sanath",
        ],
        zero_division=0,
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_val,
        y_pred
    )
)


# ============================================================
# PER-SESSION VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("PER-SESSION RESULTS")
print("=" * 60)

val_df = df.loc[
    val_mask
].copy()

val_df["prediction"] = y_pred
val_df["sanath_probability"] = y_probability

for session_name in sorted(
    val_df["session"].unique()
):

    session_data = val_df[
        val_df["session"] == session_name
    ]

    session_accuracy = accuracy_score(
        session_data["label"],
        session_data["prediction"]
    )

    print(
        f"\n{session_name}"
    )

    print(
        f"Samples  : {len(session_data)}"
    )

    print(
        f"Accuracy : "
        f"{session_accuracy:.4f}"
    )

    print(
        pd.crosstab(
            session_data["label"],
            session_data["prediction"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )


# ============================================================
# SAVE MODEL
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

print(
    f"Model : {MODEL_FILE}"
)

print(
    f"Scaler: {SCALER_FILE}"
)

print(
    f"\nTraining iterations: "
    f"{model.n_iter_}"
)

print(
    f"Final training loss: "
    f"{model.loss_:.6f}"
)