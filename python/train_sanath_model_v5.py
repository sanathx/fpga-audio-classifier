import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "sanath_mlp_v5.joblib"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "sanath_scaler_v5.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

feature_columns = [
    c for c in df.columns
    if c.startswith("feature_")
]

X = df[feature_columns].values
y = df["label"].values
sessions = df["session"].values


# ============================================================
# SESSION SPLIT
# ============================================================

VALIDATION_SESSIONS = {
    "sanath_session_03",
    "sanath_session_03_partial",
    "speech_session_03"
}

TRAIN_MASK = ~np.isin(
    sessions,
    list(VALIDATION_SESSIONS)
)

VAL_MASK = np.isin(
    sessions,
    list(VALIDATION_SESSIONS)
)

X_train = X[TRAIN_MASK]
y_train = y[TRAIN_MASK]

X_val = X[VAL_MASK]
y_val = y[VAL_MASK]

sessions_val = sessions[VAL_MASK]


# ============================================================
# PRINT DATASET
# ============================================================

print("=" * 60)
print("SANATH V5 - LIGHTWEIGHT MODEL")
print("=" * 60)

print()
print(f"Total samples      : {len(X)}")
print(f"Features            : {len(feature_columns)}")
print(f"Training samples    : {len(X_train)}")
print(f"Validation samples  : {len(X_val)}")

print()
print("Training distribution:")
print(pd.Series(y_train).value_counts())

print()
print("Validation distribution:")
print(pd.Series(y_val).value_counts())


# ============================================================
# OVERSAMPLE POSITIVE CLASS
# ============================================================

X_train_df = pd.DataFrame(X_train)
X_train_df["label"] = y_train

negative_df = X_train_df[
    X_train_df["label"] == 0
]

positive_df = X_train_df[
    X_train_df["label"] == 1
]

target_count = max(
    len(negative_df),
    len(positive_df)
)

positive_oversampled = resample(
    positive_df,
    replace=True,
    n_samples=target_count,
    random_state=42
)

negative_oversampled = resample(
    negative_df,
    replace=True,
    n_samples=target_count,
    random_state=42
)

train_balanced = pd.concat(
    [
        negative_oversampled,
        positive_oversampled
    ],
    ignore_index=True
)

train_balanced = train_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train = train_balanced.drop(
    columns=["label"]
).values

y_train = train_balanced["label"].values


print()
print("After oversampling:")
print(pd.Series(y_train).value_counts())


# ============================================================
# STANDARDIZATION
# ============================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# ============================================================
# LIGHTWEIGHT MLP
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING V5")
print("=" * 60)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# TRAINING INFO
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Iterations : {model.n_iter_}")
print(f"Final loss : {model.loss_:.6f}")

# ============================================================
# SAVE MODEL IMMEDIATELY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print()
print("V5 model saved successfully:")
print(MODEL_FILE)
print(SCALER_FILE)

# ============================================================
# VALIDATION
# ============================================================

y_pred = model.predict(
    X_val_scaled
)

accuracy = accuracy_score(
    y_val,
    y_pred
)

print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print()
print(f"Accuracy : {accuracy:.4f}")

print()

print(
    classification_report(
    y_val[mask],
    y_pred[mask],
    labels=[0, 1],
    target_names=[
        "Not Sanath",
        "Sanath"
    ],
    zero_division=0
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
# PER-SESSION RESULTS
# ============================================================

print()
print("=" * 60)
print("PER-SESSION RESULTS")
print("=" * 60)

for session in sorted(
    set(sessions_val)
):

    mask = sessions_val == session

    session_accuracy = accuracy_score(
        y_val[mask],
        y_pred[mask]
    )

    print()
    print(session)
    print(f"Samples  : {np.sum(mask)}")
    print(f"Accuracy : {session_accuracy:.4f}")

    print(
        classification_report(
            y_val[mask],
            y_pred[mask],
            target_names=[
                "Not Sanath",
                "Sanath"
            ],
            zero_division=0
        )
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)


print()
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(MODEL_FILE)
print(SCALER_FILE)

print("=" * 60)