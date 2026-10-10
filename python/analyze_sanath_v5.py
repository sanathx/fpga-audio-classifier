import os
import numpy as np
import pandas as pd
import joblib


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
    "sanath_mlp_v5.joblib"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_scaler_v5.joblib"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(FEATURE_FILE)

feature_columns = [
    c for c in df.columns
    if c.startswith("feature_")
]

X = df[feature_columns].values
y = df["label"].values
sessions = df["session"].values
filenames = df["filename"].values

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

X_scaled = scaler.transform(X)

probabilities = model.predict_proba(X_scaled)

# Find which probability column corresponds to class 1
class_1_index = list(model.classes_).index(1)

sanath_prob = probabilities[:, class_1_index]


# ============================================================
# VALIDATION DATA
# ============================================================

validation_sessions = {
    "sanath_session_03",
    "sanath_session_03_partial",
    "speech_session_03"
}

validation_mask = np.isin(
    sessions,
    list(validation_sessions)
)

val_indices = np.where(validation_mask)[0]


# ============================================================
# PRINT ALL VALIDATION SAMPLES
# ============================================================

print("=" * 70)
print("SANATH V5 PROBABILITY ANALYSIS")
print("=" * 70)

print()
print("Validation samples:")
print()

print(
    f"{'Session':30s} "
    f"{'File':20s} "
    f"{'Actual':>8s} "
    f"{'Sanath P':>10s}"
)

print("-" * 75)

for i in val_indices:

    print(
        f"{sessions[i]:30s} "
        f"{filenames[i]:20s} "
        f"{y[i]:8d} "
        f"{sanath_prob[i]:10.4f}"
    )


# ============================================================
# SANATH POSITIVE VALIDATION SAMPLES
# ============================================================

positive_mask = (
    validation_mask &
    (y == 1)
)

positive_indices = np.where(
    positive_mask
)[0]

print()
print("=" * 70)
print("HELD-OUT SANATH SAMPLES")
print("=" * 70)

if len(positive_indices) > 0:

    positive_sorted = sorted(
        positive_indices,
        key=lambda i: sanath_prob[i],
        reverse=True
    )

    for i in positive_sorted:

        print(
            f"{sessions[i]:30s} "
            f"{filenames[i]:20s} "
            f"Sanath probability = {sanath_prob[i]:.4f}"
        )

    print()
    print(
        f"Average probability : "
        f"{np.mean([sanath_prob[i] for i in positive_indices]):.4f}"
    )

    print(
        f"Maximum probability : "
        f"{np.max([sanath_prob[i] for i in positive_indices]):.4f}"
    )

    print(
        f"Minimum probability : "
        f"{np.min([sanath_prob[i] for i in positive_indices]):.4f}"
    )


# ============================================================
# NEGATIVE VALIDATION SAMPLES
# ============================================================

negative_mask = (
    validation_mask &
    (y == 0)
)

negative_indices = np.where(
    negative_mask
)[0]

print()
print("=" * 70)
print("TOP NEGATIVE SAMPLES")
print("=" * 70)

negative_sorted = sorted(
    negative_indices,
    key=lambda i: sanath_prob[i],
    reverse=True
)

for i in negative_sorted[:15]:

    print(
        f"{sessions[i]:30s} "
        f"{filenames[i]:20s} "
        f"Sanath probability = {sanath_prob[i]:.4f}"
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print()
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

val_y = y[validation_mask]
val_prob = sanath_prob[validation_mask]

for threshold in [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]:

    predictions = (
        val_prob >= threshold
    ).astype(int)

    tp = np.sum(
        (predictions == 1) &
        (val_y == 1)
    )

    tn = np.sum(
        (predictions == 0) &
        (val_y == 0)
    )

    fp = np.sum(
        (predictions == 1) &
        (val_y == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (val_y == 1)
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"TP={tp:2d} "
        f"FP={fp:2d} "
        f"FN={fn:2d} "
        f"TN={tn:2d} | "
        f"Precision={precision:.3f} "
        f"Recall={recall:.3f}"
    )


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)