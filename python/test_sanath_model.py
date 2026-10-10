from pathlib import Path

import joblib
import numpy as np
import librosa


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "data" / "models"

MODEL_FILE = MODEL_DIR / "sanath_mlp_v4.joblib"
SCALER_FILE = MODEL_DIR / "sanath_scaler_v4.joblib"

TEST_FILE = (
    ROOT
    / "data"
    / "dataset"
    / "sanath"
    / "sanath_test_01.wav"
)


# ============================================================
# AUDIO / FEATURE SETTINGS
# ============================================================

SR = 16000

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

N_FFT = 256
HOP_LENGTH = 128
N_MELS = 40

FMIN = 20
FMAX = 8000

DETECTION_THRESHOLD = 0.50


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Same per-clip standardization used during training
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std > 0:
        mel_db = (
            mel_db - mean
        ) / std

    else:
        mel_db = (
            mel_db - mean
        )

    return mel_db.flatten()


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("SANATH KEYWORD SPOTTER - V4 TEST")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = joblib.load(
    MODEL_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

print("Model loaded.")
print("Scaler loaded.")


# ============================================================
# LOAD TEST RECORDING
# ============================================================

print("\nLoading test recording...")

audio, sample_rate = librosa.load(
    TEST_FILE,
    sr=SR,
    mono=True
)

duration = len(audio) / SR

print(
    f"Sample rate : {SR} Hz"
)

print(
    f"Duration    : {duration:.2f} seconds"
)

print(
    f"Samples     : {len(audio)}"
)


# ============================================================
# CREATE WINDOWS
# ============================================================

window_samples = int(
    WINDOW_SECONDS * SR
)

hop_samples = int(
    HOP_SECONDS * SR
)


windows = []

start = 0

while start + window_samples <= len(audio):

    end = start + window_samples

    window = audio[
        start:end
    ]

    windows.append(
        (
            start,
            end,
            window
        )
    )

    start += hop_samples


print(
    f"Windows     : {len(windows)}"
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

feature_matrix = []

for _, _, window in windows:

    features = extract_features(
        window
    )

    feature_matrix.append(
        features
    )


X = np.array(
    feature_matrix
)

print(
    f"Feature matrix: {X.shape}"
)


# ============================================================
# SCALE FEATURES
# ============================================================

X_scaled = scaler.transform(
    X
)


# ============================================================
# PREDICTION
# ============================================================

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


predictions = (
    probabilities >= DETECTION_THRESHOLD
)


# ============================================================
# WINDOW-BY-WINDOW RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("WINDOW-BY-WINDOW RESULTS")
print("=" * 60)


detected_windows = []


for i, (
    (start_sample, end_sample, _),
    probability,
    prediction
) in enumerate(
    zip(
        windows,
        probabilities,
        predictions
    )
):

    start_time = (
        start_sample / SR
    )

    end_time = (
        end_sample / SR
    )

    if prediction:

        label = "SANATH"

        detected_windows.append(
            (
                start_time,
                end_time,
                probability
            )
        )

    else:

        label = "not Sanath"


    print(
        f"{i:03d} | "
        f"{start_time:7.2f}s - "
        f"{end_time:7.2f}s | "
        f"{probability:.4f} | "
        f"{label}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)


print(
    f"Total windows       : "
    f"{len(windows)}"
)

print(
    f"Detected as Sanath  : "
    f"{len(detected_windows)}"
)

print(
    f"Detection threshold : "
    f"{DETECTION_THRESHOLD:.2f}"
)


if detected_windows:

    print("\nDetected windows:")

    for (
        start_time,
        end_time,
        probability
    ) in detected_windows:

        print(
            f"  {start_time:.2f}s - "
            f"{end_time:.2f}s "
            f"(probability = "
            f"{probability:.4f})"
        )

else:

    print(
        "\nNo Sanath detections."
    )


# ============================================================
# TOP 10 PROBABILITIES
# ============================================================

print("\n")
print("=" * 60)
print("TOP 10 SANATH PROBABILITIES")
print("=" * 60)


ranked_indices = np.argsort(
    probabilities
)[::-1]


top_count = min(
    10,
    len(ranked_indices)
)


for rank in range(
    top_count
):

    index = ranked_indices[
        rank
    ]

    start_sample = windows[
        index
    ][0]

    end_sample = windows[
        index
    ][1]

    start_time = (
        start_sample / SR
    )

    end_time = (
        end_sample / SR
    )

    probability = probabilities[
        index
    ]

    print(
        f"{rank + 1:02d}. "
        f"{start_time:7.2f}s - "
        f"{end_time:7.2f}s | "
        f"{probability:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)