import os
import numpy as np
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "sanath_test_01.wav"
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
# SETTINGS
# ============================================================

SR = 16000
N_FFT = 256
HOP_LENGTH = 128

N_MELS = 40
FMIN = 20
FMAX = 8000

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5


# ============================================================
# APPROXIMATE TRUE SANATH OCCURRENCES
# ============================================================
#
# These are the manually identified approximate locations
# in sanath_test_01.wav.
#
# We are NOT training on these.
# They are ONLY being used to evaluate V6.
#

TRUE_OCCURRENCES = [
    1.0,
    3.0,
    5.0,
    9.0,
    11.0,
    13.0,
    15.0,
    17.0,
    18.0,
    19.0
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V6...")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)


# ============================================================
# LOAD AUDIO
# ============================================================

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

duration = len(audio) / SR

print(f"Audio duration: {duration:.2f} seconds")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(y):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    return mel_db.flatten()


# ============================================================
# CREATE WINDOWS
# ============================================================

window_samples = int(
    WINDOW_SECONDS * SR
)

hop_samples = int(
    HOP_SECONDS * SR
)

features = []
times = []

for start in range(
    0,
    len(audio) - window_samples + 1,
    hop_samples
):

    end = start + window_samples

    window = audio[start:end]

    features.append(
        extract_features(window)
    )

    times.append(start / SR)


X = np.array(features)

X_scaled = scaler.transform(X)

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# DISPLAY ALL WINDOWS
# ============================================================

print("\n========================================")
print("ALL V6 WINDOWS")
print("========================================")

for time, probability in zip(
    times,
    probabilities
):

    print(
        f"{time:5.1f}–"
        f"{time + WINDOW_SECONDS:5.1f} s"
        f"    {probability:.4f}"
    )


# ============================================================
# EVALUATE EACH TRUE OCCURRENCE
# ============================================================

print("\n========================================")
print("TRUE SANATH OCCURRENCES")
print("========================================")

print(
    "\nFor each occurrence, we show the strongest "
    "nearby V6 probability.\n"
)


results = []

for occurrence in TRUE_OCCURRENCES:

    # Look at windows whose centers are within
    # ±1 second of the manually identified occurrence.
    nearby = []

    for time, probability in zip(
        times,
        probabilities
    ):

        window_center = (
            time +
            WINDOW_SECONDS / 2
        )

        if abs(
            window_center - occurrence
        ) <= 1.0:

            nearby.append(
                (
                    time,
                    probability
                )
            )

    if not nearby:
        continue

    best_time, best_probability = max(
        nearby,
        key=lambda x: x[1]
    )

    results.append(
        (
            occurrence,
            best_time,
            best_probability
        )
    )

    if best_probability >= 0.80:
        status = "STRONG"
    elif best_probability >= 0.50:
        status = "DETECTED"
    else:
        status = "MISSED"

    print(
        f"True ~{occurrence:4.1f} s"
        f"   best window "
        f"{best_time:4.1f}–"
        f"{best_time + WINDOW_SECONDS:4.1f} s"
        f"   probability = "
        f"{best_probability:.4f}"
        f"   [{status}]"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("SUMMARY")
print("========================================")

strong = sum(
    p >= 0.80
    for _, _, p in results
)

detected = sum(
    p >= 0.50
    for _, _, p in results
)

missed = sum(
    p < 0.50
    for _, _, p in results
)

print(f"True occurrences : {len(results)}")
print(f"Strong >= 0.80   : {strong}")
print(f"Detected >= 0.50 : {detected}")
print(f"Missed < 0.50    : {missed}")

print("\nDetection rate:")
print(
    f"{detected / len(results) * 100:.1f}%"
)

print("\nStrong detection rate:")
print(
    f"{strong / len(results) * 100:.1f}%"
)