import os
import numpy as np
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

# Use Doorbell Session 02
AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "phone_ring",
    "labeled",
    "session_02",
    "phone_ring_0004.wav"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sound_mlp_v3.joblib"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sound_scaler_v3.joblib"
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
# LOAD MODEL
# ============================================================

print("Loading Sound V3...")

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

print(
    f"Audio length: "
    f"{len(audio) / SR:.2f} seconds"
)


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
# WINDOWS
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

    times.append(
        start / SR
    )


X = np.array(features)

X_scaled = scaler.transform(X)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_scaled)

probabilities = model.predict_proba(
    X_scaled
)

classes = model.classes_


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("DOORBELL CLASSIFICATION")
print("========================================")

for i, time in enumerate(times):

    predicted = predictions[i]

    confidence = np.max(
        probabilities[i]
    )

    print(
        f"{time:.1f}–"
        f"{time + WINDOW_SECONDS:.1f} s"
        f"  {predicted:<12}"
        f"  confidence={confidence:.4f}"
    )


# ============================================================
# OVERALL
# ============================================================

counts = {}

for prediction in predictions:

    counts[prediction] = (
        counts.get(prediction, 0) + 1
    )


print("\n========================================")
print("SUMMARY")
print("========================================")

for label, count in counts.items():

    print(
        f"{label:<12}: {count}"
    )