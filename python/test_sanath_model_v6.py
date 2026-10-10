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

THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

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

print(f"Audio length: {len(audio) / SR:.2f} seconds")


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

window_samples = int(WINDOW_SECONDS * SR)
hop_samples = int(HOP_SECONDS * SR)

features = []
times = []

for start in range(
    0,
    len(audio) - window_samples + 1,
    hop_samples
):

    end = start + window_samples

    window = audio[start:end]

    feat = extract_features(window)

    features.append(feat)

    times.append(
        start / SR
    )


X = np.array(features)

X_scaled = scaler.transform(X)


# ============================================================
# PREDICT
# ============================================================

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("SANATH V6 BENCHMARK")
print("========================================")

detections = 0

for time, probability in zip(
    times,
    probabilities
):

    if probability >= THRESHOLD:

        detections += 1

        print(
            f"{time:.1f}–{time + WINDOW_SECONDS:.1f} s"
            f"  {probability:.4f}"
        )


print("\n========================================")
print(f"Total windows : {len(probabilities)}")
print(f"Detections    : {detections}")
print(f"Threshold     : {THRESHOLD}")
print("========================================")


# ============================================================
# TOP 15 WINDOWS
# ============================================================

print("\nTop 15 windows:")

top_indices = np.argsort(
    probabilities
)[::-1][:15]

for i in top_indices:

    print(
        f"{times[i]:.1f}–"
        f"{times[i] + WINDOW_SECONDS:.1f} s"
        f"  {probabilities[i]:.4f}"
    )