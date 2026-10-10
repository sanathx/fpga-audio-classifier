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
    "sanath_mlp_v5.joblib"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_scaler_v5.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

SR = 16000

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

N_FFT = 256
HOP_LENGTH = 128

N_MELS = 40
FMIN = 20
FMAX = 8000

THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("SANATH V5 BENCHMARK")
print("=" * 60)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

print()
print("V5 model loaded.")


# ============================================================
# LOAD AUDIO
# ============================================================

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

print()
print(f"Audio duration : {len(audio) / SR:.2f} sec")
print(f"Sample rate    : {SR} Hz")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_clip):

    mel = librosa.feature.melspectrogram(
        y=audio_clip,
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

    # Per-clip standardization
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std > 0:
        mel_db = (mel_db - mean) / std
    else:
        mel_db = mel_db - mean

    return mel_db.flatten()


# ============================================================
# SLIDING WINDOW
# ============================================================

window_samples = int(
    WINDOW_SECONDS * SR
)

hop_samples = int(
    HOP_SECONDS * SR
)


# ============================================================
# RUN MODEL
# ============================================================

results = []

start = 0

while start + window_samples <= len(audio):

    clip = audio[
        start:start + window_samples
    ]

    features = extract_features(clip)

    X = features.reshape(1, -1)

    X_scaled = scaler.transform(X)

    probability = model.predict_proba(
        X_scaled
    )[0]

    class_1_index = list(
        model.classes_
    ).index(1)

    sanath_probability = probability[
        class_1_index
    ]

    predicted = (
        sanath_probability >= THRESHOLD
    )

    start_time = start / SR
    end_time = (
        start + window_samples
    ) / SR

    results.append(
        (
            start_time,
            end_time,
            sanath_probability,
            predicted
        )
    )

    print(
        f"{start_time:5.1f} - "
        f"{end_time:5.1f} sec | "
        f"Sanath = "
        f"{sanath_probability:.4f} | "
        f"{'SANATH' if predicted else 'not Sanath'}"
    )

    start += hop_samples


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

detections = [
    r for r in results
    if r[3]
]

print()
print(f"Total windows       : {len(results)}")
print(f"Detected as Sanath  : {len(detections)}")
print(f"Threshold           : {THRESHOLD:.2f}")

print()

if detections:

    print("Detected windows:")

    for start_time, end_time, probability, _ in detections:

        print(
            f"{start_time:.1f} - "
            f"{end_time:.1f} sec | "
            f"{probability:.4f}"
        )

else:

    print("No Sanath detections.")


# ============================================================
# TOP 10
# ============================================================

print()
print("=" * 60)
print("TOP 10 SANATH PROBABILITIES")
print("=" * 60)

top_results = sorted(
    results,
    key=lambda x: x[2],
    reverse=True
)[:10]

for start_time, end_time, probability, _ in top_results:

    print(
        f"{start_time:.1f} - "
        f"{end_time:.1f} sec | "
        f"{probability:.4f}"
    )

print()
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)