import os
import numpy as np
import librosa
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "phone_ring",
    "phone_ring_session_02.wav"
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

SR = 16000
N_FFT = 256
HOP = 128
N_MELS = 40

print("Loading Sound V3...")
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

audio, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True)

print(f"Audio length: {len(audio) / SR:.2f} seconds")

results = []

for start in range(0, len(audio) - SR + 1, HOP * 4):
    clip = audio[start:start + SR]

    mel = librosa.feature.melspectrogram(
        y=clip,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000,
        power=2
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Same per-clip standardization used during training
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

    features = log_mel.flatten().reshape(1, -1)

    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]
    prediction = model.classes_[np.argmax(probabilities)]
    confidence = np.max(probabilities)

    time = start / SR

    results.append((time, prediction, confidence))

print("\n========================================")
print("PHONE RING SEQUENCE")
print("========================================")

for time, prediction, confidence in results:
    print(
        f"{time:5.1f} s  "
        f"{prediction:<12} "
        f"confidence={confidence:.4f}"
    )

print("\n========================================")
print("SUMMARY")
print("========================================")

for cls in model.classes_:
    count = sum(1 for _, p, _ in results if p == cls)
    print(f"{cls:<12}: {count}")