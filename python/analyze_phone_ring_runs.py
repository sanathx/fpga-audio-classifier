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

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

audio, _ = librosa.load(AUDIO_FILE, sr=SR, mono=True)

results = []

# 1-second windows, 0.5-second step
STEP = SR // 2

for start in range(0, len(audio) - SR + 1, STEP):

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

    log_mel = (
        log_mel - np.mean(log_mel)
    ) / (np.std(log_mel) + 1e-8)

    features = log_mel.flatten().reshape(1, -1)
    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]

    idx = np.argmax(probabilities)
    prediction = model.classes_[idx]
    confidence = probabilities[idx]

    results.append((
        start / SR,
        prediction,
        confidence
    ))

# Find consecutive runs of the same prediction
runs = []

current_class = results[0][1]
run_start = results[0][0]
run_end = results[0][0] + 1.0
confidences = [results[0][2]]

for time, prediction, confidence in results[1:]:

    if prediction == current_class:
        run_end = time + 1.0
        confidences.append(confidence)

    else:
        runs.append((
            run_start,
            run_end,
            current_class,
            max(confidences),
            len(confidences)
        ))

        current_class = prediction
        run_start = time
        run_end = time + 1.0
        confidences = [confidence]

runs.append((
    run_start,
    run_end,
    current_class,
    max(confidences),
    len(confidences)
))

print("\n========================================")
print("PHONE RING PREDICTION RUNS")
print("========================================")

for start, end, cls, confidence, count in runs:
    print(
        f"{start:6.1f}–{end:6.1f} s  "
        f"{cls:<12} "
        f"{count:2d} windows  "
        f"max_conf={confidence:.4f}"
    )

print("\n========================================")
print("DOORBELL RUNS")
print("========================================")

doorbell_runs = [
    r for r in runs
    if r[2] == "doorbell"
]

if not doorbell_runs:
    print("No doorbell predictions.")

else:
    for start, end, cls, confidence, count in doorbell_runs:
        print(
            f"{start:6.1f}–{end:6.1f} s  "
            f"{count:2d} windows  "
            f"max_conf={confidence:.4f}"
        )