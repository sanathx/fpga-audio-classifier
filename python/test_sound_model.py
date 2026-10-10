from pathlib import Path

import joblib
import librosa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "data" / "models" / "sound_mlp_v2.joblib"
SCALER_FILE = ROOT / "data" / "models" / "sound_scaler_v2.joblib"


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

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)


# ============================================================
# ASK FOR AUDIO FILE
# ============================================================

audio_path = input(
    "Enter path to WAV file: "
).strip().strip('"')


audio_path = Path(audio_path)


if not audio_path.exists():

    print(
        f"\nERROR: File not found:\n"
        f"{audio_path}"
    )

    raise SystemExit


# ============================================================
# LOAD AUDIO
# ============================================================

audio, _ = librosa.load(
    audio_path,
    sr=SR,
    mono=True
)


duration = len(audio) / SR


print("\n" + "=" * 60)
print("UNSEEN AUDIO TEST")
print("=" * 60)

print(f"File     : {audio_path.name}")
print(f"Duration : {duration:.2f} seconds")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(segment):

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
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
# SLIDING-WINDOW PREDICTION
# ============================================================

window_samples = int(
    WINDOW_SECONDS * SR
)

hop_samples = int(
    HOP_SECONDS * SR
)


results = []


for start in range(
    0,
    len(audio) - window_samples + 1,
    hop_samples
):

    end = start + window_samples

    segment = audio[start:end]

    features = extract_features(
        segment
    )

    X = features.reshape(1, -1)

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(
        X_scaled
    )[0]

    classes = model.classes_

    best_index = np.argmax(
        probabilities
    )

    predicted_class = classes[
        best_index
    ]

    confidence = probabilities[
        best_index
    ]

    start_time = start / SR
    end_time = end / SR

    results.append({
        "start": start_time,
        "end": end_time,
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            classes[i]: probabilities[i]
            for i in range(len(classes))
        }
    })


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FRAME-BY-FRAME PREDICTIONS")
print("=" * 60)

for result in results:

    print(
        f"{result['start']:6.2f} - "
        f"{result['end']:6.2f}  "
        f"{result['class']:12s}  "
        f"{result['confidence'] * 100:6.2f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)


class_counts = {}

for result in results:

    class_name = result["class"]

    class_counts[class_name] = (
        class_counts.get(class_name, 0) + 1
    )


for class_name, count in class_counts.items():

    print(
        f"{class_name:12s}: "
        f"{count} frames"
    )


print("\nTest complete.")