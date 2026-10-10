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

WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.5

# Minimum confidence for a prediction to be considered
CONFIDENCE_THRESHOLD = 0.70

# Consecutive windows required for PHONE RING
PHONE_MIN_WINDOWS = 2

# Doorbell is short, so one strong prediction is enough.
DOORBELL_HIGH_CONFIDENCE = 0.90

# Prevent duplicate events
COOLDOWN_SECONDS = 2.0


def extract_features(clip):
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

    return log_mel.flatten().reshape(1, -1)


print("Loading Sound V3...")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

audio, _ = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

print(f"Audio length: {len(audio) / SR:.2f} seconds")

predictions = []

# ---------------------------------------------------------
# FRAME-LEVEL CLASSIFICATION
# ---------------------------------------------------------

for start in range(
    0,
    len(audio) - int(WINDOW_SECONDS * SR) + 1,
    int(STEP_SECONDS * SR)
):

    clip = audio[
        start:
        start + int(WINDOW_SECONDS * SR)
    ]

    features = extract_features(clip)
    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]

    best_index = np.argmax(probabilities)

    prediction = model.classes_[best_index]
    confidence = probabilities[best_index]

    time = start / SR

    predictions.append({
        "time": time,
        "class": prediction,
        "confidence": confidence
    })


# ---------------------------------------------------------
# TEMPORAL EVENT DETECTION
# ---------------------------------------------------------

events = []

last_event_time = -999

phone_run = []

for i, p in enumerate(predictions):

    current_time = p["time"]
    current_class = p["class"]
    confidence = p["confidence"]

    # -----------------------------------------
    # PHONE RING
    # -----------------------------------------

    if (
        current_class == "phone_ring"
        and confidence >= CONFIDENCE_THRESHOLD
    ):

        phone_run.append(p)

        # Need two consecutive windows
        if len(phone_run) >= PHONE_MIN_WINDOWS:

            event_time = phone_run[0]["time"]

            if event_time - last_event_time >= COOLDOWN_SECONDS:

                event_confidence = max(
                    x["confidence"]
                    for x in phone_run
                )

                events.append({
                    "time": event_time,
                    "type": "PHONE RING",
                    "confidence": event_confidence
                })

                last_event_time = event_time

            phone_run = []

    else:
        phone_run = []

    # -----------------------------------------
    # DOORBELL
    # -----------------------------------------

    if (
        current_class == "doorbell"
        and confidence >= DOORBELL_HIGH_CONFIDENCE
    ):

        if current_time - last_event_time >= COOLDOWN_SECONDS:

            events.append({
                "time": current_time,
                "type": "DOORBELL",
                "confidence": confidence
            })

            last_event_time = current_time


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("========================================")
print("SOUND EVENTS")
print("========================================")

if not events:
    print("No events detected.")

else:

    for i, event in enumerate(events, 1):

        print(
            f"EVENT {i}: {event['type']}"
        )

        print(
            f"Time: {event['time']:.1f} s"
        )

        print(
            f"Confidence: {event['confidence']:.4f}"
        )

        print()


print("========================================")
print("TOTAL EVENTS:", len(events))
print("========================================")