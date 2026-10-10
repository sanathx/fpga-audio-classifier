import os
import numpy as np
import librosa
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "background",
    "processed",
    "background_0000.wav"
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

# -----------------------------
# Temporal parameters
# -----------------------------

PHONE_THRESHOLD = 0.70
PHONE_START_WINDOWS = 2

# Once phone ring is active, allow occasional
# misclassified windows without ending the event.
PHONE_MAX_GAP_WINDOWS = 3

# Doorbell needs a strong prediction AND
# should not be inside a phone-ring event.
DOORBELL_THRESHOLD = 0.90

# Prevent duplicate notifications.
COOLDOWN_SECONDS = 3.0


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

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

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

print(
    f"Audio length: "
    f"{len(audio) / SR:.2f} seconds"
)


# =========================================================
# FRAME CLASSIFICATION
# =========================================================

predictions = []

step_samples = int(STEP_SECONDS * SR)
window_samples = int(WINDOW_SECONDS * SR)

for start in range(
    0,
    len(audio) - window_samples + 1,
    step_samples
):

    clip = audio[
        start:
        start + window_samples
    ]

    features = extract_features(clip)

    features = scaler.transform(features)

    probabilities = model.predict_proba(features)[0]

    best_index = np.argmax(probabilities)

    prediction = model.classes_[best_index]

    confidence = probabilities[best_index]

    predictions.append({
        "time": start / SR,
        "class": prediction,
        "confidence": confidence
    })


# =========================================================
# STATE MACHINE
# =========================================================

events = []

phone_active = False
phone_candidates = []
phone_gap = 0

last_event_time = -999


def emit_event(event_type, time, confidence):

    global last_event_time

    if time - last_event_time < COOLDOWN_SECONDS:
        return

    events.append({
        "type": event_type,
        "time": time,
        "confidence": confidence
    })

    last_event_time = time


for p in predictions:

    time = p["time"]
    cls = p["class"]
    confidence = p["confidence"]

    phone_positive = (
        cls == "phone_ring"
        and confidence >= PHONE_THRESHOLD
    )

    # =====================================================
    # PHONE RING STATE
    # =====================================================

    if phone_active:

        if phone_positive:

            phone_gap = 0

        else:

            phone_gap += 1

            # End phone event only after several
            # consecutive non-phone windows.
            if phone_gap >= PHONE_MAX_GAP_WINDOWS:

                phone_active = False
                phone_candidates = []
                phone_gap = 0

        # While phone event is active,
        # ignore isolated doorbell predictions.
        continue


    # =====================================================
    # LOOK FOR PHONE RING START
    # =====================================================

    if phone_positive:

        phone_candidates.append(p)

    else:

        phone_candidates = []


    if len(phone_candidates) >= PHONE_START_WINDOWS:

        start_event = phone_candidates[0]

        emit_event(
            "PHONE RING",
            start_event["time"],
            max(
                x["confidence"]
                for x in phone_candidates
            )
        )

        phone_active = True
        phone_gap = 0
        phone_candidates = []

        continue


    # =====================================================
    # DOORBELL
    # =====================================================

    if (
        cls == "doorbell"
        and confidence >= DOORBELL_THRESHOLD
    ):

        emit_event(
            "DOORBELL",
            time,
            confidence
        )


# =========================================================
# RESULTS
# =========================================================

print()
print("========================================")
print("SOUND EVENTS V4")
print("========================================")

if not events:

    print("No events detected.")

else:

    for i, event in enumerate(events, 1):

        print(
            f"EVENT {i}: "
            f"{event['type']}"
        )

        print(
            f"Time: "
            f"{event['time']:.1f} s"
        )

        print(
            f"Confidence: "
            f"{event['confidence']:.4f}"
        )

        print()


print("========================================")
print(
    f"TOTAL EVENTS: {len(events)}"
)
print("========================================")