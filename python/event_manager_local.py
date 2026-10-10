import os
import numpy as np
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

# Test audio
AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "sanath_test_01.wav"
)

# Sanath
SANATH_MODEL = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_mlp_v6.joblib"
)

SANATH_SCALER = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sanath_scaler_v6.joblib"
)

# Generic sound classifier
SOUND_MODEL = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sound_mlp_v3.joblib"
)

SOUND_SCALER = os.path.join(
    BASE_DIR,
    "data",
    "models",
    "sound_scaler_v3.joblib"
)


# ============================================================
# DSP SETTINGS
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
# EVENT SETTINGS
# ============================================================

SANATH_HIGH = 0.80
SANATH_LOW = 0.50

SANATH_MODERATE_REQUIRED = 2

COOLDOWN_SECONDS = 2.0


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

sanath_model = joblib.load(SANATH_MODEL)
sanath_scaler = joblib.load(SANATH_SCALER)

sound_model = joblib.load(SOUND_MODEL)
sound_scaler = joblib.load(SOUND_SCALER)


print("Models loaded.")


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

    times.append(
        start / SR
    )


X = np.array(features)


# ============================================================
# RUN MODELS
# ============================================================

print("\nRunning classifiers...")

# Sanath
X_sanath = sanath_scaler.transform(X)

sanath_probability = sanath_model.predict_proba(
    X_sanath
)[:, 1]


# Generic environmental sounds
X_sound = sound_scaler.transform(X)

sound_predictions = sound_model.predict(
    X_sound
)

sound_probabilities = sound_model.predict_proba(
    X_sound
)

sound_classes = sound_model.classes_


# ============================================================
# PRINT RAW CLASSIFIER RESULTS
# ============================================================

print("\n========================================")
print("CLASSIFIER RESULTS")
print("========================================")

for i, time in enumerate(times):

    sound_label = sound_predictions[i]

    sound_confidence = np.max(
        sound_probabilities[i]
    )

    print(
        f"{time:5.1f}–"
        f"{time + WINDOW_SECONDS:5.1f} s | "
        f"Sanath={sanath_probability[i]:.3f} | "
        f"Sound={sound_label:<12} "
        f"({sound_confidence:.3f})"
    )


# ============================================================
# EVENT MANAGER
# ============================================================

events = []

sanath_moderate_count = 0
sanath_start = None

cooldown_until = -1.0


for i, time in enumerate(times):

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if time < cooldown_until:
        continue


    sanath_p = sanath_probability[i]

    sound_label = sound_predictions[i]

    sound_confidence = np.max(
        sound_probabilities[i]
    )


    # ========================================================
    # SANATH
    # ========================================================

    if sanath_p >= SANATH_HIGH:

        events.append({
            "time": time,
            "event": "SANATH",
            "confidence": sanath_p
        })

        cooldown_until = (
            time +
            WINDOW_SECONDS +
            COOLDOWN_SECONDS
        )

        sanath_moderate_count = 0
        sanath_start = None

        continue


    if sanath_p >= SANATH_LOW:

        if sanath_moderate_count == 0:
            sanath_start = time

        sanath_moderate_count += 1

        if (
            sanath_moderate_count >=
            SANATH_MODERATE_REQUIRED
        ):

            events.append({
                "time": sanath_start,
                "event": "SANATH",
                "confidence": sanath_p
            })

            cooldown_until = (
                time +
                WINDOW_SECONDS +
                COOLDOWN_SECONDS
            )

            sanath_moderate_count = 0
            sanath_start = None

            continue

    else:

        sanath_moderate_count = 0
        sanath_start = None


    # ========================================================
    # ENVIRONMENTAL SOUND
    # ========================================================

    if sound_label == "background":
        continue


    if sound_confidence < 0.70:
        continue


    if sound_label == "doorbell":

        events.append({
            "time": time,
            "event": "DOORBELL",
            "confidence": sound_confidence
        })

    elif sound_label == "phone_ring":

        events.append({
            "time": time,
            "event": "PHONE_RING",
            "confidence": sound_confidence
        })


    cooldown_until = (
        time +
        WINDOW_SECONDS +
        COOLDOWN_SECONDS
    )


# ============================================================
# PRINT FINAL EVENTS
# ============================================================

print("\n========================================")
print("FINAL EVENTS")
print("========================================")

if not events:

    print("No events detected.")

else:

    for number, event in enumerate(
        events,
        1
    ):

        print(
            f"EVENT {number}: "
            f"{event['event']}"
        )

        print(
            f"  Time       : "
            f"{event['time']:.1f} s"
        )

        print(
            f"  Confidence : "
            f"{event['confidence']:.4f}"
        )


print("\n========================================")
print(
    f"TOTAL EVENTS: {len(events)}"
)
print("========================================")