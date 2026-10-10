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

# Very confident prediction
HIGH_THRESHOLD = 0.80

# Moderate prediction
LOW_THRESHOLD = 0.50

# Number of moderate predictions required
MODERATE_WINDOWS_REQUIRED = 2

# Prevent repeated alerts
COOLDOWN_SECONDS = 2.0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Sanath V6...")

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
# CREATE AUDIO WINDOWS
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
# PREDICT
# ============================================================

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# EVENT DETECTOR
# ============================================================

events = []

moderate_count = 0
moderate_start = None

cooldown_until = -1.0


for i, probability in enumerate(probabilities):

    current_time = times[i]

    # --------------------------------------------------------
    # Ignore predictions during cooldown
    # --------------------------------------------------------

    if current_time < cooldown_until:
        continue


    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    if probability >= HIGH_THRESHOLD:

        event_start = current_time

        event_end = (
            current_time +
            WINDOW_SECONDS
        )

        events.append({
            "start": event_start,
            "end": event_end,
            "probability": probability,
            "reason": "high-confidence"
        })

        cooldown_until = (
            event_end +
            COOLDOWN_SECONDS
        )

        moderate_count = 0
        moderate_start = None

        continue


    # --------------------------------------------------------
    # MODERATE CONFIDENCE
    # --------------------------------------------------------

    if probability >= LOW_THRESHOLD:

        if moderate_count == 0:

            moderate_start = current_time

        moderate_count += 1


        # Enough moderate windows
        if (
            moderate_count >=
            MODERATE_WINDOWS_REQUIRED
        ):

            event_start = moderate_start

            event_end = (
                current_time +
                WINDOW_SECONDS
            )

            events.append({
                "start": event_start,
                "end": event_end,
                "probability": probability,
                "reason": "temporal-confirmation"
            })

            cooldown_until = (
                event_end +
                COOLDOWN_SECONDS
            )

            moderate_count = 0
            moderate_start = None

    else:

        moderate_count = 0
        moderate_start = None


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FINAL SANATH EVENT DETECTOR")
print("========================================")

if len(events) == 0:

    print("No Sanath events detected.")

else:

    for number, event in enumerate(
        events,
        1
    ):

        print(
            f"EVENT {number}: "
            f"{event['start']:.1f}–"
            f"{event['end']:.1f} s"
        )

        print(
            f"  Probability: "
            f"{event['probability']:.4f}"
        )

        print(
            f"  Reason: "
            f"{event['reason']}"
        )


print("\n========================================")
print(f"Total events: {len(events)}")
print("========================================")