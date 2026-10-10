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

# Probability required for a positive window
THRESHOLD = 0.50

# Number of consecutive positive windows required
MIN_POSITIVE_WINDOWS = 2

# After an event, ignore further detections for this long
COOLDOWN_SECONDS = 2.0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V6 model...")

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

duration = len(audio) / SR

print(f"Audio length: {duration:.2f} seconds")


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

    features.append(
        extract_features(window)
    )

    times.append(start / SR)


X = np.array(features)

X_scaled = scaler.transform(X)


# ============================================================
# PREDICTION
# ============================================================

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# TEMPORAL EVENT DETECTOR
# ============================================================

events = []

consecutive_count = 0
candidate_start = None

cooldown_until = -1.0


for i, probability in enumerate(probabilities):

    current_time = times[i]

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    if current_time < cooldown_until:
        continue

    # --------------------------------------------------------
    # Positive window
    # --------------------------------------------------------

    if probability >= THRESHOLD:

        if consecutive_count == 0:
            candidate_start = current_time

        consecutive_count += 1

    # --------------------------------------------------------
    # Negative window
    # --------------------------------------------------------

    else:

        # Enough consecutive positive windows
        if consecutive_count >= MIN_POSITIVE_WINDOWS:

            event_time = candidate_start

            event_end = current_time

            events.append(
                (
                    event_time,
                    event_end,
                    consecutive_count
                )
            )

            cooldown_until = (
                event_end +
                COOLDOWN_SECONDS
            )

        consecutive_count = 0
        candidate_start = None


# ============================================================
# HANDLE EVENT AT END OF AUDIO
# ============================================================

if consecutive_count >= MIN_POSITIVE_WINDOWS:

    event_time = candidate_start

    event_end = times[-1] + WINDOW_SECONDS

    events.append(
        (
            event_time,
            event_end,
            consecutive_count
        )
    )


# ============================================================
# PRINT WINDOW PROBABILITIES
# ============================================================

print("\n========================================")
print("WINDOW PROBABILITIES")
print("========================================")

for time, probability in zip(
    times,
    probabilities
):

    marker = "  <-- POSITIVE" if probability >= THRESHOLD else ""

    print(
        f"{time:5.1f}–"
        f"{time + WINDOW_SECONDS:5.1f} s"
        f"   {probability:.4f}"
        f"{marker}"
    )


# ============================================================
# PRINT EVENTS
# ============================================================

print("\n========================================")
print("SANATH EVENTS")
print("========================================")

if len(events) == 0:

    print("No events detected.")

else:

    for number, (
        start,
        end,
        count
    ) in enumerate(events, 1):

        print(
            f"EVENT {number}: "
            f"{start:.1f}–{end:.1f} s "
            f"({count} positive windows)"
        )


print("\n========================================")
print(f"Total events: {len(events)}")
print("========================================")