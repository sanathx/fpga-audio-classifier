import os
import warnings

import joblib
import librosa
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ------------------------------------------------------------
# CHANGE ONLY THIS PATH WHEN TESTING A DIFFERENT RECORDING
# ------------------------------------------------------------

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "phone_ring",
    "phone_ring_session_02.wav"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "models"
)

SANATH_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sanath_mlp_v6.joblib"
)

SANATH_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "sanath_scaler_v6.joblib"
)

SOUND_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sound_mlp_v3.joblib"
)

SOUND_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "sound_scaler_v3.joblib"
)


# ============================================================
# AUDIO / FEATURE SETTINGS
# ============================================================

SR = 16000

WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(
    SR * WINDOW_SECONDS
)

HOP_SECONDS = 0.5
HOP_SAMPLES = int(
    SR * HOP_SECONDS
)

N_FFT = 256
HOP_LENGTH = 128

N_MELS = 40
FMIN = 20
FMAX = 8000


# ============================================================
# SANATH SETTINGS
# ============================================================

SANATH_HIGH_THRESHOLD = 0.80
SANATH_LOW_THRESHOLD = 0.50

SANATH_MODERATE_WINDOWS = 2
SANATH_COOLDOWN = 2.0


# ============================================================
# PHONE / DOORBELL SETTINGS
# ============================================================

PHONE_THRESHOLD = 0.70
PHONE_START_WINDOWS = 2

# Number of consecutive non-phone windows allowed before
# the phone state is considered finished.
PHONE_MAX_GAP_WINDOWS = 3

DOORBELL_THRESHOLD = 0.90

SOUND_COOLDOWN = 3.0

# If a phone ringtone is confirmed, remove Sanath events
# that occurred shortly before the phone confirmation.
PHONE_RETROACTIVE_WINDOW = 4.0


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

sanath_model = joblib.load(
    SANATH_MODEL_PATH
)

sanath_scaler = joblib.load(
    SANATH_SCALER_PATH
)

sound_model = joblib.load(
    SOUND_MODEL_PATH
)

sound_scaler = joblib.load(
    SOUND_SCALER_PATH
)

print("Models loaded.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
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

    mean = np.mean(mel_db)
    std = np.std(mel_db)

    if std < 1e-8:
        std = 1.0

    mel_db = (
        mel_db - mean
    ) / std

    return mel_db.flatten()


# ============================================================
# LOAD AUDIO
# ============================================================

if not os.path.exists(AUDIO_FILE):

    raise FileNotFoundError(
        f"\nAudio file not found:\n"
        f"{AUDIO_FILE}\n"
    )


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

warnings.filterwarnings(
    "ignore",
    message="PySoundFile failed"
)

warnings.filterwarnings(
    "ignore",
    message="librosa.core.audio.__audioread_load"
)


audio, _ = librosa.load(
    AUDIO_FILE,
    sr=SR,
    mono=True
)

duration = len(audio) / SR

print(
    f"Audio length: "
    f"{duration:.2f} seconds"
)


# ============================================================
# VERIFY FEATURE SIZE
# ============================================================

test_frame = audio[:WINDOW_SAMPLES]

if len(test_frame) < WINDOW_SAMPLES:

    test_frame = np.pad(
        test_frame,
        (
            0,
            WINDOW_SAMPLES - len(test_frame)
        )
    )

test_features = extract_features(
    test_frame
)

print(
    f"Feature size: "
    f"{len(test_features)}"
)

if len(test_features) != 5040:

    raise ValueError(
        f"Expected 5040 features, "
        f"but got {len(test_features)}."
    )


# ============================================================
# EVENT STORAGE
# ============================================================

events = []


# ============================================================
# SANATH STATE
# ============================================================

sanath_positive_run = 0

last_sanath_event_time = -999.0


# ============================================================
# PHONE STATE
# ============================================================

phone_active = False

phone_positive_run = 0

phone_gap = 0

last_sound_event_time = -999.0


# ============================================================
# PROCESS AUDIO
# ============================================================

for start in range(
    0,
    len(audio) - WINDOW_SAMPLES + 1,
    HOP_SAMPLES
):

    frame = audio[
        start:start + WINDOW_SAMPLES
    ]

    current_time = (
        start / SR
    )


    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    features = extract_features(
        frame
    )

    X = features.reshape(
        1,
        -1
    )


    # ========================================================
    # SANATH MODEL
    # ========================================================

    X_sanath = (
        sanath_scaler.transform(X)
    )

    sanath_probability = float(
        sanath_model.predict_proba(
            X_sanath
        )[0][1]
    )


    # ========================================================
    # SOUND MODEL
    # ========================================================

    X_sound = (
        sound_scaler.transform(X)
    )

    sound_probabilities = (
        sound_model.predict_proba(
            X_sound
        )[0]
    )

    sound_classes = (
        sound_model.classes_
    )

    sound_probs = {
        str(cls): float(prob)
        for cls, prob in zip(
            sound_classes,
            sound_probabilities
        )
    }

    phone_conf = sound_probs.get(
        "phone_ring",
        0.0
    )

    doorbell_conf = sound_probs.get(
        "doorbell",
        0.0
    )


    # ========================================================
    # STEP 1 — PHONE STATE
    # ========================================================

    phone_positive = (
        phone_conf >= PHONE_THRESHOLD
    )


    if phone_active:

        # ----------------------------------------------------
        # PHONE ALREADY ACTIVE
        # ----------------------------------------------------

        if phone_positive:

            phone_positive_run += 1

            phone_gap = 0

        else:

            phone_gap += 1

            phone_positive_run = 0


        # Allow short gaps in the ringtone classification.
        if (
            phone_gap
            > PHONE_MAX_GAP_WINDOWS
        ):

            phone_active = False

            phone_gap = 0

            phone_positive_run = 0


    else:

        # ----------------------------------------------------
        # PHONE NOT ACTIVE
        # ----------------------------------------------------

        if phone_positive:

            phone_positive_run += 1

        else:

            phone_positive_run = 0


        # ----------------------------------------------------
        # PHONE RING CONFIRMED
        # ----------------------------------------------------

        if (
            phone_positive_run
            >= PHONE_START_WINDOWS
        ):

            if (
                current_time
                - last_sound_event_time
                >= SOUND_COOLDOWN
            ):

                events.append({
                    "type": "PHONE RING",
                    "time": current_time,
                    "confidence": phone_conf
                })

                last_sound_event_time = (
                    current_time
                )


                # ------------------------------------------------
                # RETROACTIVE FALSE-SANATH SUPPRESSION
                # ------------------------------------------------
                #
                # A phone ringtone may initially trigger the
                # Sanath model before enough phone-ring windows
                # accumulate.
                #
                # Once PHONE RING is confirmed, remove recent
                # Sanath events from the same audio region.
                #

                events = [
                    event
                    for event in events
                    if not (
                        event["type"] == "SANATH"
                        and
                        0
                        <= current_time - event["time"]
                        <= PHONE_RETROACTIVE_WINDOW
                    )
                ]


            phone_active = True

            phone_gap = 0


    # ========================================================
    # STEP 2 — SANATH
    # ========================================================

    # If the phone is active, do not generate Sanath events.

    if phone_active:

        sanath_positive_run = 0

    else:

        # ----------------------------------------------------
        # HIGH CONFIDENCE SANATH
        # ----------------------------------------------------

        if (
            sanath_probability
            >= SANATH_HIGH_THRESHOLD
        ):

            if (
                current_time
                - last_sanath_event_time
                >= SANATH_COOLDOWN
            ):

                events.append({
                    "type": "SANATH",
                    "time": current_time,
                    "confidence":
                        sanath_probability
                })

                last_sanath_event_time = (
                    current_time
                )

            sanath_positive_run = 0


        # ----------------------------------------------------
        # MODERATE CONFIDENCE SANATH
        # ----------------------------------------------------

        elif (
            sanath_probability
            >= SANATH_LOW_THRESHOLD
        ):

            sanath_positive_run += 1

            if (
                sanath_positive_run
                >= SANATH_MODERATE_WINDOWS
            ):

                if (
                    current_time
                    - last_sanath_event_time
                    >= SANATH_COOLDOWN
                ):

                    events.append({
                        "type": "SANATH",
                        "time": current_time,
                        "confidence":
                            sanath_probability
                    })

                    last_sanath_event_time = (
                        current_time
                    )

                sanath_positive_run = 0


        # ----------------------------------------------------
        # NOT SANATH
        # ----------------------------------------------------

        else:

            sanath_positive_run = 0


    # ========================================================
    # STEP 3 — DOORBELL
    # ========================================================

    # Doorbell is suppressed while phone is active.

    if not phone_active:

        if (
            doorbell_conf
            >= DOORBELL_THRESHOLD
        ):

            if (
                current_time
                - last_sound_event_time
                >= SOUND_COOLDOWN
            ):

                events.append({
                    "type": "DOORBELL",
                    "time": current_time,
                    "confidence":
                        doorbell_conf
                })

                last_sound_event_time = (
                    current_time
                )


# ============================================================
# SORT EVENTS
# ============================================================

events.sort(
    key=lambda event: event["time"]
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 40)
print("FINAL EVENT MANAGER")
print("=" * 40)

if len(events) == 0:

    print("No events detected.")

else:

    for i, event in enumerate(
        events,
        start=1
    ):

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


print("=" * 40)

print(
    f"TOTAL EVENTS: "
    f"{len(events)}"
)