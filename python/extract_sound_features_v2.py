from pathlib import Path

import librosa
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"

OUTPUT = ROOT / "data" / "sound_features_v2.csv"


SR = 16000
N_FFT = 256
HOP_LENGTH = 128

N_MELS = 40
FMIN = 20
FMAX = 8000


def extract_features(wav_path):

    audio, _ = librosa.load(
        wav_path,
        sr=SR,
        mono=True
    )

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

    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    return mel_db.flatten()


files = []


# ============================================================
# BACKGROUND
# ============================================================

background_dir = (
    DATASET
    / "background"
    / "processed"
)

for wav in sorted(
    background_dir.glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "background",
        "session": "background_session_01"
    })


# ============================================================
# DOORBELL SESSION 01
# ============================================================

doorbell_root = (
    DATASET
    / "doorbell"
    / "labeled"
)

for wav in sorted(
    (
        doorbell_root
        / "event"
    ).glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "doorbell",
        "session": "doorbell_session_01"
    })


# ============================================================
# DOORBELL SESSION 02
# ============================================================

for wav in sorted(
    (
        doorbell_root
        / "session_02"
    ).glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "doorbell",
        "session": "doorbell_session_02"
    })


# ============================================================
# PHONE RING SESSION 01 POSITIVES
# ============================================================

phone_root = (
    DATASET
    / "phone_ring"
    / "labeled"
)


for wav in sorted(
    (
        phone_root
        / "session_01"
    ).glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "phone_ring",
        "session": "phone_ring_session_01"
    })


# ============================================================
# PHONE RING SESSION 02 POSITIVES
# ============================================================

for wav in sorted(
    (
        phone_root
        / "session_02"
    ).glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "phone_ring",
        "session": "phone_ring_session_02"
    })


# ============================================================
# PHONE RING NEGATIVES
# Treat as BACKGROUND
# ============================================================

for session_name in [
    "session_01",
    "session_02"
]:

    negative_dir = (
        phone_root
        / "negative"
        / session_name
    )

    for wav in sorted(
        negative_dir.glob("*.wav")
    ):

        files.append({
            "path": wav,
            "class": "background",
            "session": (
                f"phone_ring_negative_{session_name}"
            )
        })


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("SOUND FEATURE EXTRACTION V2")
print("=" * 60)

print(
    f"Total files found: {len(files)}"
)


print("\nClass distribution:")

for class_name in [
    "background",
    "doorbell",
    "phone_ring"
]:

    count = sum(
        item["class"] == class_name
        for item in files
    )

    print(
        f"{class_name:12s}: {count}"
    )


print("\nSession distribution:")

for session in sorted(
    set(
        item["session"]
        for item in files
    )
):

    count = sum(
        item["session"] == session
        for item in files
    )

    print(
        f"{session:35s}: {count}"
    )


# ============================================================
# EXTRACT
# ============================================================

rows = []

for index, item in enumerate(files):

    features = extract_features(
        item["path"]
    )

    row = {
        "filename": item["path"].name,
        "class": item["class"],
        "session": item["session"]
    }

    for i, value in enumerate(features):

        row[
            f"feature_{i}"
        ] = value

    rows.append(row)

    if (index + 1) % 25 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(files)}"
        )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT,
    index=False
)


print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Samples  : {len(df)}"
)

print(
    f"Features : 5040"
)

print(
    f"Output   : {OUTPUT}"
)

print("\nClass distribution:")

print(
    df["class"].value_counts()
)

print("\nSession distribution:")

print(
    df["session"].value_counts()
)