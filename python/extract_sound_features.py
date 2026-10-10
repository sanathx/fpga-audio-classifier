from pathlib import Path

import librosa
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset"

OUTPUT = ROOT / "data" / "sound_features.csv"


# ============================================================
# PARAMETERS
# ============================================================

SR = 16000
N_FFT = 256
HOP_LENGTH = 128

N_MELS = 40
FMIN = 20
FMAX = 8000


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

    # Per-clip normalization
    mel_db = (
        mel_db - np.mean(mel_db)
    ) / (
        np.std(mel_db) + 1e-8
    )

    return mel_db.flatten()


# ============================================================
# FIND DATASET FILES
# ============================================================

files = []


# ------------------------------------------------------------
# BACKGROUND
# ------------------------------------------------------------

background_dir = (
    DATASET
    / "background"
    / "processed"
)

for wav in sorted(background_dir.glob("*.wav")):

    files.append({
        "path": wav,
        "class": "background",
        "session": "background_session_01"
    })


# ------------------------------------------------------------
# DOORBELL
# ------------------------------------------------------------

doorbell_root = (
    DATASET
    / "doorbell"
    / "labeled"
)

# Session 01
doorbell_session_01 = (
    doorbell_root
    / "event"
)

for wav in sorted(
    doorbell_session_01.glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "doorbell",
        "session": "doorbell_session_01"
    })


# Session 02
doorbell_session_02 = (
    doorbell_root
    / "session_02"
)

for wav in sorted(
    doorbell_session_02.glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "doorbell",
        "session": "doorbell_session_02"
    })


# ------------------------------------------------------------
# KNOCK
# ------------------------------------------------------------

knock_root = (
    DATASET
    / "knock"
    / "labeled"
)

# Session 01
knock_session_01 = (
    knock_root
    / "session_01"
)

for wav in sorted(
    knock_session_01.glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "knock",
        "session": "knock_session_01"
    })


# Session 02
knock_session_02 = (
    knock_root
    / "session_02"
)

for wav in sorted(
    knock_session_02.glob("*.wav")
):

    files.append({
        "path": wav,
        "class": "knock",
        "session": "knock_session_02"
    })


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("GENERAL SOUND FEATURE EXTRACTION")
print("=" * 60)

print(
    f"Total files found: {len(files)}"
)

print("\nClass distribution:")

for class_name in [
    "background",
    "doorbell",
    "knock"
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
    set(item["session"] for item in files)
):

    count = sum(
        item["session"] == session
        for item in files
    )

    print(
        f"{session:25s}: {count}"
    )


# ============================================================
# EXTRACT FEATURES
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

        row[f"feature_{i}"] = value

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


# ============================================================
# FINAL SUMMARY
# ============================================================

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