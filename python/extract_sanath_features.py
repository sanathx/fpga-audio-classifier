from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "dataset" / "sanath"
OUTPUT = ROOT / "data" / "sanath_features.csv"


# ============================================================
# AUDIO / MEL PARAMETERS
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
# FIND FILES
# ============================================================

files = []

# ------------------------------------------------------------
# POSITIVE
# ------------------------------------------------------------

positive_root = DATASET / "labeled" / "positive"

for session_dir in sorted(positive_root.iterdir()):

    if not session_dir.is_dir():
        continue

    for wav in sorted(session_dir.glob("*.wav")):

        files.append({
            "path": wav,
            "label": 1,
            "session": session_dir.name
        })


# ------------------------------------------------------------
# NEGATIVE
# ------------------------------------------------------------

negative_root = DATASET / "labeled" / "negative"

for category_dir in sorted(negative_root.iterdir()):

    if not category_dir.is_dir():
        continue

    for wav in sorted(category_dir.glob("*.wav")):

        files.append({
            "path": wav,
            "label": 0,
            "session": category_dir.name
        })


# ============================================================
# PROCESS
# ============================================================

print("=" * 60)
print("SANATH LOG-MEL FEATURE EXTRACTION")
print("=" * 60)

print(f"Positive files : {sum(x['label'] == 1 for x in files)}")
print(f"Negative files : {sum(x['label'] == 0 for x in files)}")
print(f"Total files    : {len(files)}")


rows = []

for index, item in enumerate(files):

    features = extract_features(item["path"])

    row = {
        "filename": item["path"].name,
        "label": item["label"],
        "session": item["session"]
    }

    for i, value in enumerate(features):

        row[f"feature_{i}"] = value

    rows.append(row)

    if (index + 1) % 25 == 0:
        print(
            f"Processed {index + 1}/{len(files)}"
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
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"Samples      : {len(df)}")
print(f"Features     : 5040")
print(f"Output       : {OUTPUT}")

print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

print("\nSession distribution:")
print(df["session"].value_counts())