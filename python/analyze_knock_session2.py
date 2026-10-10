import os
import glob
import numpy as np
import librosa

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "knock",
    "processed",
    "knock_session_02"
)

files = []

for filepath in glob.glob(
    os.path.join(PROCESSED_DIR, "knock_*.wav")
):
    filename = os.path.basename(filepath)

    try:
        clip_num = int(
            filename.replace("knock_", "").replace(".wav", "")
        )
    except ValueError:
        continue

    files.append((clip_num, filepath))

files.sort()

print("=" * 60)
print("KNOCK SESSION 02 ANALYSIS")
print("=" * 60)

print(f"Session 02 clips: {len(files)}")

results = []

for clip_num, filepath in files:

    audio, sr = librosa.load(
        filepath,
        sr=None,
        mono=True
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    results.append((clip_num, rms, peak))

print("\nTOP ENERGY CLIPS")
print("-" * 60)

top = sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[:20]

for rank, (clip_num, rms, peak) in enumerate(top, 1):

    time = clip_num * 0.5

    print(
        f"{rank:2d}. knock_{clip_num:04d}.wav | "
        f"Time={time:5.1f}s | "
        f"RMS={rms:.6f} | "
        f"Peak={peak:.6f}"
    )

print("\nTIMELINE")
print("-" * 60)

for clip_num, rms, peak in results:

    time = clip_num * 0.5

    print(
        f"{clip_num:04d} | "
        f"{time:5.1f}s | "
        f"RMS={rms:.6f} | "
        f"Peak={peak:.6f}"
    )

print("=" * 60)