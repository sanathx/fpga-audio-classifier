import os
import glob
import numpy as np
import librosa

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

CLASS_NAME = "doorbell"

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    CLASS_NAME,
    "processed"
)

TARGET_SR = 16000

# Session 01 contained clips 0000-0044.
# Session 02 therefore starts at clip 0045.
SESSION2_START = 45


print("=" * 60)
print("DOORBELL SESSION 02 ANALYSIS")
print("=" * 60)

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.wav"))
)

# Keep only Session 02
files = [
    f for f in files
    if int(os.path.basename(f).split("_")[-1].split(".")[0])
    >= SESSION2_START
]

if len(files) == 0:
    print("ERROR: No Session 02 clips found.")
    raise SystemExit

print(f"Session 02 clips: {len(files)}")
print()

# ------------------------------------------------------------
# Analyze
# ------------------------------------------------------------

results = []

for file_path in files:

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    results.append(
        (
            file_path,
            rms,
            peak
        )
    )

# Sort by RMS
results.sort(
    key=lambda x: x[1],
    reverse=True
)

# ------------------------------------------------------------
# Print top clips
# ------------------------------------------------------------

print("=" * 60)
print("TOP 20 HIGHEST-ENERGY SESSION 02 CLIPS")
print("=" * 60)

for rank, (file_path, rms, peak) in enumerate(
    results[:20],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{os.path.basename(file_path)} | "
        f"RMS={rms:.6f} | "
        f"Peak={peak:.6f}"
    )

# ------------------------------------------------------------
# Print clips in chronological order
# ------------------------------------------------------------

print()
print("=" * 60)
print("SESSION 02 CLIP TIMELINE")
print("=" * 60)

chronological = sorted(
    results,
    key=lambda x: x[0]
)

for file_path, rms, peak in chronological:

    clip_number = int(
        os.path.basename(file_path)
        .split("_")[-1]
        .split(".")[0]
    )

    # Session 02 local clip number
    local_clip = clip_number - SESSION2_START

    # Each clip starts 0.5 seconds after previous clip
    start_time = local_clip * 0.5

    print(
        f"{os.path.basename(file_path)} | "
        f"Session2 time ≈ {start_time:6.1f}s | "
        f"RMS={rms:.6f}"
    )

print()
print("=" * 60)
print("DONE")
print("=" * 60)