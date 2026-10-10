import os
import numpy as np
import librosa
import soundfile as sf

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

# Change this for each event class/session
CLASS_NAME = "sanath"
SESSION_NAME = "sanath_session_02"

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    CLASS_NAME,
    SESSION_NAME + ".wav"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    CLASS_NAME,
    "processed",
    SESSION_NAME
)

TARGET_SR = 16000
CLIP_DURATION = 1.0
HOP_DURATION = 0.5

CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)
HOP_SAMPLES = int(TARGET_SR * HOP_DURATION)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EVENT DATASET PREPROCESSING")
print("=" * 60)

print(f"Class  : {CLASS_NAME}")
print(f"Session: {SESSION_NAME}")
print(f"Input  : {INPUT_FILE}")
print(f"Output : {OUTPUT_DIR}")
print()

# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input file does not exist.")
    print(INPUT_FILE)
    raise SystemExit

# ------------------------------------------------------------
# Load audio
# ------------------------------------------------------------

print("Loading audio...")

audio, sr = librosa.load(
    INPUT_FILE,
    sr=TARGET_SR,
    mono=True
)

print(f"Loaded sample rate : {TARGET_SR} Hz")
print(f"Samples            : {len(audio)}")
print(f"Duration           : {len(audio) / TARGET_SR:.2f} s")

# ------------------------------------------------------------
# Normalize only if necessary
# ------------------------------------------------------------

peak = np.max(np.abs(audio))

if peak > 1.0:
    audio = audio / peak

print(f"Peak amplitude     : {np.max(np.abs(audio)):.4f}")
print()

# ------------------------------------------------------------
# Create clips
# ------------------------------------------------------------

print("Creating 1-second clips...")

clip_number = 0
start = 0

while start + CLIP_SAMPLES <= len(audio):

    clip = audio[start:start + CLIP_SAMPLES]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{CLASS_NAME}_{clip_number:04d}.wav"
    )

    sf.write(
        output_file,
        clip,
        TARGET_SR,
        subtype="PCM_16"
    )

    clip_number += 1
    start += HOP_SAMPLES

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Generated clips : {clip_number}")
print(f"Clip duration   : {CLIP_DURATION:.1f} s")
print(f"Hop duration    : {HOP_DURATION:.1f} s")
print(f"Sample rate      : {TARGET_SR} Hz")
print(f"Output folder    : {OUTPUT_DIR}")
print("=" * 60)