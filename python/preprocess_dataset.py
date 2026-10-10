import os
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "background",
    "background_session_01.wav"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "background",
    "processed"
)

TARGET_SR = 16000

CLIP_DURATION = 1.0       # seconds
HOP_DURATION = 0.5        # seconds

CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)
HOP_SAMPLES = int(TARGET_SR * HOP_DURATION)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AUDIO
# ============================================================

print("=" * 60)
print("BACKGROUND DATASET PREPROCESSING")
print("=" * 60)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_DIR}")
print()

print("Loading audio...")

audio, sr = librosa.load(
    INPUT_FILE,
    sr=TARGET_SR,
    mono=True
)

print(f"Original/loaded sample rate : {TARGET_SR} Hz")
print(f"Samples                     : {len(audio)}")
print(f"Duration                    : {len(audio) / TARGET_SR:.2f} s")

# Normalize only if necessary.
# We do NOT aggressively normalize the audio because
# loudness is potentially useful information.
peak = np.max(np.abs(audio))

if peak > 1.0:
    audio = audio / peak

print(f"Peak amplitude              : {np.max(np.abs(audio)):.4f}")
print()


# ============================================================
# CREATE OVERLAPPING CLIPS
# ============================================================

print("Creating 1-second clips...")

clip_number = 0
start = 0

while start + CLIP_SAMPLES <= len(audio):

    clip = audio[start:start + CLIP_SAMPLES]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"background_{clip_number:04d}.wav"
    )

    sf.write(
        output_file,
        clip,
        TARGET_SR,
        subtype="PCM_16"
    )

    clip_number += 1
    start += HOP_SAMPLES


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Generated clips : {clip_number}")
print(f"Clip duration   : {CLIP_DURATION:.1f} s")
print(f"Hop duration    : {HOP_DURATION:.1f} s")
print(f"Sample rate     : {TARGET_SR} Hz")
print(f"Output folder   : {OUTPUT_DIR}")
print("=" * 60)