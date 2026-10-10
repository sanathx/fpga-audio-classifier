import os
import numpy as np
import librosa
import soundfile as sf

# ============================================================
# SANATH SESSION 04 PREPROCESSING
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "sanath_session_04.wav"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "processed",
    "sanath_session_04"
)

# Target audio format
TARGET_SR = 16000

# Sliding-window settings
CLIP_DURATION = 1.0       # seconds
HOP_DURATION = 0.5        # seconds

CLIP_SAMPLES = int(TARGET_SR * CLIP_DURATION)
HOP_SAMPLES = int(TARGET_SR * HOP_DURATION)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    print("ERROR: Input file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading:")
print(INPUT_FILE)

audio, sr = librosa.load(
    INPUT_FILE,
    sr=TARGET_SR,
    mono=True
)

audio = audio.astype(np.float32)

print()
print("Input converted to:")
print(f"Sample rate : {TARGET_SR} Hz")
print(f"Channels    : Mono")
print(f"Samples     : {len(audio)}")
print(f"Duration    : {len(audio) / TARGET_SR:.2f} sec")
print(f"Peak        : {np.max(np.abs(audio)):.4f}")


# ============================================================
# CREATE 1-SECOND CLIPS WITH 0.5-SECOND HOP
# ============================================================

num_clips = 0
start = 0

while start + CLIP_SAMPLES <= len(audio):

    clip = audio[start:start + CLIP_SAMPLES]

    filename = f"sanath_{num_clips:04d}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)

    sf.write(
        output_path,
        clip,
        TARGET_SR,
        subtype="PCM_16"
    )

    num_clips += 1
    start += HOP_SAMPLES


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SESSION 04 PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Output directory : {OUTPUT_DIR}")
print(f"Total clips      : {num_clips}")
print(f"Clip duration    : {CLIP_DURATION} sec")
print(f"Hop              : {HOP_DURATION} sec")
print(f"Sample rate      : {TARGET_SR} Hz")
print(f"Format           : Mono, 16-bit PCM")
print("=" * 60)