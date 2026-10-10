import os
import numpy as np
import librosa
import soundfile as sf

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "sanath_negative.wav"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "processed",
    "sanath_negative"
)

TARGET_SR = 16000
CLIP_DURATION = 1.0
HOP_DURATION = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("SANATH NEGATIVE DATASET PREPROCESSING")
print("=" * 60)

print(f"Input  : {INPUT_FILE}")
print(f"Output : {OUTPUT_DIR}")

print("\nLoading audio...")

audio, sr = librosa.load(
    INPUT_FILE,
    sr=TARGET_SR,
    mono=True
)

print(f"Loaded sample rate : {sr} Hz")
print(f"Samples            : {len(audio)}")
print(f"Duration           : {len(audio) / sr:.2f} s")
print(f"Peak amplitude     : {np.max(np.abs(audio)):.4f}")

clip_length = int(CLIP_DURATION * TARGET_SR)
hop_length = int(HOP_DURATION * TARGET_SR)

print("\nCreating 1-second clips...")

count = 0

for start in range(
    0,
    len(audio) - clip_length + 1,
    hop_length
):

    clip = audio[start:start + clip_length]

    filename = f"sanath_negative_{count:04d}.wav"

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    sf.write(
        output_file,
        clip,
        TARGET_SR,
        subtype="PCM_16"
    )

    count += 1

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Generated clips : {count}")
print(f"Clip duration   : {CLIP_DURATION} s")
print(f"Hop duration    : {HOP_DURATION} s")
print(f"Sample rate     : {TARGET_SR} Hz")
print(f"Output folder   : {OUTPUT_DIR}")

print("=" * 60)