import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "phone_ring",
    "processed",
    "phone_ring_session_02"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "phone_ring",
    "labeled",
    "negative",
    "session_02"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_CLIPS = 45

POSITIVE_START = 4
POSITIVE_END = 44

SAFETY_MARGIN = 2

excluded_start = max(0, POSITIVE_START - SAFETY_MARGIN)
excluded_end = min(TOTAL_CLIPS - 1, POSITIVE_END + SAFETY_MARGIN)

copied = 0

for clip_num in range(TOTAL_CLIPS):

    # Skip positive clips and transition/safety-margin clips
    if excluded_start <= clip_num <= excluded_end:
        continue

    filename = f"phone_ring_{clip_num:04d}.wav"

    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        copied += 1

print("=" * 60)
print("PHONE RING SESSION 02 NEGATIVE LABELING")
print("=" * 60)

print(f"Total clips       : {TOTAL_CLIPS}")
print(f"Positive range    : {POSITIVE_START:04d}–{POSITIVE_END:04d}")
print(f"Excluded margin   : {excluded_start:04d}–{excluded_end:04d}")
print(f"Negative clips    : {copied}")
print(f"Output directory  : {OUTPUT_DIR}")