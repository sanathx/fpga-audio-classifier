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
    "session_02"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_START = 4
EVENT_END = 44

copied = 0

for clip_num in range(EVENT_START, EVENT_END + 1):

    filename = f"phone_ring_{clip_num:04d}.wav"
    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        copied += 1

print("=" * 50)
print("PHONE RING SESSION 02 POSITIVE LABELING")
print("=" * 50)
print(f"Positive clips: {copied}")
print(f"Range: {EVENT_START:04d}–{EVENT_END:04d}")
print(f"Output: {OUTPUT_DIR}")