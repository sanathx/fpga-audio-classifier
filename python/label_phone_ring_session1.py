import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "phone_ring",
    "processed",
    "phone_ring_session_01"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "phone_ring",
    "labeled",
    "session_01"
)

START_CLIP = 12
END_CLIP = 51

os.makedirs(OUTPUT_DIR, exist_ok=True)

copied = 0

for i in range(START_CLIP, END_CLIP + 1):

    filename = f"phone_ring_{i:04d}.wav"

    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Copied: {filename}")
        copied += 1
    else:
        print(f"WARNING: Missing {filename}")

print("\n" + "=" * 50)
print("PHONE RING SESSION 01 LABELING COMPLETE")
print("=" * 50)
print(f"Positive clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")