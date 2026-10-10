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
    "negative",
    "session_01"
)

PHONE_RING_START = 12
PHONE_RING_END = 51

EXCLUSION_MARGIN = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_near_phone_ring(clip_num):
    return (
        PHONE_RING_START - EXCLUSION_MARGIN
        <= clip_num
        <= PHONE_RING_END + EXCLUSION_MARGIN
    )


copied = 0

# Session 01 has 61 clips: 0000–0060
for clip_num in range(61):

    if is_near_phone_ring(clip_num):
        continue

    filename = f"phone_ring_{clip_num:04d}.wav"

    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Copied negative: {filename}")
        copied += 1
    else:
        print(f"WARNING: Missing {filename}")


print("\n" + "=" * 50)
print("PHONE RING SESSION 01 NEGATIVE LABELING COMPLETE")
print("=" * 50)
print(f"Negative clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")