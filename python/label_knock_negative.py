import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "knock",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "knock",
    "labeled",
    "negative",
    "session_01"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Confirmed knock regions
KNOCK_RANGES = [
    (23, 25),
    (41, 44),
    (53, 54),
]

# Safety margin:
# Exclude clips immediately around each knock
EXCLUSION_MARGIN = 2

def is_near_knock(clip_num):
    for start, end in KNOCK_RANGES:
        if (start - EXCLUSION_MARGIN) <= clip_num <= (end + EXCLUSION_MARGIN):
            return True
    return False


copied = 0

for clip_num in range(72):

    if is_near_knock(clip_num):
        continue

    filename = f"knock_{clip_num:04d}.wav"

    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Copied negative: {filename}")
        copied += 1

print("\n" + "=" * 50)
print("NEGATIVE KNOCK LABELING COMPLETE")
print("=" * 50)
print(f"Negative clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")