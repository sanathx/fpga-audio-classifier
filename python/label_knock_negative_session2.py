import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "knock",
    "processed",
    "knock_session_02"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "knock",
    "labeled",
    "negative",
    "session_02"
)

KNOCK_RANGES = [
    (19, 19),
    (24, 24),
    (28, 29),
    (33, 34),
    (37, 38),
]

EXCLUSION_MARGIN = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_near_knock(clip_num):
    for start, end in KNOCK_RANGES:
        if (start - EXCLUSION_MARGIN) <= clip_num <= (end + EXCLUSION_MARGIN):
            return True
    return False


copied = 0

for clip_num in range(50):

    if is_near_knock(clip_num):
        continue

    filename = f"knock_{clip_num:04d}.wav"

    source = os.path.join(SOURCE_DIR, filename)
    destination = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Copied negative: {filename}")
        copied += 1
    else:
        print(f"WARNING: Missing {filename}")


print("\n" + "=" * 50)
print("SESSION 02 NEGATIVE LABELING COMPLETE")
print("=" * 50)
print(f"Negative clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")