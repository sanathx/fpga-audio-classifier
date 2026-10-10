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
    "session_02"
)

EVENT_RANGES = [
    (19, 19),
    (24, 24),
    (28, 29),
    (33, 34),
    (37, 38),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

copied = 0

for start, end in EVENT_RANGES:
    for i in range(start, end + 1):

        filename = f"knock_{i:04d}.wav"

        source = os.path.join(SOURCE_DIR, filename)
        destination = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(source):
            shutil.copy2(source, destination)
            print(f"Copied: {filename}")
            copied += 1
        else:
            print(f"WARNING: Missing {filename}")

print("\n" + "=" * 50)
print("KNOCK SESSION 02 LABELING COMPLETE")
print("=" * 50)
print(f"Positive clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")