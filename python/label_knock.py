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
    "session_01"
)

# Confirmed knock events
EVENT_RANGES = [
    (23, 25),
    (41, 44),
    (53, 54),
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
print("KNOCK LABELING COMPLETE")
print("=" * 50)
print(f"Event clips copied: {copied}")
print(f"Output: {OUTPUT_DIR}")