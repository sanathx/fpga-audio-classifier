import os
import glob
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "background",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "labeled",
    "negative",
    "background"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(
    glob.glob(
        os.path.join(SOURCE_DIR, "*.wav")
    )
)

# Use up to 100 background clips for the first dataset.
MAX_NEGATIVES = 100

copied = 0

for filepath in files:

    if copied >= MAX_NEGATIVES:
        break

    filename = os.path.basename(filepath)

    destination = os.path.join(
        OUTPUT_DIR,
        filename
    )

    shutil.copy2(
        filepath,
        destination
    )

    copied += 1

print("=" * 60)
print("SANATH BACKGROUND NEGATIVE DATASET")
print("=" * 60)
print(f"Background clips available : {len(files)}")
print(f"Negative clips copied      : {copied}")
print(f"Output                     : {OUTPUT_DIR}")