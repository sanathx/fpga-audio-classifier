import os
import glob
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "processed",
    "sanath_negative"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "labeled",
    "negative",
    "speech"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(
    glob.glob(
        os.path.join(SOURCE_DIR, "*.wav")
    )
)

copied = 0

for filepath in files:

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
print("SANATH SPEECH NEGATIVE LABELING")
print("=" * 60)

print(f"Source clips : {len(files)}")
print(f"Copied       : {copied}")
print(f"Output       : {OUTPUT_DIR}")

print("=" * 60)