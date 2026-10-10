import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "doorbell",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "doorbell",
    "labeled",
    "session_02"
)

EVENT_RANGES = [
    (47, 54),
    (65, 74),
    (80, 89),
    (94, 103)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DOORBELL SESSION 02 LABELING")
print("=" * 60)

count = 0

for start, end in EVENT_RANGES:

    print(f"\nEvent: {start:04d} -> {end:04d}")

    for clip_number in range(start, end + 1):

        filename = f"doorbell_{clip_number:04d}.wav"

        source = os.path.join(
            SOURCE_DIR,
            filename
        )

        destination = os.path.join(
            OUTPUT_DIR,
            filename
        )

        if not os.path.exists(source):
            print(f"WARNING: Missing {filename}")
            continue

        shutil.copy2(
            source,
            destination
        )

        print(f"  Labeled: {filename}")
        count += 1

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Confirmed events : {len(EVENT_RANGES)}")
print(f"Labeled clips    : {count}")
print(f"Output folder    : {OUTPUT_DIR}")
print("=" * 60)