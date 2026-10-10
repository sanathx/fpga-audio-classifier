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
    "event"
)

EVENT_CLIPS = list(range(23, 31))

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DOORBELL EVENT LABELING")
print("=" * 60)

for clip_number in EVENT_CLIPS:

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
        print(f"ERROR: Missing {filename}")
        continue

    shutil.copy2(
        source,
        destination
    )

    print(f"Labeled: {filename}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Event clips: {len(EVENT_CLIPS)}")
print(f"Output     : {OUTPUT_DIR}")