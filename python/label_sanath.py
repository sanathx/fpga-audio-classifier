import os
import shutil

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "processed"
)

OUTPUT_BASE = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "labeled"
)

# Confirmed "Sanath" clips
SESSION_RANGES = {
    "sanath_session_01": [
        (0, 1),
        (4, 5),
        (6, 6),
        (10, 11),
        (17, 18),
    ],

    "sanath_session_02": [
        (0, 1),
        (2, 4),
        (5, 6),
        (7, 7),
        (8, 8),
        (9, 10),
        (12, 14),
    ]
}

total_copied = 0

for session, ranges in SESSION_RANGES.items():

    source_dir = os.path.join(
        BASE_DIR,
        session
    )

    output_dir = os.path.join(
        OUTPUT_BASE,
        "positive",
        session
    )

    os.makedirs(output_dir, exist_ok=True)

    copied = 0

    for start, end in ranges:

        for clip_num in range(start, end + 1):

            filename = f"sanath_{clip_num:04d}.wav"

            source = os.path.join(
                source_dir,
                filename
            )

            destination = os.path.join(
                output_dir,
                filename
            )

            if os.path.exists(source):

                shutil.copy2(
                    source,
                    destination
                )

                copied += 1
                total_copied += 1

    print(
        f"{session}: copied {copied} positive clips"
    )

print("=" * 60)
print("SANATH POSITIVE LABELING COMPLETE")
print("=" * 60)
print(f"Total positive clips: {total_copied}")
print(
    f"Output: {os.path.join(OUTPUT_BASE, 'positive')}"
)