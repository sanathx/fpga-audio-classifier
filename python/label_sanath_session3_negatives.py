from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "dataset" / "sanath"
    / "processed" / "sanath_session_03"
)

DEST = (
    ROOT / "data" / "dataset" / "sanath"
    / "labeled" / "negative" / "speech_session_03"
)

# Positive regions from Session 03.
POSITIVE_RANGES = [
    (2, 3),
    (11, 12),
    (25, 26),
    (33, 34),
    (37, 38),
    (45, 46),
    (52, 53),
    (59, 60),
]

# Exclude two clips on either side of each positive region.
EXCLUDED = set()

for start, end in POSITIVE_RANGES:
    for i in range(start - 2, end + 3):
        if 0 <= i < 63:
            EXCLUDED.add(i)

DEST.mkdir(parents=True, exist_ok=True)

copied = 0

for i in range(63):
    if i in EXCLUDED:
        continue

    src = SOURCE / f"sanath_{i:04d}.wav"
    dst = DEST / src.name

    if src.exists():
        shutil.copy2(src, dst)
        copied += 1

print("=" * 55)
print("SANATH SESSION 03 NEGATIVE LABELING")
print("=" * 55)
print(f"Total clips       : 63")
print(f"Excluded clips    : {len(EXCLUDED)}")
print(f"Negative clips    : {copied}")
print(f"Output            : {DEST}")