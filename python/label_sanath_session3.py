
from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\sandb\Desktop\fpga-audio-classifier")

SOURCE = (
    ROOT / "data" / "dataset" / "sanath"
    / "processed" / "sanath_session_03"
)

DEST = (
    ROOT / "data" / "dataset" / "sanath"
    / "labeled" / "positive" / "sanath_session_03"
)

# Candidate clips based on approximate keyword timestamps.
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

DEST.mkdir(parents=True, exist_ok=True)

# Remove previous session-03 labels before rebuilding.
for old_file in DEST.glob("*.wav"):
    old_file.unlink()

copied = 0

for start, end in POSITIVE_RANGES:
    for index in range(start, end + 1):
        filename = f"sanath_{index:04d}.wav"
        src = SOURCE / filename
        dst = DEST / filename

        if not src.exists():
            raise FileNotFoundError(src)

        shutil.copy2(src, dst)
        print(f"Copied: {filename}")
        copied += 1

print("\nSANATH SESSION 03 LABELING COMPLETE")
print(f"Positive candidate clips: {copied}")
print(f"Output: {DEST}")
