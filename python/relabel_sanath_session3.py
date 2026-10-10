from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    ROOT
    / "data"
    / "dataset"
    / "sanath"
    / "processed"
    / "sanath_session_03"
)

POSITIVE_DIR = (
    ROOT
    / "data"
    / "dataset"
    / "sanath"
    / "labeled"
    / "positive"
    / "sanath_session_03"
)

NEGATIVE_DIR = (
    ROOT
    / "data"
    / "dataset"
    / "sanath"
    / "labeled"
    / "negative"
    / "speech_session_03"
)


# ============================================================
# VERIFIED POSITIVES
# ============================================================

positive_indices = [
    2,
    12,
    26,
    46,
    53,
]


# ============================================================
# REBUILD POSITIVE DIRECTORY
# ============================================================

if POSITIVE_DIR.exists():

    for file in POSITIVE_DIR.glob("*.wav"):
        file.unlink()

else:

    POSITIVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


print("=" * 60)
print("REBUILDING SANATH SESSION 03 POSITIVES")
print("=" * 60)


for index in positive_indices:

    source = (
        PROCESSED_DIR
        / f"sanath_{index:04d}.wav"
    )

    destination = (
        POSITIVE_DIR
        / source.name
    )

    if not source.exists():

        print(
            f"ERROR: Missing {source}"
        )

        continue

    shutil.copy2(
        source,
        destination
    )

    print(
        f"Positive: {source.name}"
    )


# ============================================================
# REBUILD NEGATIVES
# ============================================================

if NEGATIVE_DIR.exists():

    for file in NEGATIVE_DIR.glob("*.wav"):
        file.unlink()

else:

    NEGATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# Exclude ±2 clips around each verified positive.
excluded = set()

for index in positive_indices:

    for offset in range(-2, 3):

        candidate = index + offset

        if candidate >= 0:

            excluded.add(candidate)


print("\nExcluded from negatives:")

print(
    sorted(excluded)
)


# ============================================================
# COPY NEGATIVES
# ============================================================

negative_count = 0

for wav in sorted(
    PROCESSED_DIR.glob("*.wav")
):

    try:

        index = int(
            wav.stem.split("_")[-1]
        )

    except ValueError:

        continue


    if index in excluded:

        continue


    destination = (
        NEGATIVE_DIR
        / wav.name
    )

    shutil.copy2(
        wav,
        destination
    )

    negative_count += 1


# ============================================================
# SUMMARY
# ============================================================

positive_count = len(
    list(
        POSITIVE_DIR.glob("*.wav")
    )
)

negative_count = len(
    list(
        NEGATIVE_DIR.glob("*.wav")
    )
)


print("\n" + "=" * 60)
print("SESSION 03 LABELING COMPLETE")
print("=" * 60)

print(
    f"Positive clips : {positive_count}"
)

print(
    f"Negative clips : {negative_count}"
)

print(
    f"Total clips    : "
    f"{positive_count + negative_count}"
)

print(
    f"\nPositive output:"
    f"\n{POSITIVE_DIR}"
)

print(
    f"\nNegative output:"
    f"\n{NEGATIVE_DIR}"
)