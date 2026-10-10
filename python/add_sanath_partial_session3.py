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

PARTIAL_DIR = (
    ROOT
    / "data"
    / "dataset"
    / "sanath"
    / "labeled"
    / "positive"
    / "sanath_session_03_partial"
)


# Verified partial occurrences
partial_indices = [
    3,
    11,
    25,
    45,
]


PARTIAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 60)
print("ADDING SANATH SESSION 03 PARTIAL POSITIVES")
print("=" * 60)


for index in partial_indices:

    source = (
        PROCESSED_DIR
        / f"sanath_{index:04d}.wav"
    )

    destination = (
        PARTIAL_DIR
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
        f"Partial positive: {source.name}"
    )


print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

print(
    f"Partial positives added: "
    f"{len(list(PARTIAL_DIR.glob('*.wav')))}"
)

print(
    f"Output: {PARTIAL_DIR}"
)