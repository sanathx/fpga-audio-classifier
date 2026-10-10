import os
import shutil

# ============================================================
# SANATH SESSION 04 LABELING
# ============================================================

BASE_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "processed",
    "sanath_session_04"
)

POSITIVE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "labeled",
    "positive",
    "sanath_session_04"
)

PARTIAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "labeled",
    "positive",
    "sanath_session_04_partial"
)

NEGATIVE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "sanath",
    "labeled",
    "negative",
    "speech_session_04"
)


# ============================================================
# VERIFIED LABELS
# ============================================================

CLEAN_POSITIVE = [
    2,
    7,
    11,
    15,
    16,
    19,
    20,
    24,
    31,
    32,
    35,
    39,
    43,
    48,
    49,
    51,
    57,
    61
]

PARTIAL_POSITIVE = [
    3,
    6,
    8,
    10,
    18,
    25,
    34,
    36,
    40,
    50,
    58,
    60
]


# ============================================================
# FIND ALL CLIPS
# ============================================================

all_files = sorted(
    [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".wav")
    ]
)

if not all_files:
    print("ERROR: No WAV clips found.")
    raise SystemExit(1)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(POSITIVE_DIR, exist_ok=True)
os.makedirs(PARTIAL_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)


# ============================================================
# LABEL CLIPS
# ============================================================

clean_set = set(CLEAN_POSITIVE)
partial_set = set(PARTIAL_POSITIVE)

clean_count = 0
partial_count = 0
negative_count = 0

for filename in all_files:

    # Extract clip number
    # Example: sanath_0007.wav -> 7
    number = int(
        filename.replace("sanath_", "").replace(".wav", "")
    )

    source_path = os.path.join(SOURCE_DIR, filename)

    # --------------------------------------------------------
    # CLEAN POSITIVE
    # --------------------------------------------------------

    if number in clean_set:

        destination = os.path.join(POSITIVE_DIR, filename)

        shutil.copy2(source_path, destination)

        print(f"CLEAN    : {filename}")
        clean_count += 1

    # --------------------------------------------------------
    # PARTIAL POSITIVE
    # --------------------------------------------------------

    elif number in partial_set:

        destination = os.path.join(PARTIAL_DIR, filename)

        shutil.copy2(source_path, destination)

        print(f"PARTIAL  : {filename}")
        partial_count += 1

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    else:

        destination = os.path.join(NEGATIVE_DIR, filename)

        shutil.copy2(source_path, destination)

        print(f"NEGATIVE : {filename}")
        negative_count += 1


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 60)
print("SESSION 04 LABELING COMPLETE")
print("=" * 60)

print(f"Clean positives : {clean_count}")
print(f"Partial positives: {partial_count}")
print(f"Negatives       : {negative_count}")
print(f"Total            : {clean_count + partial_count + negative_count}")

print()
print("Clean positives:")
print(POSITIVE_DIR)

print()
print("Partial positives:")
print(PARTIAL_DIR)

print()
print("Negatives:")
print(NEGATIVE_DIR)

print("=" * 60)