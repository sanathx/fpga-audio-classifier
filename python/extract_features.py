from pathlib import Path
import numpy as np

# ============================================================
# Configuration
# ============================================================

N = 256
SAMPLE_RATE = 48000

project_root = Path(__file__).resolve().parent.parent

rtl_output_file = project_root / "data" / "fft_output.txt"
selected_frames_file = project_root / "data" / "selected_frames.txt"
feature_file = project_root / "data" / "features.csv"


# ============================================================
# Read selected frame indices
# ============================================================

with open(selected_frames_file, "r") as f:
    selected_frames = [
        int(line.strip())
        for line in f
        if line.strip()
    ]


# ============================================================
# Read RTL FFT output
# ============================================================

rtl_frames = []

current_frame = None

with open(rtl_output_file, "r") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        if line.startswith("# FRAME"):

            current_frame = []

            rtl_frames.append(current_frame)

        else:

            real, imag = map(
                int,
                line.split()
            )

            current_frame.append(
                complex(real, imag)
            )


# ============================================================
# Validate input
# ============================================================

if len(rtl_frames) != len(selected_frames):

    raise RuntimeError(
        f"Expected {len(selected_frames)} FFT frames "
        f"but found {len(rtl_frames)}."
    )


for i, frame in enumerate(rtl_frames):

    if len(frame) != N:

        raise RuntimeError(
            f"Frame {i} contains {len(frame)} "
            f"FFT samples instead of {N}."
        )


# ============================================================
# Frequency axis
# ============================================================

frequencies = (
    np.arange(N) *
    SAMPLE_RATE /
    N
)


# ============================================================
# Feature extraction
# ============================================================

features = []


for frame_number, fft_frame in enumerate(rtl_frames):

    fft_values = np.array(
        fft_frame,
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Magnitude spectrum
    # --------------------------------------------------------

    magnitude = np.abs(fft_values)

    # Only use positive frequencies.
    #
    # For real audio:
    #
    # X[N-k] = conjugate(X[k])
    #
    # Therefore the useful spectrum is:
    #
    # 0 ... N/2
    # --------------------------------------------------------

    positive_magnitude = magnitude[:N // 2 + 1]

    positive_frequency = frequencies[:N // 2 + 1]


    # --------------------------------------------------------
    # Total spectral energy
    # --------------------------------------------------------

    total_energy = np.sum(
        positive_magnitude ** 2
    )


    # --------------------------------------------------------
    # Frequency-band energies
    # --------------------------------------------------------

    low_mask = (
        (positive_frequency >= 0) &
        (positive_frequency < 1000)
    )

    mid_mask = (
        (positive_frequency >= 1000) &
        (positive_frequency < 3000)
    )

    high_mask = (
        (positive_frequency >= 3000) &
        (positive_frequency < 6000)
    )


    low_energy = np.sum(
        positive_magnitude[low_mask] ** 2
    )

    mid_energy = np.sum(
        positive_magnitude[mid_mask] ** 2
    )

    high_energy = np.sum(
        positive_magnitude[high_mask] ** 2
    )


    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    magnitude_sum = np.sum(
        positive_magnitude
    )

    if magnitude_sum > 0:

        spectral_centroid = np.sum(
            positive_frequency *
            positive_magnitude
        ) / magnitude_sum

    else:

        spectral_centroid = 0


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    features.append(
        [
            frame_number + 1,
            selected_frames[frame_number],
            total_energy,
            low_energy,
            mid_energy,
            high_energy,
            spectral_centroid
        ]
    )


# ============================================================
# Save CSV
# ============================================================

header = (
    "frame,"
    "audio_frame,"
    "total_energy,"
    "low_energy,"
    "mid_energy,"
    "high_energy,"
    "spectral_centroid"
)


with open(feature_file, "w") as f:

    f.write(header + "\n")

    for row in features:

        f.write(
            f"{row[0]},"
            f"{row[1]},"
            f"{row[2]:.6f},"
            f"{row[3]:.6f},"
            f"{row[4]:.6f},"
            f"{row[5]:.6f},"
            f"{row[6]:.6f}\n"
        )


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"FFT frames processed : {len(features)}"
)

print(
    f"Features per frame   : 5"
)

print(
    f"Output file          : {feature_file}"
)

print()
print(
    "Features:"
)

print(
    "  1. Total spectral energy"
)

print(
    "  2. Low-frequency energy (0-1 kHz)"
)

print(
    "  3. Mid-frequency energy (1-3 kHz)"
)

print(
    "  4. High-frequency energy (3-6 kHz)"
)

print(
    "  5. Spectral centroid"
)

print("=" * 70)