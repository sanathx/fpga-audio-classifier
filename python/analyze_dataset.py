import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "background",
    "processed"
)

TARGET_SR = 16000

# Number of clips to inspect visually
NUM_SPECTROGRAMS = 6


print("=" * 60)
print("BACKGROUND DATASET ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Find files
# ------------------------------------------------------------

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.wav"))
)

if len(files) == 0:
    print("ERROR: No WAV files found.")
    print(f"Folder checked: {INPUT_DIR}")
    raise SystemExit

print(f"Input folder : {INPUT_DIR}")
print(f"WAV files    : {len(files)}")
print()


# ------------------------------------------------------------
# Analyze every clip
# ------------------------------------------------------------

rms_values = []
peak_values = []
zero_crossing_values = []
durations = []

print("Analyzing clips...")

for index, file_path in enumerate(files):

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # RMS energy
    rms = np.sqrt(np.mean(audio ** 2))

    # Peak amplitude
    peak = np.max(np.abs(audio))

    # Zero crossing rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)[0]
    )

    duration = len(audio) / sr

    rms_values.append(rms)
    peak_values.append(peak)
    zero_crossing_values.append(zcr)
    durations.append(duration)

    if (index + 1) % 100 == 0:
        print(f"  Analyzed {index + 1}/{len(files)}")


rms_values = np.array(rms_values)
peak_values = np.array(peak_values)
zero_crossing_values = np.array(zero_crossing_values)
durations = np.array(durations)


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

print()
print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Number of clips       : {len(files)}")
print(f"Total audio duration  : {np.sum(durations):.2f} s")
print(f"Average clip duration : {np.mean(durations):.3f} s")
print()

print("RMS amplitude")
print(f"  Minimum : {np.min(rms_values):.6f}")
print(f"  Mean    : {np.mean(rms_values):.6f}")
print(f"  Median  : {np.median(rms_values):.6f}")
print(f"  Maximum : {np.max(rms_values):.6f}")
print()

print("Peak amplitude")
print(f"  Minimum : {np.min(peak_values):.6f}")
print(f"  Mean    : {np.mean(peak_values):.6f}")
print(f"  Median  : {np.median(peak_values):.6f}")
print(f"  Maximum : {np.max(peak_values):.6f}")
print()

print("Zero Crossing Rate")
print(f"  Minimum : {np.min(zero_crossing_values):.6f}")
print(f"  Mean    : {np.mean(zero_crossing_values):.6f}")
print(f"  Median  : {np.median(zero_crossing_values):.6f}")
print(f"  Maximum : {np.max(zero_crossing_values):.6f}")


# ------------------------------------------------------------
# Detect very quiet clips
# ------------------------------------------------------------

print()
print("=" * 60)
print("QUIET CLIP ANALYSIS")
print("=" * 60)

# This threshold is intentionally conservative.
QUIET_THRESHOLD = 0.005

quiet_indices = np.where(
    rms_values < QUIET_THRESHOLD
)[0]

quiet_percentage = (
    len(quiet_indices) / len(files)
) * 100

print(
    f"Clips below RMS {QUIET_THRESHOLD}: "
    f"{len(quiet_indices)} / {len(files)} "
    f"({quiet_percentage:.2f}%)"
)


# ------------------------------------------------------------
# Identify representative clips
# ------------------------------------------------------------

print()
print("=" * 60)
print("REPRESENTATIVE CLIPS")
print("=" * 60)

# Lowest energy
lowest_idx = np.argmin(rms_values)

# Median-energy clip
median_idx = np.argsort(rms_values)[len(rms_values) // 2]

# Highest energy
highest_idx = np.argmax(rms_values)

representative_indices = [
    lowest_idx,
    median_idx,
    highest_idx
]

print(
    f"Lowest energy : {os.path.basename(files[lowest_idx])}"
)
print(
    f"Median energy : {os.path.basename(files[median_idx])}"
)
print(
    f"Highest energy: {os.path.basename(files[highest_idx])}"
)


# ------------------------------------------------------------
# Print a few random-ish evenly spaced clips
# ------------------------------------------------------------

print()
print("Additional clips:")

extra_indices = np.linspace(
    0,
    len(files) - 1,
    NUM_SPECTROGRAMS,
    dtype=int
)

for idx in extra_indices:
    print(
        f"  {os.path.basename(files[idx])}"
        f" | RMS={rms_values[idx]:.6f}"
        f" | Peak={peak_values[idx]:.6f}"
    )


# ------------------------------------------------------------
# Create spectrogram plots
# ------------------------------------------------------------

print()
print("=" * 60)
print("CREATING SPECTROGRAMS")
print("=" * 60)

PLOT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "background",
    "analysis"
)

os.makedirs(PLOT_DIR, exist_ok=True)


for plot_number, idx in enumerate(extra_indices):

    audio, sr = librosa.load(
        files[idx],
        sr=TARGET_SR,
        mono=True
    )

    # STFT
    D = librosa.stft(
        audio,
        n_fft=512,
        hop_length=256
    )

    magnitude = np.abs(D)

    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    plt.figure(figsize=(10, 5))

    librosa.display.specshow(
        magnitude_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.ylim(0, 8000)

    filename = os.path.basename(files[idx])

    plt.title(
        f"Background Spectrogram\n{filename}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        PLOT_DIR,
        f"spectrogram_{plot_number + 1}.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()


print(
    f"Saved spectrograms to:\n{PLOT_DIR}"
)


# ------------------------------------------------------------
# Save numerical analysis
# ------------------------------------------------------------

analysis_file = os.path.join(
    PLOT_DIR,
    "background_analysis.csv"
)

with open(analysis_file, "w") as f:

    f.write(
        "filename,rms,peak,zero_crossing_rate,duration\n"
    )

    for i, file_path in enumerate(files):

        f.write(
            f"{os.path.basename(file_path)},"
            f"{rms_values[i]:.8f},"
            f"{peak_values[i]:.8f},"
            f"{zero_crossing_values[i]:.8f},"
            f"{durations[i]:.4f}\n"
        )


print(
    f"Saved numerical analysis to:\n{analysis_file}"
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"Clips analyzed      : {len(files)}")
print(f"Quiet clips         : {len(quiet_indices)}")
print(f"Quiet percentage    : {quiet_percentage:.2f}%")
print(f"Analysis directory  : {PLOT_DIR}")

print()
print("Next step:")
print("Inspect the generated spectrogram PNG files.")
print("=" * 60)