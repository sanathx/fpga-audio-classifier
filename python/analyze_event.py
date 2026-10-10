import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

CLASS_NAME = "sanath"

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    CLASS_NAME,
    "processed"
)

TARGET_SR = 16000

print("=" * 60)
print("EVENT DATASET ANALYSIS")
print("=" * 60)

print(f"Class       : {CLASS_NAME}")
print(f"Input folder: {INPUT_DIR}")
print()

# ------------------------------------------------------------
# Find clips
# ------------------------------------------------------------

files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.wav"))
)

if len(files) == 0:
    print("ERROR: No WAV files found.")
    raise SystemExit

print(f"Found {len(files)} clips.")
print()

# ------------------------------------------------------------
# Analyze every clip
# ------------------------------------------------------------

rms_values = []
peak_values = []
zcr_values = []

for index, file_path in enumerate(files):

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)[0]
    )

    rms_values.append(rms)
    peak_values.append(peak)
    zcr_values.append(zcr)

rms_values = np.array(rms_values)
peak_values = np.array(peak_values)
zcr_values = np.array(zcr_values)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 60)
print("STATISTICS")
print("=" * 60)

print(f"RMS minimum : {np.min(rms_values):.6f}")
print(f"RMS mean    : {np.mean(rms_values):.6f}")
print(f"RMS median  : {np.median(rms_values):.6f}")
print(f"RMS maximum : {np.max(rms_values):.6f}")

print()

print(f"Peak minimum : {np.min(peak_values):.6f}")
print(f"Peak mean    : {np.mean(peak_values):.6f}")
print(f"Peak maximum : {np.max(peak_values):.6f}")

print()

# ------------------------------------------------------------
# Rank clips by RMS
# ------------------------------------------------------------

sorted_indices = np.argsort(
    rms_values
)[::-1]

print("=" * 60)
print("TOP 10 HIGHEST-ENERGY CLIPS")
print("=" * 60)

for rank, idx in enumerate(sorted_indices[:10], start=1):

    print(
        f"{rank:2d}. "
        f"{os.path.basename(files[idx])} | "
        f"RMS={rms_values[idx]:.6f} | "
        f"Peak={peak_values[idx]:.6f}"
    )

# ------------------------------------------------------------
# Analysis directory
# ------------------------------------------------------------

ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    CLASS_NAME,
    "analysis"
)

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Create spectrograms for highest-energy clips
# ------------------------------------------------------------

print()
print("=" * 60)
print("CREATING SPECTROGRAMS")
print("=" * 60)

TOP_N = min(10, len(files))

for rank, idx in enumerate(sorted_indices[:TOP_N], start=1):

    audio, sr = librosa.load(
        files[idx],
        sr=TARGET_SR,
        mono=True
    )

    D = librosa.stft(
        audio,
        n_fft=512,
        hop_length=128
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
        hop_length=128,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.ylim(0, 8000)

    filename = os.path.basename(files[idx])

    plt.title(
        f"Doorbell Clip #{rank}\n{filename}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        ANALYSIS_DIR,
        f"doorbell_top_{rank:02d}.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

# ------------------------------------------------------------
# Save numerical analysis
# ------------------------------------------------------------

CSV_FILE = os.path.join(
    ANALYSIS_DIR,
    "doorbell_analysis.csv"
)

with open(CSV_FILE, "w") as f:

    f.write(
        "filename,rms,peak,zero_crossing_rate\n"
    )

    for i, file_path in enumerate(files):

        f.write(
            f"{os.path.basename(file_path)},"
            f"{rms_values[i]:.8f},"
            f"{peak_values[i]:.8f},"
            f"{zcr_values[i]:.8f}\n"
        )

# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print(f"Saved spectrograms to:")
print(ANALYSIS_DIR)

print()

print(f"Saved numerical analysis to:")
print(CSV_FILE)

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)