from pathlib import Path
import wave
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N = 256
HOP = 128
THRESHOLD = 500

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
audio_file = project_root / "data" / "clap.wav"


# ---------------------------------------------------------
# Read WAV
# ---------------------------------------------------------

with wave.open(str(audio_file), "rb") as wav:

    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    num_samples = wav.getnframes()

    raw_audio = wav.readframes(num_samples)


# ---------------------------------------------------------
# Convert to mono
# ---------------------------------------------------------

audio = np.frombuffer(raw_audio, dtype=np.int16)

if channels == 2:
    audio = audio.reshape(-1, 2)
    mono = audio.mean(axis=1)
else:
    mono = audio

mono = mono.astype(np.int16)


# ---------------------------------------------------------
# Find first active sample
# ---------------------------------------------------------

active_indices = np.where(np.abs(mono) > THRESHOLD)[0]

if len(active_indices) == 0:
    raise RuntimeError("No active audio found.")

start = active_indices[0]


# ---------------------------------------------------------
# Analyze first 10 frames
# ---------------------------------------------------------

print("=" * 70)
print("AUDIO SPECTRUM ANALYSIS")
print("=" * 70)

print()
print(f"Sample rate : {sample_rate} Hz")
print(f"FFT size    : {N}")
print(f"Frequency resolution : {sample_rate / N:.2f} Hz/bin")
print(f"Frames      : 10")

print()


for frame_number in range(10):

    frame_start = start + frame_number * HOP

    frame = mono[
        frame_start:
        frame_start + N
    ]

    # ---------------------------------------------
    # FFT
    # ---------------------------------------------

    fft = np.fft.fft(frame)

    magnitude = np.abs(fft)

    # ---------------------------------------------
    # Only positive frequencies
    # ---------------------------------------------

    positive_magnitude = magnitude[:N // 2]

    frequencies = np.arange(N // 2) * sample_rate / N

    # ---------------------------------------------
    # Find strongest frequencies
    # ---------------------------------------------

    largest_bins = np.argsort(
        positive_magnitude
    )[-5:][::-1]

    print("-" * 70)

    print(f"Frame {frame_number + 1}")

    print(
        f"Time: "
        f"{frame_start / sample_rate:.6f} s"
    )

    print()

    print("Strongest frequency components:")

    for k in largest_bins:

        print(
            f"  Bin {k:3d} | "
            f"{frequencies[k]:8.1f} Hz | "
            f"Magnitude = "
            f"{positive_magnitude[k]:10.2f}"
        )


print()
print("=" * 70)