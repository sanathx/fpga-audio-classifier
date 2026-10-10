from pathlib import Path
import wave
import numpy as np

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent
audio_file = project_root / "data" / "clap.wav"

# ---------------------------------------------------------
# Read WAV file
# ---------------------------------------------------------

with wave.open(str(audio_file), "rb") as wav:

    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    sample_rate = wav.getframerate()
    num_samples = wav.getnframes()

    raw_audio = wav.readframes(num_samples)

# ---------------------------------------------------------
# Convert raw bytes to integer samples
# ---------------------------------------------------------

audio = np.frombuffer(raw_audio, dtype=np.int16)

# Reshape stereo audio
if channels == 2:
    audio = audio.reshape(-1, 2)

    # Convert stereo → mono
    mono = audio.mean(axis=1)

else:
    mono = audio

mono = mono.astype(np.int16)

# ---------------------------------------------------------
# Display information
# ---------------------------------------------------------

print("=" * 60)
print("AUDIO PREPROCESSING")
print("=" * 60)

print(f"Sample rate       : {sample_rate} Hz")
print(f"Channels          : {channels}")
print(f"Original samples  : {num_samples}")
print(f"Mono samples      : {len(mono)}")
print(f"Duration          : {len(mono) / sample_rate:.3f} seconds")

print()
print("Amplitude:")
print(f"Minimum           : {mono.min()}")
print(f"Maximum           : {mono.max()}")
print(f"Mean              : {mono.mean():.2f}")
print(f"RMS               : {np.sqrt(np.mean(mono.astype(float)**2)):.2f}")

print()
print("First 20 mono samples:")

for i in range(20):
    print(f"x[{i:2d}] = {mono[i]}")

print("=" * 60)
# ---------------------------------------------------------
# Find active audio region
# ---------------------------------------------------------

threshold = 500

active_indices = np.where(np.abs(mono) > threshold)[0]

if len(active_indices) == 0:

    print("\nNo samples exceeded the activity threshold.")

else:

    first_active = active_indices[0]
    last_active = active_indices[-1]

    print()
    print("Active audio region:")
    print(f"First active sample : {first_active}")
    print(f"Last active sample  : {last_active}")

    print(
        f"First active time   : "
        f"{first_active / sample_rate:.3f} seconds"
    )

    print(
        f"Last active time    : "
        f"{last_active / sample_rate:.3f} seconds"
    )