import os
import librosa

CLIP_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier\data\dataset\sanath\processed\sanath_session_03"

files = sorted(
    f for f in os.listdir(CLIP_DIR)
    if f.lower().endswith(".wav")
)

print("=" * 60)
print("SANATH SESSION 03 CLIP TIMELINE")
print("=" * 60)

print(f"Total clips: {len(files)}")
print()

for filename in files:

    filepath = os.path.join(
        CLIP_DIR,
        filename
    )

    audio, sr = librosa.load(
        filepath,
        sr=None,
        mono=True
    )

    index = int(
        filename.split("_")[1].split(".")[0]
    )

    start_time = index * 0.5
    end_time = start_time + 1.0

    print(
        f"{filename} | "
        f"{start_time:5.1f}s - "
        f"{end_time:5.1f}s | "
        f"peak={abs(audio).max():.4f}"
    )