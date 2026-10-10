import os
import glob
import numpy as np
import librosa

PROJECT_ROOT = r"C:\Users\sandb\Desktop\fpga-audio-classifier"

SESSIONS = [
    "sanath_session_01",
    "sanath_session_02"
]

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "sanath",
    "processed"
)

for session in SESSIONS:

    folder = os.path.join(BASE_DIR, session)

    files = sorted(
        glob.glob(os.path.join(folder, "*.wav"))
    )

    print("\n" + "=" * 65)
    print(f"SANATH ANALYSIS — {session}")
    print("=" * 65)

    print(f"Clips found: {len(files)}")

    results = []

    for filepath in files:

        filename = os.path.basename(filepath)

        clip_num = int(
            filename.replace("sanath_", "").replace(".wav", "")
        )

        audio, sr = librosa.load(
            filepath,
            sr=None,
            mono=True
        )

        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        results.append(
            (clip_num, rms, peak)
        )

    results.sort()

    print("\nTOP ENERGY CLIPS")
    print("-" * 65)

    top = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for rank, (clip, rms, peak) in enumerate(top, 1):

        print(
            f"{rank:2d}. "
            f"sanath_{clip:04d}.wav | "
            f"Time={clip * 0.5:5.1f}s | "
            f"RMS={rms:.6f} | "
            f"Peak={peak:.6f}"
        )

    print("\nTIMELINE")
    print("-" * 65)

    for clip, rms, peak in results:

        print(
            f"{clip:04d} | "
            f"{clip * 0.5:5.1f}s | "
            f"RMS={rms:.6f} | "
            f"Peak={peak:.6f}"
        )