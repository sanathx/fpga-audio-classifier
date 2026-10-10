import os
import librosa
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\sandb\Desktop\fpga-audio-classifier\data\dataset\sanath\sanath_test_02.wav"

OUTPUT_DIR = r"C:\Users\sandb\Desktop\fpga-audio-classifier\data\dataset\sanath\processed\sanath_session_03"

TARGET_SR = 16000

CLIP_DURATION = 1.0
HOP_DURATION = 0.5


# ============================================================
# LOAD AUDIO
# ============================================================

print("=" * 60)
print("SANATH SESSION 03 PREPROCESSING")
print("=" * 60)

print("\nLoading audio...")

audio, sr = librosa.load(
    INPUT_FILE,
    sr=TARGET_SR,
    mono=True
)

print(f"Target sample rate : {TARGET_SR} Hz")
print(f"Samples             : {len(audio)}")
print(f"Duration             : {len(audio) / TARGET_SR:.2f} sec")
print(f"Peak amplitude       : {abs(audio).max():.4f}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CREATE 1-SECOND CLIPS
# ============================================================

clip_samples = int(
    CLIP_DURATION * TARGET_SR
)

hop_samples = int(
    HOP_DURATION * TARGET_SR
)

count = 0
start = 0

while start + clip_samples <= len(audio):

    clip = audio[
        start:start + clip_samples
    ]

    filename = f"sanath_{count:04d}.wav"

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    sf.write(
        output_file,
        clip,
        TARGET_SR,
        subtype="PCM_16"
    )

    count += 1
    start += hop_samples


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Clips created : {count}")

print(f"Output:")
print(OUTPUT_DIR)

print("\nClip format:")
print("Duration      : 1.0 second")
print("Sample rate   : 16000 Hz")
print("Hop           : 0.5 second")
print("Format        : 16-bit PCM WAV")