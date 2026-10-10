from pathlib import Path
import wave
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

N = 256
HOP = 128

# Audio activity threshold
THRESHOLD = 500

# Minimum gap between separate events
# 0.10 seconds = 100 ms
MIN_EVENT_GAP = 0.10

# Number of frames to extract around each event
FRAMES_PER_EVENT = 10


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

audio_file = project_root / "data" / "clap.wav"
input_file = project_root / "data" / "fft_input.txt"
selected_frames_file = project_root / "data" / "selected_frames.txt"


# ---------------------------------------------------------
# Read WAV
# ---------------------------------------------------------

with wave.open(str(audio_file), "rb") as wav:

    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    num_samples = wav.getnframes()

    raw_audio = wav.readframes(num_samples)


# ---------------------------------------------------------
# Stereo → mono
# ---------------------------------------------------------

audio = np.frombuffer(raw_audio, dtype=np.int16)

if channels == 2:

    audio = audio.reshape(-1, 2)

    mono = audio.mean(axis=1)

else:

    mono = audio


mono = mono.astype(np.float64)


# ---------------------------------------------------------
# Calculate frame RMS
# ---------------------------------------------------------

num_frames = 1 + (len(mono) - N) // HOP

frame_rms = np.zeros(num_frames)

for i in range(num_frames):

    start = i * HOP

    frame = mono[start:start + N]

    frame_rms[i] = np.sqrt(
        np.mean(frame ** 2)
    )


# ---------------------------------------------------------
# Detect active frames
# ---------------------------------------------------------

active = frame_rms > THRESHOLD


# ---------------------------------------------------------
# Group active frames into events
# ---------------------------------------------------------

events = []

current_start = None
current_end = None

max_gap_frames = int(
    MIN_EVENT_GAP * sample_rate / HOP
)


for i in range(num_frames):

    if active[i]:

        if current_start is None:

            current_start = i
            current_end = i

        elif i - current_end <= max_gap_frames:

            current_end = i

        else:

            events.append(
                (current_start, current_end)
            )

            current_start = i
            current_end = i


# Add final event

if current_start is not None:

    events.append(
        (current_start, current_end)
    )


# ---------------------------------------------------------
# Remove very short events
# ---------------------------------------------------------

events = [
    event
    for event in events
    if event[1] - event[0] + 1 >= 1
]


# ---------------------------------------------------------
# Select frames around each event
# ---------------------------------------------------------

selected_frames = []

for event_number, (event_start, event_end) in enumerate(events):

    # Center of event
    center = (event_start + event_end) // 2

    half = FRAMES_PER_EVENT // 2

    first_frame = max(
        0,
        center - half
    )

    last_frame = min(
        num_frames,
        first_frame + FRAMES_PER_EVENT
    )

    for frame_index in range(
        first_frame,
        last_frame
    ):

        if frame_index not in selected_frames:

            selected_frames.append(frame_index)


# ---------------------------------------------------------
# Bit reversal
# ---------------------------------------------------------

def bit_reverse(index, bits):

    return int(
        f"{index:0{bits}b}"[::-1],
        2
    )


BITS = int(np.log2(N))


# ---------------------------------------------------------
# Write RTL input
# ---------------------------------------------------------

with open(input_file, "w") as f:

    for frame_index in selected_frames:

        frame_start = frame_index * HOP

        frame = mono[
            frame_start:
            frame_start + N
        ]

        for i in range(N):

            reversed_index = bit_reverse(
                i,
                BITS
            )

            sample = int(
                np.round(
                    frame[reversed_index]
                )
            )

            f.write(
                f"{sample} 0\n"
            )


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 70)
print("FULL AUDIO EVENT DETECTION")
print("=" * 70)

print()
print(f"Sample rate       : {sample_rate} Hz")
print(f"Audio duration    : {len(mono) / sample_rate:.3f} s")
print(f"FFT size          : {N}")
print(f"Hop size          : {HOP}")
print(f"Frame duration    : {N / sample_rate * 1000:.3f} ms")
print(f"Hop duration      : {HOP / sample_rate * 1000:.3f} ms")

print()
print(f"Total frames      : {num_frames}")
print(f"Detected events   : {len(events)}")

print()
print("Detected events:")
print("-" * 70)


for i, (start_frame, end_frame) in enumerate(events):

    start_sample = start_frame * HOP
    end_sample = (
        end_frame * HOP + N - 1
    )

    start_time = start_sample / sample_rate
    end_time = end_sample / sample_rate

    print(
        f"Event {i + 1}: "
        f"{start_time:.3f} s → "
        f"{end_time:.3f} s "
        f"("
        f"{end_frame - start_frame + 1}"
        f" frames)"
    )


print()
print(f"Selected FFT frames : {len(selected_frames)}")

print(
    f"RTL input samples   : "
    f"{len(selected_frames) * N}"
)

print()
print(f"Generated: {input_file}")

print("=" * 70)
with open(selected_frames_file, "w") as f:
    for frame_index in selected_frames:
        f.write(f"{frame_index}\n")