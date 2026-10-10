from pathlib import Path
import wave
import numpy as np

# ============================================================
# Configuration
# ============================================================

N = 256
HOP = 128
THRESHOLD = 500
MIN_EVENT_GAP = 0.10
FRAMES_PER_EVENT = 10

project_root = Path(__file__).resolve().parent.parent

audio_file = project_root / "data" / "clap.wav"
rtl_output_file = project_root / "data" / "fft_output.txt"


# ============================================================
# Load audio
# ============================================================

with wave.open(str(audio_file), "rb") as wav:

    channels = wav.getnchannels()
    sample_rate = wav.getframerate()
    num_samples = wav.getnframes()

    raw_audio = wav.readframes(num_samples)


audio = np.frombuffer(raw_audio, dtype=np.int16)

if channels == 2:
    audio = audio.reshape(-1, 2)
    mono = audio.mean(axis=1)
else:
    mono = audio

mono = mono.astype(np.float64)


# ============================================================
# Detect active frames
# ============================================================

num_frames = 1 + (len(mono) - N) // HOP

frame_rms = np.zeros(num_frames)

for i in range(num_frames):

    start = i * HOP

    frame = mono[start:start + N]

    frame_rms[i] = np.sqrt(np.mean(frame ** 2))


active = frame_rms > THRESHOLD


# ============================================================
# Group active frames into events
# ============================================================

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


if current_start is not None:

    events.append(
        (current_start, current_end)
    )


# ============================================================
# Select the same 30 frames used by generate_fft_input.py
# ============================================================

selected_frames = []


for event_start, event_end in events:

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

            selected_frames.append(
                frame_index
            )


print()
print("=" * 70)
print("30-FRAME FFT VALIDATION")
print("=" * 70)

print(f"Audio sample rate : {sample_rate} Hz")
print(f"FFT size          : {N}")
print(f"Total audio frames: {num_frames}")
print(f"Detected events   : {len(events)}")
print(f"Selected frames   : {len(selected_frames)}")


# ============================================================
# Read RTL output
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

            rtl_frames.append(
                current_frame
            )

        else:

            real, imag = map(
                int,
                line.split()
            )

            current_frame.append(
                complex(real, imag)
            )


print(f"RTL frames        : {len(rtl_frames)}")


# ============================================================
# Basic validation
# ============================================================

if len(rtl_frames) != len(selected_frames):

    print()
    print("ERROR:")
    print(
        f"Expected {len(selected_frames)} RTL frames "
        f"but found {len(rtl_frames)}."
    )

    raise SystemExit


for i, frame in enumerate(rtl_frames):

    if len(frame) != N:

        print()
        print(
            f"ERROR: RTL frame {i} contains "
            f"{len(frame)} samples instead of {N}."
        )

        raise SystemExit


# ============================================================
# Compare FFTs
# ============================================================

all_errors = []

frame_results = []


for rtl_index, frame_index in enumerate(
    selected_frames
):

    # --------------------------------------------------------
    # Extract exactly the same frame
    # --------------------------------------------------------

    frame_start = frame_index * HOP

    frame = mono[
        frame_start:frame_start + N
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The RTL receives bit-reversed input.
    # NumPy receives the original natural-order frame.
    #
    # Therefore we simply use np.fft.fft(frame)
    # as the mathematical reference.
    # --------------------------------------------------------

    reference = np.fft.fft(frame)

    rtl = np.array(
        rtl_frames[rtl_index],
        dtype=np.complex128
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = np.abs(
        reference - rtl
    )

    max_error = np.max(error)

    mean_error = np.mean(error)

    rms_error = np.sqrt(
        np.mean(error ** 2)
    )

    all_errors.extend(error)

    # --------------------------------------------------------
    # Peak magnitude
    # --------------------------------------------------------

    reference_magnitude = np.abs(reference)

    rtl_magnitude = np.abs(rtl)

    ref_peak_bin = np.argmax(
        reference_magnitude
    )

    rtl_peak_bin = np.argmax(
        rtl_magnitude
    )

    ref_peak_mag = (
        reference_magnitude[ref_peak_bin]
    )

    rtl_peak_mag = (
        rtl_magnitude[rtl_peak_bin]
    )

    # --------------------------------------------------------
    # Conjugate-equivalent peak
    # --------------------------------------------------------

    conjugate_bin = (
        N - ref_peak_bin
    ) % N

    magnitude_at_ref_bin = (
        rtl_magnitude[ref_peak_bin]
    )

    magnitude_at_conjugate = (
        rtl_magnitude[conjugate_bin]
    )

    best_rtl_peak = max(
        magnitude_at_ref_bin,
        magnitude_at_conjugate
    )

    peak_error_percent = (
        abs(best_rtl_peak - ref_peak_mag)
        / ref_peak_mag
        * 100
        if ref_peak_mag != 0
        else 0
    )

    frame_results.append(
        (
            rtl_index + 1,
            frame_index,
            max_error,
            mean_error,
            rms_error,
            ref_peak_bin,
            rtl_peak_bin,
            ref_peak_mag,
            best_rtl_peak,
            peak_error_percent
        )
    )


# ============================================================
# Print per-frame results
# ============================================================

print()
print("-" * 120)

print(
    f"{'Frame':>5} "
    f"{'AudioIdx':>8} "
    f"{'MaxErr':>10} "
    f"{'MeanErr':>10} "
    f"{'RMSE':>10} "
    f"{'RefPeak':>8} "
    f"{'RTLPeak':>8} "
    f"{'PeakErr%':>10}"
)

print("-" * 120)


for result in frame_results:

    (
        frame_number,
        audio_index,
        max_error,
        mean_error,
        rms_error,
        ref_peak_bin,
        rtl_peak_bin,
        ref_peak_mag,
        best_rtl_peak,
        peak_error_percent
    ) = result

    print(
        f"{frame_number:5d} "
        f"{audio_index:8d} "
        f"{max_error:10.2f} "
        f"{mean_error:10.2f} "
        f"{rms_error:10.2f} "
        f"{ref_peak_bin:8d} "
        f"{rtl_peak_bin:8d} "
        f"{peak_error_percent:10.3f}"
    )


# ============================================================
# Overall statistics
# ============================================================

all_errors = np.array(all_errors)

overall_max = np.max(all_errors)

overall_mean = np.mean(all_errors)

overall_rms = np.sqrt(
    np.mean(all_errors ** 2)
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("OVERALL VALIDATION")
print("=" * 70)

print(
    f"Frames compared       : {len(frame_results)}"
)

print(
    f"FFT outputs compared  : "
    f"{len(frame_results) * N}"
)

print(
    f"Maximum absolute error: "
    f"{overall_max:.2f}"
)

print(
    f"Mean absolute error   : "
    f"{overall_mean:.2f}"
)

print(
    f"RMS error             : "
    f"{overall_rms:.2f}"
)

print("=" * 70)

print()
print("Validation complete.")