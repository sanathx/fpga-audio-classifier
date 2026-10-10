import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# Configuration
# ==================================================

N = 256
FS = 8000

# Two frequency components
FREQUENCY_1 = 500
FREQUENCY_2 = 1500


# ==================================================
# Generate sampled signal
# ==================================================

t = np.arange(N) / FS

x = (
    0.7 * np.sin(2 * np.pi * FREQUENCY_1 * t)
    + 0.3 * np.sin(2 * np.pi * FREQUENCY_2 * t)
)


# ==================================================
# FFT
# ==================================================

X = np.fft.fft(x)

magnitude = np.abs(X)

frequencies = np.fft.fftfreq(N, 1 / FS)


# ==================================================
# Find dominant frequencies
# ==================================================

positive_magnitude = magnitude[:N // 2]
positive_frequencies = frequencies[:N // 2]

peak_indices = np.argsort(positive_magnitude)[-5:]

peak_indices = peak_indices[
    np.argsort(positive_magnitude[peak_indices])[::-1]
]


# ==================================================
# Print information
# ==================================================

print("-----------------------------------------------")
print("          Audio FFT Reference")
print("-----------------------------------------------")

print(f"FFT size      : {N}")
print(f"Sampling rate : {FS} Hz")

print("\nStrongest frequency components:")

for index in peak_indices:
    print(
        f"Bin {index:3d} : "
        f"{positive_frequencies[index]:7.2f} Hz  "
        f"Magnitude = {positive_magnitude[index]:.3f}"
    )

print("-----------------------------------------------")


# ==================================================
# Time-domain plot
# ==================================================

plt.figure()

plt.plot(t, x)

plt.title("Audio-like Input Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid()

plt.show()


# ==================================================
# Frequency-domain plot
# ==================================================

plt.figure()

plt.plot(
    positive_frequencies,
    positive_magnitude
)

plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.grid()

plt.show()