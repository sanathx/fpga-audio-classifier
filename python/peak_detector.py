import numpy as np


# ============================================================
# Configuration
# ============================================================

N = 256
FS = 8000


# ============================================================
# Generate test signal
# ============================================================

def generate_test_signal():

    t = np.arange(N) / FS

    x = (
        0.7 * np.sin(2 * np.pi * 500 * t)
        +
        0.3 * np.sin(2 * np.pi * 1500 * t)
    )

    return x


# ============================================================
# FFT
# ============================================================

def calculate_fft(x):

    return np.fft.fft(x)


# ============================================================
# Calculate magnitude squared
# ============================================================

def calculate_power_spectrum(X):

    real = X.real
    imag = X.imag

    power = real * real + imag * imag

    return power


# ============================================================
# Find dominant frequency
# ============================================================

def find_dominant_frequency(power):

    # Only inspect positive-frequency half
    positive_power = power[:N // 2]

    # Ignore DC bin
    positive_power[0] = 0

    peak_bin = np.argmax(positive_power)

    peak_frequency = peak_bin * FS / N

    return peak_bin, peak_frequency


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("-----------------------------------------------")
    print("       FFT Peak Detector")
    print("-----------------------------------------------")

    print(f"FFT size      : {N}")
    print(f"Sampling rate : {FS} Hz")

    # Generate signal
    x = generate_test_signal()

    # FFT
    X = calculate_fft(x)

    # Power spectrum
    power = calculate_power_spectrum(X)

    # Peak detection
    peak_bin, peak_frequency = find_dominant_frequency(power)

    print("\nDominant frequency:")
    print(f"Peak FFT bin   : {peak_bin}")
    print(f"Peak frequency : {peak_frequency:.2f} Hz")

    print("-----------------------------------------------")