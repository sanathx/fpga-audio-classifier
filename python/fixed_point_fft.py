import numpy as np


# ============================================================
# Configuration
# ============================================================

N = 256
FS = 8000

Q = 15
SCALE = 1 << Q

INPUT_BITS = 16
INTERNAL_BITS = 24

INT16_MAX = (1 << (INPUT_BITS - 1)) - 1
INT16_MIN = -(1 << (INPUT_BITS - 1))

INT24_MAX = (1 << (INTERNAL_BITS - 1)) - 1
INT24_MIN = -(1 << (INTERNAL_BITS - 1))


# ============================================================
# Fixed-point utilities
# ============================================================

def float_to_fixed(x):
    """
    Convert floating-point value to signed Q1.15.
    """

    value = int(np.round(x * SCALE))

    value = max(INT16_MIN, min(INT16_MAX, value))

    return value


def fixed_to_float(x):
    """
    Convert Q1.15 integer back to floating point.
    """

    return x / SCALE


def saturate_24(x):
    """
    Saturate value to signed 24-bit range.
    """

    return max(INT24_MIN, min(INT24_MAX, x))


# ============================================================
# Q1.15 multiplication
# ============================================================

def q15_multiply(a, b):
    """
    Multiply two fixed-point values.

    Integer multiplication:
        Q1.15 × Q1.15 = Q2.30

    Shift right by 15:
        Q2.30 -> Q1.15
    """

    product = a * b

    result = product >> Q

    return result


# ============================================================
# Complex multiplication
# ============================================================

def complex_multiply(a, b):

    ar, ai = a
    br, bi = b

    real = (
        q15_multiply(ar, br)
        - q15_multiply(ai, bi)
    )

    imag = (
        q15_multiply(ar, bi)
        + q15_multiply(ai, br)
    )

    return real, imag


# ============================================================
# Twiddle factor
# ============================================================

def twiddle(k, n):

    angle = -2 * np.pi * k / n

    real = np.cos(angle)
    imag = np.sin(angle)

    return (
        float_to_fixed(real),
        float_to_fixed(imag)
    )


# ============================================================
# Bit reversal
# ============================================================

def bit_reverse_indices(n):

    bits = int(np.log2(n))

    indices = []

    for i in range(n):

        reversed_bits = 0

        for b in range(bits):

            if i & (1 << b):
                reversed_bits |= 1 << (bits - 1 - b)

        indices.append(reversed_bits)

    return indices


# ============================================================
# 256-point radix-2 DIT FFT
# ============================================================

def fixed_point_fft(x):

    # --------------------------------------------------------
    # Convert input to internal representation
    # --------------------------------------------------------

    data = []

    for sample in x:

        real = float_to_fixed(np.real(sample))
        imag = float_to_fixed(np.imag(sample))

        data.append((real, imag))


    # --------------------------------------------------------
    # Bit-reversal permutation
    # --------------------------------------------------------

    indices = bit_reverse_indices(N)

    reordered = [data[i] for i in indices]

    data = reordered


    # --------------------------------------------------------
    # FFT stages
    # --------------------------------------------------------

    num_stages = int(np.log2(N))

    for stage in range(num_stages):

        # FFT block size:
        #
        # stage 0 -> 2
        # stage 1 -> 4
        # stage 2 -> 8
        # ...
        # stage 7 -> 256

        block_size = 2 ** (stage + 1)

        half_block = block_size // 2


        # ----------------------------------------------------
        # Process every FFT block
        # ----------------------------------------------------

        for block_start in range(0, N, block_size):

            for j in range(half_block):

                top_index = block_start + j
                bottom_index = block_start + j + half_block


                # ------------------------------------------------
                # Select twiddle factor
                # ------------------------------------------------

                twiddle_index = j * N // block_size

                W = twiddle(twiddle_index, N)


                # ------------------------------------------------
                # Read butterfly inputs
                # ------------------------------------------------

                A = data[top_index]
                B = data[bottom_index]


                # ------------------------------------------------
                # Complex multiplication
                # ------------------------------------------------

                P = complex_multiply(B, W)


                # ------------------------------------------------
                # Butterfly
                # ------------------------------------------------

                y0_real = A[0] + P[0]
                y0_imag = A[1] + P[1]

                y1_real = A[0] - P[0]
                y1_imag = A[1] - P[1]


                # ------------------------------------------------
                # Saturate to 24-bit internal width
                # ------------------------------------------------

                data[top_index] = (
                    saturate_24(y0_real),
                    saturate_24(y0_imag)
                )

                data[bottom_index] = (
                    saturate_24(y1_real),
                    saturate_24(y1_imag)
                )


    return data


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
# Main test
# ============================================================

if __name__ == "__main__":

    print("-----------------------------------------------")
    print("       256-Point Fixed-Point DIT FFT")
    print("-----------------------------------------------")

    print(f"FFT size       : {N}")
    print(f"Sampling rate  : {FS} Hz")
    print(f"Input format   : Q1.15")
    print(f"Internal width : {INTERNAL_BITS} bits")
    print(f"FFT stages     : {int(np.log2(N))}")

    # --------------------------------------------------------
    # Generate input
    # --------------------------------------------------------

    x = generate_test_signal()

    # --------------------------------------------------------
    # Fixed-point FFT
    # --------------------------------------------------------

    X_fixed = fixed_point_fft(x)

    # --------------------------------------------------------
    # Convert fixed-point output to floating point
    # --------------------------------------------------------

    X_fixed_float = np.array([
        fixed_to_float(real) + 1j * fixed_to_float(imag)
        for real, imag in X_fixed
    ])

    # --------------------------------------------------------
    # NumPy reference
    # --------------------------------------------------------

    X_numpy = np.fft.fft(x)

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = np.abs(X_numpy - X_fixed_float)

    print("\nMaximum absolute error:")
    print(f"{np.max(error):.6f}")

    print("\nFirst 10 FFT outputs:")

    for i in range(10):

        print(
            f"X[{i:3d}] = "
            f"{X_fixed_float[i].real:9.4f} "
            f"+ j{X_fixed_float[i].imag:9.4f}"
        )

    # --------------------------------------------------------
    # Frequency spectrum
    # --------------------------------------------------------

    magnitude = np.abs(X_fixed_float)

    frequencies = np.fft.fftfreq(N, 1 / FS)

    positive_magnitude = magnitude[:N // 2]
    positive_frequencies = frequencies[:N // 2]

    peak_indices = np.argsort(
        positive_magnitude
    )[-5:]

    peak_indices = peak_indices[
        np.argsort(
            positive_magnitude[peak_indices]
        )[::-1]
    ]

    print("\nStrongest frequency components:")

    for index in peak_indices:

        print(
            f"Bin {index:3d} : "
            f"{positive_frequencies[index]:7.2f} Hz "
            f" "
            f"Magnitude = "
            f"{positive_magnitude[index]:.3f}"
        )

    print("-----------------------------------------------")