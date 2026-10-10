`timescale 1ns/1ps

module fft_stage256 #(
    parameter int STAGE = 1,
    parameter int N = 256,
    parameter int DATA_WIDTH = 24,
    parameter int TWIDDLE_WIDTH = 16,
    parameter int FRAC_BITS = 15
)(
    input  logic signed [DATA_WIDTH-1:0] x_real [0:N-1],
    input  logic signed [DATA_WIDTH-1:0] x_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] y_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] y_imag [0:N-1]
);

    // --------------------------------------------------------
    // Number of butterflies
    // --------------------------------------------------------

    localparam int NUM_BUTTERFLIES = N / 2;

    // --------------------------------------------------------
    // FFT geometry for this stage
    // --------------------------------------------------------

    localparam int HALF_GROUP = 2 ** (STAGE - 1);
    localparam int GROUP_SIZE = 2 ** STAGE;

    // --------------------------------------------------------
    // Twiddle factors
    // --------------------------------------------------------

    logic signed [TWIDDLE_WIDTH-1:0]
        W_real [0:NUM_BUTTERFLIES-1];

    logic signed [TWIDDLE_WIDTH-1:0]
        W_imag [0:NUM_BUTTERFLIES-1];

    // --------------------------------------------------------
    // Butterfly outputs
    // --------------------------------------------------------

    logic signed [DATA_WIDTH-1:0]
        butterfly_y0_real [0:NUM_BUTTERFLIES-1];

    logic signed [DATA_WIDTH-1:0]
        butterfly_y0_imag [0:NUM_BUTTERFLIES-1];

    logic signed [DATA_WIDTH-1:0]
        butterfly_y1_real [0:NUM_BUTTERFLIES-1];

    logic signed [DATA_WIDTH-1:0]
        butterfly_y1_imag [0:NUM_BUTTERFLIES-1];

    genvar b;

    // --------------------------------------------------------
    // Generate all 128 butterflies
    // --------------------------------------------------------

    generate

        for (b = 0; b < NUM_BUTTERFLIES; b++) begin : BUTTERFLY_GEN

            // ------------------------------------------------
            // Butterfly position
            // ------------------------------------------------

            localparam int GROUP =
                b / HALF_GROUP;

            localparam int OFFSET =
                b % HALF_GROUP;

            // ------------------------------------------------
            // Input indices
            // ------------------------------------------------

            localparam int A_INDEX =
                GROUP * GROUP_SIZE + OFFSET;

            localparam int B_INDEX =
                A_INDEX + HALF_GROUP;

            // ------------------------------------------------
            // Twiddle index
            // ------------------------------------------------

            localparam int TWIDDLE_INDEX =
                OFFSET * (N / GROUP_SIZE);

            // ------------------------------------------------
            // Twiddle ROM
            // ------------------------------------------------

            twiddle_rom256 ROM (
                .addr(TWIDDLE_INDEX[6:0]),
                .W_real(W_real[b]),
                .W_imag(W_imag[b])
            );

            // ------------------------------------------------
            // Butterfly
            // ------------------------------------------------

            butterfly256 BUTTERFLY (

                .A_real(x_real[A_INDEX]),
                .A_imag(x_imag[A_INDEX]),

                .B_real(x_real[B_INDEX]),
                .B_imag(x_imag[B_INDEX]),

                .W_real(W_real[b]),
                .W_imag(W_imag[b]),

                .Y0_real(butterfly_y0_real[b]),
                .Y0_imag(butterfly_y0_imag[b]),

                .Y1_real(butterfly_y1_real[b]),
                .Y1_imag(butterfly_y1_imag[b])
            );

            // ------------------------------------------------
            // Connect butterfly outputs
            // ------------------------------------------------

            assign y_real[A_INDEX] =
                butterfly_y0_real[b];

            assign y_imag[A_INDEX] =
                butterfly_y0_imag[b];

            assign y_real[B_INDEX] =
                butterfly_y1_real[b];

            assign y_imag[B_INDEX] =
                butterfly_y1_imag[b];

        end

    endgenerate

endmodule