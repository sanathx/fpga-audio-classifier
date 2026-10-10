`timescale 1ns/1ps

module fft256 #(
    parameter int N = 256,
    parameter int DATA_WIDTH = 24,
    parameter int TWIDDLE_WIDTH = 16,
    parameter int FRAC_BITS = 15
)(
    input logic signed [DATA_WIDTH-1:0] x_real [0:N-1],
    input logic signed [DATA_WIDTH-1:0] x_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] X_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] X_imag [0:N-1],

    // --------------------------------------------------------
    // DEBUG OUTPUTS
    // --------------------------------------------------------

    output logic signed [DATA_WIDTH-1:0] stage1_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage1_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage2_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage2_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage3_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage3_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage4_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage4_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage5_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage5_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage6_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage6_debug_imag [0:N-1],

    output logic signed [DATA_WIDTH-1:0] stage7_debug_real [0:N-1],
    output logic signed [DATA_WIDTH-1:0] stage7_debug_imag [0:N-1]
);

    // --------------------------------------------------------
    // Connections between FFT stages
    // --------------------------------------------------------

    logic signed [DATA_WIDTH-1:0] stage1_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage1_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage2_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage2_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage3_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage3_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage4_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage4_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage5_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage5_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage6_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage6_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] stage7_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] stage7_imag [0:N-1];

    // --------------------------------------------------------
    // STAGE 1
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(1),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE1 (
        .x_real(x_real),
        .x_imag(x_imag),

        .y_real(stage1_real),
        .y_imag(stage1_imag)
    );

    // --------------------------------------------------------
    // STAGE 2
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(2),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE2 (
        .x_real(stage1_real),
        .x_imag(stage1_imag),

        .y_real(stage2_real),
        .y_imag(stage2_imag)
    );

    // --------------------------------------------------------
    // STAGE 3
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(3),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE3 (
        .x_real(stage2_real),
        .x_imag(stage2_imag),

        .y_real(stage3_real),
        .y_imag(stage3_imag)
    );

    // --------------------------------------------------------
    // STAGE 4
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(4),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE4 (
        .x_real(stage3_real),
        .x_imag(stage3_imag),

        .y_real(stage4_real),
        .y_imag(stage4_imag)
    );

    // --------------------------------------------------------
    // STAGE 5
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(5),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE5 (
        .x_real(stage4_real),
        .x_imag(stage4_imag),

        .y_real(stage5_real),
        .y_imag(stage5_imag)
    );

    // --------------------------------------------------------
    // STAGE 6
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(6),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE6 (
        .x_real(stage5_real),
        .x_imag(stage5_imag),

        .y_real(stage6_real),
        .y_imag(stage6_imag)
    );

    // --------------------------------------------------------
    // STAGE 7
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(7),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE7 (
        .x_real(stage6_real),
        .x_imag(stage6_imag),

        .y_real(stage7_real),
        .y_imag(stage7_imag)
    );

    // --------------------------------------------------------
    // STAGE 8
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(8),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) STAGE8 (
        .x_real(stage7_real),
        .x_imag(stage7_imag),

        .y_real(X_real),
        .y_imag(X_imag)
    );

    // --------------------------------------------------------
    // DEBUG CONNECTIONS
    // --------------------------------------------------------

    assign stage1_debug_real = stage1_real;
    assign stage1_debug_imag = stage1_imag;

    assign stage2_debug_real = stage2_real;
    assign stage2_debug_imag = stage2_imag;

    assign stage3_debug_real = stage3_real;
    assign stage3_debug_imag = stage3_imag;

    assign stage4_debug_real = stage4_real;
    assign stage4_debug_imag = stage4_imag;

    assign stage5_debug_real = stage5_real;
    assign stage5_debug_imag = stage5_imag;

    assign stage6_debug_real = stage6_real;
    assign stage6_debug_imag = stage6_imag;

    assign stage7_debug_real = stage7_real;
    assign stage7_debug_imag = stage7_imag;

endmodule