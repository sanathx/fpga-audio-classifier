`timescale 1ns/1ps

module butterfly256 #(
    parameter int DATA_WIDTH = 24,
    parameter int TWIDDLE_WIDTH = 16,
    parameter int FRAC_BITS = 15
)(
    input logic signed [DATA_WIDTH-1:0] A_real,
    input logic signed [DATA_WIDTH-1:0] A_imag,

    input logic signed [DATA_WIDTH-1:0] B_real,
    input logic signed [DATA_WIDTH-1:0] B_imag,

    input logic signed [TWIDDLE_WIDTH-1:0] W_real,
    input logic signed [TWIDDLE_WIDTH-1:0] W_imag,

    output logic signed [DATA_WIDTH-1:0] Y0_real,
    output logic signed [DATA_WIDTH-1:0] Y0_imag,

    output logic signed [DATA_WIDTH-1:0] Y1_real,
    output logic signed [DATA_WIDTH-1:0] Y1_imag
);

    // --------------------------------------------------------
    // B × W
    // --------------------------------------------------------

    logic signed [DATA_WIDTH-1:0] P_real;
    logic signed [DATA_WIDTH-1:0] P_imag;

    complex_mult #(
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) MULT (
        .a_real(B_real),
        .a_imag(B_imag),

        .b_real(W_real),
        .b_imag(W_imag),

        .y_real(P_real),
        .y_imag(P_imag)
    );

    // --------------------------------------------------------
    // Butterfly
    //
    // Y0 = A + BW
    // Y1 = A - BW
    // --------------------------------------------------------

    assign Y0_real = A_real + P_real;
    assign Y0_imag = A_imag + P_imag;

    assign Y1_real = A_real - P_real;
    assign Y1_imag = A_imag - P_imag;

endmodule