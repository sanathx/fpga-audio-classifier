`timescale 1ns/1ps

module butterfly256_tb;

    localparam int DATA_WIDTH = 24;
    localparam int TWIDDLE_WIDTH = 16;
    localparam int FRAC_BITS = 15;

    logic signed [DATA_WIDTH-1:0] A_real;
    logic signed [DATA_WIDTH-1:0] A_imag;

    logic signed [DATA_WIDTH-1:0] B_real;
    logic signed [DATA_WIDTH-1:0] B_imag;

    logic signed [TWIDDLE_WIDTH-1:0] W_real;
    logic signed [TWIDDLE_WIDTH-1:0] W_imag;

    logic signed [DATA_WIDTH-1:0] Y0_real;
    logic signed [DATA_WIDTH-1:0] Y0_imag;

    logic signed [DATA_WIDTH-1:0] Y1_real;
    logic signed [DATA_WIDTH-1:0] Y1_imag;

    butterfly256 #(
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) DUT (
        .A_real(A_real),
        .A_imag(A_imag),

        .B_real(B_real),
        .B_imag(B_imag),

        .W_real(W_real),
        .W_imag(W_imag),

        .Y0_real(Y0_real),
        .Y0_imag(Y0_imag),

        .Y1_real(Y1_real),
        .Y1_imag(Y1_imag)
    );

    initial begin

        // ----------------------------------------------------
        // Test:
        //
        // A = 0.5 + j0
        // B = 0.25 + j0
        // W = 1.0 + j0
        //
        // BW = 0.25
        //
        // Y0 = A + BW = 0.75
        // Y1 = A - BW = 0.25
        // ----------------------------------------------------

        A_real = 24'sd16384;   // +0.5
        A_imag = 24'sd0;

        B_real = 24'sd8192;    // +0.25
        B_imag = 24'sd0;

        W_real = 16'sd0;
        W_imag = -16'sd32768;  // -j1

        #10;

        $display("-----------------------------------------------");
        $display("             Butterfly Test");
        $display("-----------------------------------------------");

        $display("A  = %0d + j%0d",
                 A_real, A_imag);

        $display("B  = %0d + j%0d",
                 B_real, B_imag);

        $display("W  = %0d + j%0d",
                 W_real, W_imag);

        $display("Y0 = %0d + j%0d",
                 Y0_real, Y0_imag);

        $display("Y1 = %0d + j%0d",
                 Y1_real, Y1_imag);

        $display("-----------------------------------------------");

        $finish;

    end

endmodule