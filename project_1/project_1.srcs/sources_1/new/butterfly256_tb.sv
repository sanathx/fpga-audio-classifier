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


    butterfly256 DUT (

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

        // -----------------------------------------------------
        // Test:
        //
        // A = 0.5 + j0
        // B = 0.75 + j0
        // W = -j1
        // -----------------------------------------------------

        A_real = 24'sd16384;
        A_imag = 24'sd0;

        B_real = 24'sd24576;
        B_imag = 24'sd0;

        W_real = 16'sd0;
        W_imag = -16'sd32768;

        #10;

        $display("-----------------------------------------------");
        $display("           256-Point Butterfly");
        $display("-----------------------------------------------");

        $display(
            "Y0 = %0d + j%0d",
            Y0_real,
            Y0_imag
        );

        $display(
            "Y1 = %0d + j%0d",
            Y1_real,
            Y1_imag
        );

        $display("-----------------------------------------------");

        $finish;

    end

endmodule