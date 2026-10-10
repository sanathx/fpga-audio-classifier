`timescale 1ns/1ps

module complex_mult_tb;

    localparam int DATA_WIDTH = 24;
    localparam int TWIDDLE_WIDTH = 16;
    localparam int FRAC_BITS = 15;

    logic signed [DATA_WIDTH-1:0] a_real;
    logic signed [DATA_WIDTH-1:0] a_imag;

    logic signed [TWIDDLE_WIDTH-1:0] b_real;
    logic signed [TWIDDLE_WIDTH-1:0] b_imag;

    logic signed [DATA_WIDTH-1:0] y_real;
    logic signed [DATA_WIDTH-1:0] y_imag;

    complex_mult #(
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(TWIDDLE_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) DUT (
        .a_real(a_real),
        .a_imag(a_imag),
        .b_real(b_real),
        .b_imag(b_imag),
        .y_real(y_real),
        .y_imag(y_imag)
    );

    initial begin

        // ----------------------------------------------------
        // Test:
        //
        // A = 0.5 + j0
        // B = 1.0 + j0
        //
        // Expected:
        //
        // A × B = 0.5 + j0
        //
        // Q1.15:
        // 0.5 = 16384
        // 1.0 = 32767
        // ----------------------------------------------------

        a_real = 24'sd16384;   // +0.5
        a_imag = 24'sd0;

        b_real = 16'sd0;
        b_imag = -16'sd32768;  // -1.0

        #10;

        $display("-----------------------------------------------");
        $display("          Complex Multiplier Test");
        $display("-----------------------------------------------");

        $display("A = %0d + j%0d", a_real, a_imag);
        $display("B = %0d + j%0d", b_real, b_imag);

        $display("Y = %0d + j%0d", y_real, y_imag);

        $display("-----------------------------------------------");

        $finish;

    end

endmodule