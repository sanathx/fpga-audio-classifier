`timescale 1ns/1ps

module twiddle_rom256_tb;

    logic [6:0] addr;

    logic signed [15:0] W_real;
    logic signed [15:0] W_imag;

    twiddle_rom256 DUT (
        .addr(addr),
        .W_real(W_real),
        .W_imag(W_imag)
    );

    initial begin

        $display("-----------------------------------------------");
        $display("       256-Point Twiddle ROM Test");
        $display("-----------------------------------------------");

        addr = 7'd0;
        #10;
        $display("W[0]  = %0d + j%0d", W_real, W_imag);

        addr = 7'd1;
        #10;
        $display("W[1]  = %0d + j%0d", W_real, W_imag);

        addr = 7'd2;
        #10;
        $display("W[2]  = %0d + j%0d", W_real, W_imag);

        addr = 7'd16;
        #10;
        $display("W[16] = %0d + j%0d", W_real, W_imag);

        addr = 7'd32;
        #10;
        $display("W[32] = %0d + j%0d", W_real, W_imag);

        addr = 7'd64;
        #10;
        $display("W[64] = %0d + j%0d", W_real, W_imag);

        addr = 7'd127;
        #10;
        $display("W[127] = %0d + j%0d", W_real, W_imag);

        $display("-----------------------------------------------");

        $finish;

    end

endmodule