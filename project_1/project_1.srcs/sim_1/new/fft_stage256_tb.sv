`timescale 1ns/1ps

module fft_stage256_tb;

    localparam int N = 256;
    localparam int DATA_WIDTH = 24;

    logic signed [DATA_WIDTH-1:0] x_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] x_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] y_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] y_imag [0:N-1];

    // --------------------------------------------------------
    // DUT - Stage 8
    // --------------------------------------------------------

    fft_stage256 #(
        .STAGE(8),
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .TWIDDLE_WIDTH(16),
        .FRAC_BITS(15)
    ) DUT (
        .x_real(x_real),
        .x_imag(x_imag),
        .y_real(y_real),
        .y_imag(y_imag)
    );

    // --------------------------------------------------------
    // Test
    // --------------------------------------------------------

    initial begin

        // ----------------------------------------------------
        // Initialize all inputs
        // ----------------------------------------------------

        for (int i = 0; i < N; i++) begin
            x_real[i] = 0;
            x_imag[i] = 0;
        end

        // ----------------------------------------------------
        // Stage 8 test
        //
        // Stage 8 butterflies:
        //
        // x[0]   <-> x[128]
        // x[1]   <-> x[129]
        // ...
        // x[127] <-> x[255]
        //
        // Twiddle addresses:
        //
        // 0, 1, 2, ... 127
        // ----------------------------------------------------

        // A side: x[0] - x[127]
        for (int i = 0; i < 128; i++) begin
            x_real[i] = 24'sd4096 + (i * 32);
        end

        // B side: x[128] - x[255]
        for (int i = 128; i < 256; i++) begin
            x_real[i] = 24'sd2048 + ((i - 128) * 16);
        end

        // ----------------------------------------------------
        // Allow combinational logic to settle
        // ----------------------------------------------------

        #10;

        // ----------------------------------------------------
        // Display results
        // ----------------------------------------------------

        $display("-----------------------------------------------");
        $display("          256-Point FFT Stage 8 Test");
        $display("-----------------------------------------------");

        for (int i = 0; i < 256; i++) begin
            $display("y[%0d] = %0d + j%0d",
                     i,
                     y_real[i],
                     y_imag[i]);
        end

        $display("-----------------------------------------------");

        $finish;

    end

endmodule