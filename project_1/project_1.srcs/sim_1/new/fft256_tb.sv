`timescale 1ns/1ps

module fft256_tb;

    parameter N = 256;
    parameter DATA_WIDTH = 24;
    parameter NUM_FRAMES = 30;

    logic signed [DATA_WIDTH-1:0] x_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] x_imag [0:N-1];

    logic signed [DATA_WIDTH-1:0] y_real [0:N-1];
    logic signed [DATA_WIDTH-1:0] y_imag [0:N-1];

    integer i;
    integer frame;
    integer infile;
    integer outfile;
    integer status;

    fft256 DUT (
        .x_real(x_real),
        .x_imag(x_imag),
        .X_real(y_real),
        .X_imag(y_imag)
    );

    initial begin
        $display("======================================");
$display("NUM_FRAMES = %0d", NUM_FRAMES);
$display("======================================");

        // -------------------------------------------------
        // Open input file
        // -------------------------------------------------

        infile = $fopen(
            "C:/Users/sandb/Desktop/fpga-audio-classifier/data/fft_input.txt",
            "r"
        );

        if (infile == 0) begin
            $display("ERROR: Could not open fft_input.txt");
            $finish;
        end

        $display("Successfully opened fft_input.txt");


        // -------------------------------------------------
        // Open output file
        // -------------------------------------------------

        outfile = $fopen(
            "C:/Users/sandb/Desktop/fpga-audio-classifier/data/fft_output.txt",
            "w"
        );

        if (outfile == 0) begin
            $display("ERROR: Could not create fft_output.txt");
            $fclose(infile);
            $finish;
        end


        // -------------------------------------------------
        // Process each frame
        // -------------------------------------------------

        for (frame = 0; frame < NUM_FRAMES; frame = frame + 1) begin

            // ---------------------------------------------
            // Clear input arrays
            // ---------------------------------------------

            for (i = 0; i < N; i = i + 1) begin

                x_real[i] = 0;
                x_imag[i] = 0;

            end


            // ---------------------------------------------
            // Read one 256-sample frame
            // ---------------------------------------------

            for (i = 0; i < N; i = i + 1) begin

                status = $fscanf(
                    infile,
                    "%d %d\n",
                    x_real[i],
                    x_imag[i]
                );

                if (status != 2) begin

                    $display(
                        "ERROR reading frame %0d sample %0d",
                        frame,
                        i
                    );

                    $fclose(infile);
                    $fclose(outfile);

                    $finish;

                end

            end


            // ---------------------------------------------
            // Allow combinational FFT to settle
            // ---------------------------------------------

            #100;


            // ---------------------------------------------
            // Write frame header
            // ---------------------------------------------

            $fwrite(
                outfile,
                "# FRAME %0d\n",
                frame
            );


            // ---------------------------------------------
            // Write FFT output
            // ---------------------------------------------

            for (i = 0; i < N; i = i + 1) begin

                $fwrite(
                    outfile,
                    "%0d %0d\n",
                    y_real[i],
                    y_imag[i]
                );

            end


            // ---------------------------------------------
            // Display progress
            // ---------------------------------------------

            $display(
                "Processed frame %0d / %0d",
                frame + 1,
                NUM_FRAMES
            );

            $display(
                "  Y[4]   = %0d + j%0d",
                y_real[4],
                y_imag[4]
            );

            $display(
                "  Y[252] = %0d + j%0d",
                y_real[252],
                y_imag[252]
            );

        end


        // -------------------------------------------------
        // Close files
        // -------------------------------------------------

        $fclose(infile);
        $fclose(outfile);


        $display("");
        $display("============================================");
        $display("All %0d FFT frames processed.", NUM_FRAMES);
        $display("FFT output written to fft_output.txt");
        $display("============================================");


        $finish;

    end

endmodule