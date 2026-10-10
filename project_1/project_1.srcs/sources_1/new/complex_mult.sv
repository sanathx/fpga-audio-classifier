
module complex_mult #(
    parameter int DATA_WIDTH = 24,
    parameter int TWIDDLE_WIDTH = 16,
    parameter int FRAC_BITS = 15
)(
    input  logic signed [DATA_WIDTH-1:0] a_real,
    input  logic signed [DATA_WIDTH-1:0] a_imag,

    input  logic signed [TWIDDLE_WIDTH-1:0] b_real,
    input  logic signed [TWIDDLE_WIDTH-1:0] b_imag,

    output logic signed [DATA_WIDTH-1:0] y_real,
    output logic signed [DATA_WIDTH-1:0] y_imag
);

    // --------------------------------------------------------
    // Intermediate multiplication results
    // --------------------------------------------------------

    logic signed [DATA_WIDTH+TWIDDLE_WIDTH-1:0] mult_rr;
    logic signed [DATA_WIDTH+TWIDDLE_WIDTH-1:0] mult_ii;
    logic signed [DATA_WIDTH+TWIDDLE_WIDTH-1:0] mult_ri;
    logic signed [DATA_WIDTH+TWIDDLE_WIDTH-1:0] mult_ir;

    logic signed [DATA_WIDTH+TWIDDLE_WIDTH:0] real_temp;
    logic signed [DATA_WIDTH+TWIDDLE_WIDTH:0] imag_temp;

    // --------------------------------------------------------
    // Complex multiplication
    //
    // (a + jb)(c + jd)
    //
    // real = ac - bd
    // imag = ad + bc
    // --------------------------------------------------------

    assign mult_rr = a_real * b_real;
    assign mult_ii = a_imag * b_imag;

    assign mult_ri = a_real * b_imag;
    assign mult_ir = a_imag * b_real;

    assign real_temp = mult_rr - mult_ii;
    assign imag_temp = mult_ri + mult_ir;

    // --------------------------------------------------------
    // Convert back from Q-format
    // --------------------------------------------------------

    assign y_real = real_temp >>> FRAC_BITS;
    assign y_imag = imag_temp >>> FRAC_BITS;

endmodule