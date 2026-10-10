`timescale 1ns/1ps

module twiddle_rom256 (
    input  logic [6:0] addr,

    output logic signed [15:0] W_real,
    output logic signed [15:0] W_imag
);

    always_comb begin

        case (addr)

            7'd0:   begin W_real =  16'sd32767; W_imag =      0; end
            7'd1:   begin W_real =  16'sd32758; W_imag =   -804; end
            7'd2:   begin W_real =  16'sd32729; W_imag =  -1608; end
            7'd3:   begin W_real =  16'sd32679; W_imag =  -2411; end
            7'd4:   begin W_real =  16'sd32610; W_imag =  -3212; end
            7'd5:   begin W_real =  16'sd32522; W_imag =  -4011; end
            7'd6:   begin W_real =  16'sd32413; W_imag =  -4808; end
            7'd7:   begin W_real =  16'sd32286; W_imag =  -5602; end
            7'd8:   begin W_real =  16'sd32138; W_imag =  -6393; end
            7'd9:   begin W_real =  16'sd31972; W_imag =  -7180; end
            7'd10:  begin W_real =  16'sd31785; W_imag =  -7962; end
            7'd11:  begin W_real =  16'sd31580; W_imag =  -8739; end
            7'd12:  begin W_real =  16'sd31356; W_imag =  -9512; end
            7'd13:  begin W_real =  16'sd31113; W_imag = -10278; end
            7'd14:  begin W_real =  16'sd30852; W_imag = -11039; end
            7'd15:  begin W_real =  16'sd30571; W_imag = -11793; end
            7'd16:  begin W_real =  16'sd30273; W_imag = -12539; end
            7'd17:  begin W_real =  16'sd29956; W_imag = -13279; end
            7'd18:  begin W_real =  16'sd29621; W_imag = -14010; end
            7'd19:  begin W_real =  16'sd29268; W_imag = -14732; end
            7'd20:  begin W_real =  16'sd28898; W_imag = -15446; end
            7'd21:  begin W_real =  16'sd28510; W_imag = -16151; end
            7'd22:  begin W_real =  16'sd28105; W_imag = -16846; end
            7'd23:  begin W_real =  16'sd27683; W_imag = -17530; end
            7'd24:  begin W_real =  16'sd27245; W_imag = -18205; end
            7'd25:  begin W_real =  16'sd26790; W_imag = -18868; end
            7'd26:  begin W_real =  16'sd26319; W_imag = -19519; end
            7'd27:  begin W_real =  16'sd25832; W_imag = -20159; end
            7'd28:  begin W_real =  16'sd25329; W_imag = -20787; end
            7'd29:  begin W_real =  16'sd24811; W_imag = -21403; end
            7'd30:  begin W_real =  16'sd24279; W_imag = -22005; end
            7'd31:  begin W_real =  16'sd23731; W_imag = -22594; end
            7'd32:  begin W_real =  16'sd23170; W_imag = -23170; end
            7'd33:  begin W_real =  16'sd22594; W_imag = -23731; end
            7'd34:  begin W_real =  16'sd22005; W_imag = -24279; end
            7'd35:  begin W_real =  16'sd21403; W_imag = -24811; end
            7'd36:  begin W_real =  16'sd20787; W_imag = -25329; end
            7'd37:  begin W_real =  16'sd20159; W_imag = -25832; end
            7'd38:  begin W_real =  16'sd19519; W_imag = -26319; end
            7'd39:  begin W_real =  16'sd18868; W_imag = -26790; end
            7'd40:  begin W_real =  16'sd18205; W_imag = -27245; end
            7'd41:  begin W_real =  16'sd17530; W_imag = -27683; end
            7'd42:  begin W_real =  16'sd16846; W_imag = -28105; end
            7'd43:  begin W_real =  16'sd16151; W_imag = -28510; end
            7'd44:  begin W_real =  16'sd15446; W_imag = -28898; end
            7'd45:  begin W_real =  16'sd14732; W_imag = -29268; end
            7'd46:  begin W_real =  16'sd14010; W_imag = -29621; end
            7'd47:  begin W_real =  16'sd13279; W_imag = -29956; end
            7'd48:  begin W_real =  16'sd12539; W_imag = -30273; end
            7'd49:  begin W_real =  16'sd11793; W_imag = -30571; end
            7'd50:  begin W_real =  16'sd11039; W_imag = -30852; end
            7'd51:  begin W_real =  16'sd10278; W_imag = -31113; end
            7'd52:  begin W_real =  16'sd9512;  W_imag = -31356; end
            7'd53:  begin W_real =  16'sd8739;  W_imag = -31580; end
            7'd54:  begin W_real =  16'sd7962;  W_imag = -31785; end
            7'd55:  begin W_real =  16'sd7180;  W_imag = -31972; end
            7'd56:  begin W_real =  16'sd6393;  W_imag = -32138; end
            7'd57:  begin W_real =  16'sd5602;  W_imag = -32286; end
            7'd58:  begin W_real =  16'sd4808;  W_imag = -32413; end
            7'd59:  begin W_real =  16'sd4011;  W_imag = -32522; end
            7'd60:  begin W_real =  16'sd3212;  W_imag = -32610; end
            7'd61:  begin W_real =  16'sd2411;  W_imag = -32679; end
            7'd62:  begin W_real =  16'sd1608;  W_imag = -32729; end
            7'd63:  begin W_real =  16'sd804;   W_imag = -32758; end
            7'd64:  begin W_real =      0;      W_imag = -32768; end
            7'd65:  begin W_real =   -804;      W_imag = -32758; end
            7'd66:  begin W_real =  -1608;      W_imag = -32729; end
            7'd67:  begin W_real =  -2411;      W_imag = -32679; end
            7'd68:  begin W_real =  -3212;      W_imag = -32610; end
            7'd69:  begin W_real =  -4011;      W_imag = -32522; end
            7'd70:  begin W_real =  -4808;      W_imag = -32413; end
            7'd71:  begin W_real =  -5602;      W_imag = -32286; end
            7'd72:  begin W_real =  -6393;      W_imag = -32138; end
            7'd73:  begin W_real =  -7180;      W_imag = -31972; end
            7'd74:  begin W_real =  -7962;      W_imag = -31785; end
            7'd75:  begin W_real =  -8739;      W_imag = -31580; end
            7'd76:  begin W_real =  -9512;      W_imag = -31356; end
            7'd77:  begin W_real = -10278;      W_imag = -31113; end
            7'd78:  begin W_real = -11039;      W_imag = -30852; end
            7'd79:  begin W_real = -11793;      W_imag = -30571; end
            7'd80:  begin W_real = -12539;      W_imag = -30273; end
            7'd81:  begin W_real = -13279;      W_imag = -29956; end
            7'd82:  begin W_real = -14010;      W_imag = -29621; end
            7'd83:  begin W_real = -14732;      W_imag = -29268; end
            7'd84:  begin W_real = -15446;      W_imag = -28898; end
            7'd85:  begin W_real = -16151;      W_imag = -28510; end
            7'd86:  begin W_real = -16846;      W_imag = -28105; end
            7'd87:  begin W_real = -17530;      W_imag = -27683; end
            7'd88:  begin W_real = -18205;      W_imag = -27245; end
            7'd89:  begin W_real = -18868;      W_imag = -26790; end
            7'd90:  begin W_real = -19519;      W_imag = -26319; end
            7'd91:  begin W_real = -20159;      W_imag = -25832; end
            7'd92:  begin W_real = -20787;      W_imag = -25329; end
            7'd93:  begin W_real = -21403;      W_imag = -24811; end
            7'd94:  begin W_real = -22005;      W_imag = -24279; end
            7'd95:  begin W_real = -22594;      W_imag = -23731; end
            7'd96:  begin W_real = -23170;      W_imag = -23170; end
            7'd97:  begin W_real = -23731;      W_imag = -22594; end
            7'd98:  begin W_real = -24279;      W_imag = -22005; end
            7'd99:  begin W_real = -24811;      W_imag = -21403; end
            7'd100: begin W_real = -25329;      W_imag = -20787; end
            7'd101: begin W_real = -25832;      W_imag = -20159; end
            7'd102: begin W_real = -26319;      W_imag = -19519; end
            7'd103: begin W_real = -26790;      W_imag = -18868; end
            7'd104: begin W_real = -27245;      W_imag = -18205; end
            7'd105: begin W_real = -27683;      W_imag = -17530; end
            7'd106: begin W_real = -28105;      W_imag = -16846; end
            7'd107: begin W_real = -28510;      W_imag = -16151; end
            7'd108: begin W_real = -28898;      W_imag = -15446; end
            7'd109: begin W_real = -29268;      W_imag = -14732; end
            7'd110: begin W_real = -29621;      W_imag = -14010; end
            7'd111: begin W_real = -29956;      W_imag = -13279; end
            7'd112: begin W_real = -30273;      W_imag = -12539; end
            7'd113: begin W_real = -30571;      W_imag = -11793; end
            7'd114: begin W_real = -30852;      W_imag = -11039; end
            7'd115: begin W_real = -31113;      W_imag = -10278; end
            7'd116: begin W_real = -31356;      W_imag =  -9512; end
            7'd117: begin W_real = -31580;      W_imag =  -8739; end
            7'd118: begin W_real = -31785;      W_imag =  -7962; end
            7'd119: begin W_real = -31972;      W_imag =  -7180; end
            7'd120: begin W_real = -32138;      W_imag =  -6393; end
            7'd121: begin W_real = -32286;      W_imag =  -5602; end
            7'd122: begin W_real = -32413;      W_imag =  -4808; end
            7'd123: begin W_real = -32522;      W_imag =  -4011; end
            7'd124: begin W_real = -32610;      W_imag =  -3212; end
            7'd125: begin W_real = -32679;      W_imag =  -2411; end
            7'd126: begin W_real = -32729;      W_imag =  -1608; end
            7'd127: begin W_real = -32758;      W_imag =   -804; end

            default: begin
                W_real = 16'sd0;
                W_imag = 16'sd0;
            end

        endcase

    end

endmodule