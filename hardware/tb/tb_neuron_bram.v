`timescale 1ns / 1ps

module tb_neuron_bram;
    reg clk = 0;
    always #5 clk = ~clk;

    reg we_a;
    reg [3:0] addr_a;
    reg [15:0] din_a;
    wire [15:0] dout_a;

    reg we_b;
    reg [3:0] addr_b;
    reg [15:0] din_b;
    wire [15:0] dout_b;

    neuron_bram #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(4)
    ) dut (
        .clk(clk),
        .we_a(we_a),
        .addr_a(addr_a),
        .din_a(din_a),
        .dout_a(dout_a),
        .we_b(we_b),
        .addr_b(addr_b),
        .din_b(din_b),
        .dout_b(dout_b)
    );

    initial begin
        we_a = 0; we_b = 0;
        addr_a = 0; addr_b = 0;
        din_a = 0; din_b = 0;

        @(posedge clk);
        we_a <= 1; addr_a <= 4'd3; din_a <= 16'h0011;
        we_b <= 1; addr_b <= 4'd5; din_b <= 16'h00AA;

        @(posedge clk);
        we_a <= 0; we_b <= 0;
        addr_a <= 4'd3;
        addr_b <= 4'd5;

        @(posedge clk);
        $display("BRAM A[3]=0x%0h (expected 0x11)", dout_a);
        $display("BRAM B[5]=0x%0h (expected 0xAA)", dout_b);

        #20;
        $finish;
    end
endmodule
