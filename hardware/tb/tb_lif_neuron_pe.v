`timescale 1ns / 1ps

module tb_lif_neuron_pe;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg in_valid;
    reg [6:0] in_neuron_id;
    reg [15:0] in_timestamp;
    reg signed [15:0] membrane_in;
    reg signed [15:0] syn_input;

    wire out_valid;
    wire [6:0] out_neuron_id;
    wire [15:0] out_timestamp;
    wire signed [15:0] membrane_out;
    wire out_spike;

    lif_neuron_pe dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .membrane_in(membrane_in),
        .syn_input(syn_input),
        .out_valid(out_valid),
        .out_neuron_id(out_neuron_id),
        .out_timestamp(out_timestamp),
        .membrane_out(membrane_out),
        .out_spike(out_spike)
    );

    initial begin
        rst_n = 0;
        in_valid = 0;
        in_neuron_id = 0;
        in_timestamp = 0;
        membrane_in = 0;
        syn_input = 0;

        repeat (2) @(posedge clk);
        rst_n <= 1;

        // Two updates: second should likely trigger a spike after integration.
        @(posedge clk);
        in_valid <= 1;
        in_neuron_id <= 7'd3;
        in_timestamp <= 16'd10;
        membrane_in <= 16'sd30;
        syn_input <= 16'sd40;

        @(posedge clk);
        in_neuron_id <= 7'd3;
        in_timestamp <= 16'd11;
        membrane_in <= 16'sd50;
        syn_input <= 16'sd35;

        @(posedge clk);
        in_valid <= 0;

        repeat (8) begin
            @(posedge clk);
            if (out_valid) begin
                $display(
                    "PE out: id=%0d ts=%0d mem=%0d spike=%0d",
                    out_neuron_id,
                    out_timestamp,
                    membrane_out,
                    out_spike
                );
            end
        end

        $finish;
    end
endmodule
