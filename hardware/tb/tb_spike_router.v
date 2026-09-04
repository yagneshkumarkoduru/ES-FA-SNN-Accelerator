`timescale 1ns / 1ps

module tb_spike_router;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg mode_dense;
    reg in_valid;
    reg in_spike;
    reg [6:0] in_neuron_id;
    reg [15:0] in_timestamp;

    wire out_valid;
    wire out_spike;
    wire [6:0] out_neuron_id;
    wire [15:0] out_timestamp;

    spike_router dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode_dense(mode_dense),
        .in_valid(in_valid),
        .in_spike(in_spike),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .out_valid(out_valid),
        .out_spike(out_spike),
        .out_neuron_id(out_neuron_id),
        .out_timestamp(out_timestamp)
    );

    initial begin
        rst_n = 0;
        mode_dense = 1;
        in_valid = 0;
        in_spike = 0;
        in_neuron_id = 0;
        in_timestamp = 0;

        repeat (2) @(posedge clk);
        rst_n <= 1;

        // Dense mode forwards everything.
        @(posedge clk);
        in_valid <= 1; in_spike <= 0; in_neuron_id <= 7'd5; in_timestamp <= 16'd1;
        @(posedge clk);
        $display("Dense mode out_valid=%0d (expect 1)", out_valid);

        // Sparse mode forwards only active spikes.
        mode_dense <= 0;
        @(posedge clk);
        in_valid <= 1; in_spike <= 0; in_neuron_id <= 7'd6; in_timestamp <= 16'd2;
        @(posedge clk);
        $display("Sparse mode out_valid=%0d (expect 0)", out_valid);
        @(posedge clk);
        in_valid <= 1; in_spike <= 1; in_neuron_id <= 7'd7; in_timestamp <= 16'd3;
        @(posedge clk);
        $display("Sparse+spike out_valid=%0d (expect 1)", out_valid);

        #20;
        $finish;
    end
endmodule
