`timescale 1ns / 1ps

module tb_basic_scheduler;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg enable;
    reg in_valid;
    reg [15:0] in_timestamp;
    reg pe_ready;

    wire out_valid;
    wire [6:0] out_neuron_id;
    wire [15:0] out_timestamp;
    wire [31:0] op_count;

    basic_scheduler #(
        .NUM_NEURONS(8),
        .NEURON_ID_W(7),
        .TS_WIDTH(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .in_valid(in_valid),
        .in_timestamp(in_timestamp),
        .pe_ready(pe_ready),
        .out_valid(out_valid),
        .out_neuron_id(out_neuron_id),
        .out_timestamp(out_timestamp),
        .op_count(op_count)
    );

    integer i;
    initial begin
        rst_n = 0;
        enable = 0;
        in_valid = 0;
        in_timestamp = 0;
        pe_ready = 1;

        repeat (2) @(posedge clk);
        rst_n <= 1;
        enable <= 1;

        for (i = 0; i < 5; i = i + 1) begin
            @(posedge clk);
            in_valid <= 1;
            in_timestamp <= i;
        end

        @(posedge clk);
        in_valid <= 0;

        repeat (3) begin
            @(posedge clk);
            if (out_valid) begin
                $display("RR schedule: id=%0d ts=%0d", out_neuron_id, out_timestamp);
            end
        end

        $display("Basic scheduler op_count=%0d", op_count);
        #20;
        $finish;
    end
endmodule
