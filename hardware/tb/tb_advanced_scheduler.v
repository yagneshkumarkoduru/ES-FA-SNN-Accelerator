`timescale 1ns / 1ps

module tb_advanced_scheduler;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg enable;
    reg in_event_valid;
    reg [6:0] in_event_neuron_id;
    reg [15:0] in_event_timestamp;
    wire in_event_ready;
    reg pe_ready;

    wire out_valid;
    wire [6:0] out_neuron_id;
    wire [15:0] out_timestamp;
    wire [8:0] queue_count;
    wire [31:0] op_count;

    advanced_scheduler #(
        .NEURON_ID_W(7),
        .TS_WIDTH(16),
        .QUEUE_DEPTH(256),
        .QUEUE_PTR_W(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .in_event_valid(in_event_valid),
        .in_event_neuron_id(in_event_neuron_id),
        .in_event_timestamp(in_event_timestamp),
        .in_event_ready(in_event_ready),
        .pe_ready(pe_ready),
        .out_valid(out_valid),
        .out_neuron_id(out_neuron_id),
        .out_timestamp(out_timestamp),
        .queue_count(queue_count),
        .op_count(op_count)
    );

    initial begin
        rst_n = 0;
        enable = 0;
        in_event_valid = 0;
        in_event_neuron_id = 0;
        in_event_timestamp = 0;
        pe_ready = 0;

        repeat (2) @(posedge clk);
        rst_n <= 1;
        enable <= 1;

        @(posedge clk); in_event_valid <= 1; in_event_neuron_id <= 7'd4; in_event_timestamp <= 16'd20;
        @(posedge clk); in_event_neuron_id <= 7'd2; in_event_timestamp <= 16'd9;
        @(posedge clk); in_event_neuron_id <= 7'd7; in_event_timestamp <= 16'd15;
        @(posedge clk); in_event_valid <= 0;

        pe_ready <= 1;
        repeat (8) begin
            @(posedge clk);
            if (out_valid) begin
                $display("ADV schedule id=%0d ts=%0d q=%0d", out_neuron_id, out_timestamp, queue_count);
            end
        end

        $display("Advanced scheduler op_count=%0d", op_count);
        #20;
        $finish;
    end
endmodule
