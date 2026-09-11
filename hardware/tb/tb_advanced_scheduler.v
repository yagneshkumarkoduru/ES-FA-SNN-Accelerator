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

    integer observed;
    integer seen_id [0:2];
    integer exp_id  [0:2];
    integer k;
    integer check_ok;

    // Continuous capture of the popped event sequence for the final
    // self-check (expected oldest-timestamp-first order: ids 2, 7, 4).
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            seen_id[observed] = out_neuron_id;
            observed = observed + 1;
        end
    end

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
        observed = 0;
        exp_id[0] = 2;
        exp_id[1] = 7;
        exp_id[2] = 4;

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

        // Self-check: 3 events pushed -> 3 popped, oldest-timestamp-first.
        check_ok = (observed == 3) && (op_count == 32'd3);
        for (k = 0; k < 3; k = k + 1) begin
            if (seen_id[k] !== exp_id[k]) begin
                check_ok = 0;
            end
        end
        if (check_ok) begin
            $display("PASS: event-driven pop sequence and op count verified");
        end else begin
            $display("FAIL: advanced scheduler check (observed=%0d op_count=%0d)", observed, op_count);
        end

        #20;
        $finish;
    end
endmodule
