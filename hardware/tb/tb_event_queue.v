`timescale 1ns / 1ps

module tb_event_queue;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg push_valid;
    reg [6:0] push_id;
    reg [15:0] push_ts;
    wire push_ready;

    reg pop_req;
    wire pop_valid;
    wire [6:0] pop_id;
    wire [15:0] pop_ts;
    wire [8:0] queue_count;

    event_queue #(
        .DEPTH(256),
        .NEURON_ID_W(7),
        .TS_WIDTH(16),
        .PTR_W(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .push_valid(push_valid),
        .push_neuron_id(push_id),
        .push_timestamp(push_ts),
        .push_ready(push_ready),
        .pop_req(pop_req),
        .pop_valid(pop_valid),
        .pop_neuron_id(pop_id),
        .pop_timestamp(pop_ts),
        .queue_count(queue_count)
    );

    initial begin
        rst_n = 0;
        push_valid = 0;
        push_id = 0;
        push_ts = 0;
        pop_req = 0;

        repeat (2) @(posedge clk);
        rst_n <= 1;

        // Push out-of-order timestamps.
        @(posedge clk); push_valid <= 1; push_id <= 7'd1; push_ts <= 16'd10;
        @(posedge clk); push_id <= 7'd2; push_ts <= 16'd5;
        @(posedge clk); push_id <= 7'd3; push_ts <= 16'd12;
        @(posedge clk); push_valid <= 0;

        // Pop all events.
        repeat (4) begin
            @(posedge clk);
            pop_req <= 1;
            @(posedge clk);
            pop_req <= 0;
            if (pop_valid) begin
                $display("POP id=%0d ts=%0d count=%0d", pop_id, pop_ts, queue_count);
            end
        end

        #20;
        $finish;
    end
endmodule
