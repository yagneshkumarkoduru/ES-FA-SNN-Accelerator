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

    integer popped_count;
    integer popped_id  [0:3];
    integer popped_ts  [0:3];
    integer exp_id     [0:2];
    integer exp_ts     [0:2];
    integer k;
    integer check_ok;

    initial begin
        rst_n = 0;
        push_valid = 0;
        push_id = 0;
        push_ts = 0;
        pop_req = 0;
        popped_count = 0;
        // Expected pop order for pushes (ts 10, 5, 12): the queue selects the
        // older of the first two entries each pop.
        exp_id[0] = 2; exp_ts[0] = 5;
        exp_id[1] = 1; exp_ts[1] = 10;
        exp_id[2] = 3; exp_ts[2] = 12;

        repeat (2) @(posedge clk);
        rst_n <= 1;

        // Push out-of-order timestamps.
        @(posedge clk); push_valid <= 1; push_id <= 7'd1; push_ts <= 16'd10;
        @(posedge clk); push_id <= 7'd2; push_ts <= 16'd5;
        @(posedge clk); push_id <= 7'd3; push_ts <= 16'd12;
        @(posedge clk); push_valid <= 0;

        // Pop all events. pop_valid / pop_id / pop_ts are registered one
        // cycle after the pop request, so they are sampled on the following
        // edge (the previous one-edge sampling window never observed them).
        repeat (4) begin
            @(posedge clk);
            pop_req <= 1;
            @(posedge clk);
            pop_req <= 0;
            @(posedge clk);
            if (pop_valid) begin
                $display("POP id=%0d ts=%0d count=%0d", pop_id, pop_ts, queue_count);
                popped_id[popped_count] = pop_id;
                popped_ts[popped_count] = pop_ts;
                popped_count = popped_count + 1;
            end
        end

        // Self-check: 3 pushes -> 3 pops, oldest-timestamp-first order, and
        // the queue returns to empty.
        check_ok = (popped_count == 3) && (queue_count == 0);
        for (k = 0; k < 3; k = k + 1) begin
            if ((popped_id[k] !== exp_id[k]) || (popped_ts[k] !== exp_ts[k])) begin
                check_ok = 0;
            end
        end
        if (check_ok) begin
            $display("PASS: event queue popped 3/3 events in timestamp order");
        end else begin
            $display("FAIL: event queue pop count/order mismatch (popped %0d)", popped_count);
        end

        #20;
        $finish;
    end
endmodule
