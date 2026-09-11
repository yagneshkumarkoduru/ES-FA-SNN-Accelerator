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
    integer observed;
    integer seen_id [0:7];
    integer k;
    integer check_ok;

    // Continuous capture: every accepted schedule pulse is recorded so the
    // final self-check sees all pulses, not just the ones inside the
    // display window below.
    always @(posedge clk) begin
        if (rst_n && out_valid) begin
            seen_id[observed] = out_neuron_id;
            observed = observed + 1;
        end
    end

    initial begin
        rst_n = 0;
        enable = 0;
        in_valid = 0;
        in_timestamp = 0;
        pe_ready = 1;
        observed = 0;

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

        // Self-check: 5 accepted inputs -> 5 ops, round-robin ids 0..4.
        check_ok = (observed == 5) && (op_count == 32'd5);
        for (k = 0; k < 5; k = k + 1) begin
            if (seen_id[k] !== k) begin
                check_ok = 0;
            end
        end
        if (check_ok) begin
            $display("PASS: round-robin id sequence and op count verified");
        end else begin
            $display("FAIL: round-robin check (observed=%0d op_count=%0d)", observed, op_count);
        end

        #20;
        $finish;
    end
endmodule
