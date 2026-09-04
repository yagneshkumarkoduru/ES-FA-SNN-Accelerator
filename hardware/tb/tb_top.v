`timescale 1ns / 1ps

module tb_top;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg in_valid;
    reg in_spike;
    reg [6:0] in_neuron_id;
    reg [15:0] in_timestamp;

    reg cfg_weight_we;
    reg [0:0] cfg_weight_bank;
    reg [5:0] cfg_weight_addr;
    reg [7:0] cfg_weight_data;

    wire out_valid_basic;
    wire out_spike_basic;
    wire [6:0] out_id_basic;
    wire [15:0] out_ts_basic;
    wire [31:0] basic_ops_basic;
    wire [31:0] advanced_ops_basic_unused;
    wire [31:0] pe_ops_basic;

    wire out_valid_adv;
    wire out_spike_adv;
    wire [6:0] out_id_adv;
    wire [15:0] out_ts_adv;
    wire [31:0] basic_ops_adv_unused;
    wire [31:0] advanced_ops_adv;
    wire [31:0] pe_ops_adv;

    integer i;
    integer basic_spike_count;
    integer adv_spike_count;

    snn_top #(
        .NUM_NEURONS(128),
        .NEURON_ID_W(7),
        .TS_WIDTH(16),
        .STATE_WIDTH(16),
        .WEIGHT_ADDR_W(6)
    ) dut_basic (
        .clk(clk),
        .rst_n(rst_n),
        .mode_advanced(1'b0),
        .in_valid(in_valid),
        .in_spike(in_spike),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_bank(cfg_weight_bank),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .out_valid(out_valid_basic),
        .out_spike(out_spike_basic),
        .out_neuron_id(out_id_basic),
        .out_timestamp(out_ts_basic),
        .basic_op_count(basic_ops_basic),
        .advanced_op_count(advanced_ops_basic_unused),
        .pe_op_count(pe_ops_basic)
    );

    snn_top #(
        .NUM_NEURONS(128),
        .NEURON_ID_W(7),
        .TS_WIDTH(16),
        .STATE_WIDTH(16),
        .WEIGHT_ADDR_W(6)
    ) dut_adv (
        .clk(clk),
        .rst_n(rst_n),
        .mode_advanced(1'b1),
        .in_valid(in_valid),
        .in_spike(in_spike),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_bank(cfg_weight_bank),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .out_valid(out_valid_adv),
        .out_spike(out_spike_adv),
        .out_neuron_id(out_id_adv),
        .out_timestamp(out_ts_adv),
        .basic_op_count(basic_ops_adv_unused),
        .advanced_op_count(advanced_ops_adv),
        .pe_op_count(pe_ops_adv)
    );

    always @(posedge clk) begin
        if (out_valid_basic && out_spike_basic) begin
            basic_spike_count <= basic_spike_count + 1;
        end
        if (out_valid_adv && out_spike_adv) begin
            adv_spike_count <= adv_spike_count + 1;
        end
    end

    initial begin
        rst_n = 0;
        in_valid = 0;
        in_spike = 0;
        in_neuron_id = 0;
        in_timestamp = 0;
        cfg_weight_we = 0;
        cfg_weight_bank = 0;
        cfg_weight_addr = 0;
        cfg_weight_data = 0;
        basic_spike_count = 0;
        adv_spike_count = 0;

        repeat (3) @(posedge clk);
        rst_n <= 1;

        // Program simple positive weights for all neuron addresses.
        for (i = 0; i < 128; i = i + 1) begin
            @(posedge clk);
            cfg_weight_we <= 1;
            cfg_weight_bank <= i[0];
            cfg_weight_addr <= i[6:1];
            cfg_weight_data <= 8'd24;
        end
        @(posedge clk);
        cfg_weight_we <= 0;

        // Feed 200 input events with 25% activity.
        for (i = 0; i < 200; i = i + 1) begin
            @(posedge clk);
            in_valid <= 1;
            in_spike <= (i % 4 == 0);
            in_neuron_id <= i % 128;
            in_timestamp <= i;
        end

        @(posedge clk);
        in_valid <= 0;
        in_spike <= 0;

        // Let pipelines and event queue drain.
        repeat (300) @(posedge clk);

        $display("--------------------------------------------------");
        $display("Basic Scheduler ops      : %0d", basic_ops_basic);
        $display("Advanced Scheduler ops   : %0d", advanced_ops_adv);
        $display("PE ops (basic mode)      : %0d", pe_ops_basic);
        $display("PE ops (advanced mode)   : %0d", pe_ops_adv);
        $display("Output spikes basic/adv  : %0d / %0d", basic_spike_count, adv_spike_count);

        if (advanced_ops_adv < basic_ops_basic) begin
            $display("PASS: Event-driven scheduling reduced operations.");
        end else begin
            $display("FAIL: Event-driven scheduling did not reduce operations.");
        end
        $display("--------------------------------------------------");

        #20;
        $finish;
    end
endmodule
