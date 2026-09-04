`timescale 1ns / 1ps

module tb_kv260_modes;
    reg clk = 0;
    always #5 clk = ~clk;  // 100 MHz simulation clock

    reg rst_n;
    reg mode_advanced;
    reg in_valid;
    reg in_spike;
    reg [6:0] in_neuron_id;
    reg [15:0] in_timestamp;

    reg cfg_weight_we;
    reg [0:0] cfg_weight_bank;
    reg [5:0] cfg_weight_addr;
    reg [7:0] cfg_weight_data;

    wire out_valid;
    wire out_spike;
    wire [6:0] out_id;
    wire [15:0] out_ts;
    wire [31:0] basic_ops;
    wire [31:0] adv_ops;
    wire [31:0] pe_ops;

    integer i;
    integer out_spike_count;
    integer cycle_count;
    integer first_input_cycle;
    integer done_cycle;

    snn_top #(
        .NUM_NEURONS(128),
        .NEURON_ID_W(7),
        .TS_WIDTH(16),
        .STATE_WIDTH(16),
        .WEIGHT_ADDR_W(6)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode_advanced(mode_advanced),
        .in_valid(in_valid),
        .in_spike(in_spike),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_bank(cfg_weight_bank),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .out_valid(out_valid),
        .out_spike(out_spike),
        .out_neuron_id(out_id),
        .out_timestamp(out_ts),
        .basic_op_count(basic_ops),
        .advanced_op_count(adv_ops),
        .pe_op_count(pe_ops)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            if (out_valid && out_spike) begin
                out_spike_count <= out_spike_count + 1;
            end
        end
    end

    initial begin
        rst_n = 0;
        mode_advanced = 0;
        in_valid = 0;
        in_spike = 0;
        in_neuron_id = 0;
        in_timestamp = 0;
        cfg_weight_we = 0;
        cfg_weight_bank = 0;
        cfg_weight_addr = 0;
        cfg_weight_data = 0;
        out_spike_count = 0;
        cycle_count = 0;
        first_input_cycle = -1;
        done_cycle = -1;

`ifdef MODE_ADV_EVENT
        mode_advanced = 1;
`else
        mode_advanced = 0;
`endif

        repeat (3) @(posedge clk);
        rst_n <= 1;

        // Initialize all weights to a small positive value.
        for (i = 0; i < 128; i = i + 1) begin
            @(posedge clk);
            cfg_weight_we <= 1;
            cfg_weight_bank <= i[0];
            cfg_weight_addr <= i[6:1];
            cfg_weight_data <= 8'd24;
        end
        @(posedge clk);
        cfg_weight_we <= 0;

        // Feed event stream with 25% activity in sparse input mode.
        for (i = 0; i < 256; i = i + 1) begin
            @(posedge clk);
            if (first_input_cycle < 0) begin
                first_input_cycle <= cycle_count;
            end
            in_valid <= 1;
            in_spike <= (i % 4 == 0);
            in_neuron_id <= i % 128;
            in_timestamp <= i;
        end

        @(posedge clk);
        in_valid <= 0;
        in_spike <= 0;

        repeat (320) @(posedge clk);
        done_cycle = cycle_count;

        $display("RESULT_MODE=%0d", mode_advanced);
        $display("RESULT_CYCLE_COUNT=%0d", cycle_count);
        $display("RESULT_ACTIVE_WINDOW_CYCLES=%0d", done_cycle - first_input_cycle);
        $display("RESULT_BASIC_OPS=%0d", basic_ops);
        $display("RESULT_ADV_OPS=%0d", adv_ops);
        $display("RESULT_PE_OPS=%0d", pe_ops);
        $display("RESULT_OUT_SPIKES=%0d", out_spike_count);
        $finish;
    end
endmodule
