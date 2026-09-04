`timescale 1ns / 1ps

module snn_top #(
    parameter NUM_NEURONS  = 128,
    parameter NEURON_ID_W  = 7,
    parameter TS_WIDTH     = 16,
    parameter STATE_WIDTH  = 16,
    parameter WEIGHT_ADDR_W = 6
) (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           mode_advanced,   // 0: basic RR, 1: event-driven

    input  wire                           in_valid,
    input  wire                           in_spike,
    input  wire [NEURON_ID_W-1:0]         in_neuron_id,
    input  wire [TS_WIDTH-1:0]            in_timestamp,

    // Config write port for weight banks.
    input  wire                           cfg_weight_we,
    input  wire [0:0]                     cfg_weight_bank,
    input  wire [WEIGHT_ADDR_W-1:0]       cfg_weight_addr,
    input  wire [7:0]                     cfg_weight_data,

    output wire                           out_valid,
    output wire                           out_spike,
    output wire [NEURON_ID_W-1:0]         out_neuron_id,
    output wire [TS_WIDTH-1:0]            out_timestamp,

    output wire [31:0]                    basic_op_count,
    output wire [31:0]                    advanced_op_count,
    output reg  [31:0]                    pe_op_count
);
    // ------------------------------------------------------------------------
    // Router
    // ------------------------------------------------------------------------
    wire r_valid;
    wire r_spike;
    wire [NEURON_ID_W-1:0] r_id;
    wire [TS_WIDTH-1:0] r_ts;

    spike_router #(
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH)
    ) u_router (
        .clk(clk),
        .rst_n(rst_n),
        .mode_dense(!mode_advanced),
        .in_valid(in_valid),
        .in_spike(in_spike),
        .in_neuron_id(in_neuron_id),
        .in_timestamp(in_timestamp),
        .out_valid(r_valid),
        .out_spike(r_spike),
        .out_neuron_id(r_id),
        .out_timestamp(r_ts)
    );

    // ------------------------------------------------------------------------
    // Schedulers
    // ------------------------------------------------------------------------
    wire b_valid;
    wire [NEURON_ID_W-1:0] b_id;
    wire [TS_WIDTH-1:0] b_ts;
    wire [31:0] b_ops;

    basic_scheduler #(
        .NUM_NEURONS(NUM_NEURONS),
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH)
    ) u_basic_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .enable(!mode_advanced),
        .in_valid(r_valid),
        .in_timestamp(r_ts),
        .pe_ready(1'b1),
        .out_valid(b_valid),
        .out_neuron_id(b_id),
        .out_timestamp(b_ts),
        .op_count(b_ops)
    );

    wire a_valid;
    wire [NEURON_ID_W-1:0] a_id;
    wire [TS_WIDTH-1:0] a_ts;
    wire [8:0] a_queue_count;
    wire [31:0] a_ops;

    advanced_scheduler #(
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH),
        .QUEUE_DEPTH(256),
        .QUEUE_PTR_W(8)
    ) u_advanced_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .enable(mode_advanced),
        .in_event_valid(r_valid),
        .in_event_neuron_id(r_id),
        .in_event_timestamp(r_ts),
        .in_event_ready(),
        .pe_ready(1'b1),
        .out_valid(a_valid),
        .out_neuron_id(a_id),
        .out_timestamp(a_ts),
        .queue_count(a_queue_count),
        .op_count(a_ops)
    );

    assign basic_op_count = b_ops;
    assign advanced_op_count = a_ops;

    wire sched_valid = mode_advanced ? a_valid : b_valid;
    wire [NEURON_ID_W-1:0] sched_id = mode_advanced ? a_id : b_id;
    wire [TS_WIDTH-1:0] sched_ts = mode_advanced ? a_ts : b_ts;

    // ------------------------------------------------------------------------
    // Memory hierarchy: neuron state BRAM + banked weight BRAM
    // ------------------------------------------------------------------------
    wire [STATE_WIDTH-1:0] neuron_mem_dout_a;
    wire [STATE_WIDTH-1:0] neuron_mem_dout_b_unused;

    wire pe_out_valid;
    wire [NEURON_ID_W-1:0] pe_out_id;
    wire [TS_WIDTH-1:0] pe_out_ts;
    wire signed [STATE_WIDTH-1:0] pe_out_mem;
    wire pe_out_spike;

    neuron_bram #(
        .DATA_WIDTH(STATE_WIDTH),
        .ADDR_WIDTH(NEURON_ID_W)
    ) u_neuron_mem (
        .clk(clk),
        .we_a(1'b0),
        .addr_a(sched_id),
        .din_a({STATE_WIDTH{1'b0}}),
        .dout_a(neuron_mem_dout_a),
        .we_b(pe_out_valid),
        .addr_b(pe_out_id),
        .din_b(pe_out_mem),
        .dout_b(neuron_mem_dout_b_unused)
    );

    wire w_req0_grant;
    wire w_req0_valid;
    wire [7:0] w_req0_data;
    wire dummy_req1_grant;
    wire dummy_req1_valid;
    wire [7:0] dummy_req1_data;

    weight_bram_bank #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(WEIGHT_ADDR_W),
        .NUM_BANKS(2),
        .BANK_SEL_W(1)
    ) u_weight_mem (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(cfg_weight_we),
        .wr_bank(cfg_weight_bank),
        .wr_addr(cfg_weight_addr),
        .wr_data(cfg_weight_data),
        .req0_valid(sched_valid),
        .req0_bank(sched_id[0]),
        .req0_addr(sched_id[NEURON_ID_W-1:1]),
        .req0_grant(w_req0_grant),
        .req0_data_valid(w_req0_valid),
        .req0_data(w_req0_data),
        .req1_valid(1'b0),
        .req1_bank(1'b0),
        .req1_addr({WEIGHT_ADDR_W{1'b0}}),
        .req1_grant(dummy_req1_grant),
        .req1_data_valid(dummy_req1_valid),
        .req1_data(dummy_req1_data)
    );

    // ------------------------------------------------------------------------
    // Read->Compute pipeline glue
    // ------------------------------------------------------------------------
    reg                        sched_valid_d;
    reg [NEURON_ID_W-1:0]      sched_id_d;
    reg [TS_WIDTH-1:0]         sched_ts_d;

    reg                        pe_in_valid;
    reg [NEURON_ID_W-1:0]      pe_in_id;
    reg [TS_WIDTH-1:0]         pe_in_ts;
    reg signed [STATE_WIDTH-1:0] pe_in_mem;
    reg signed [STATE_WIDTH-1:0] pe_in_syn;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sched_valid_d <= 1'b0;
            sched_id_d    <= {NEURON_ID_W{1'b0}};
            sched_ts_d    <= {TS_WIDTH{1'b0}};
            pe_in_valid   <= 1'b0;
            pe_in_id      <= {NEURON_ID_W{1'b0}};
            pe_in_ts      <= {TS_WIDTH{1'b0}};
            pe_in_mem     <= {STATE_WIDTH{1'b0}};
            pe_in_syn     <= {STATE_WIDTH{1'b0}};
            pe_op_count   <= 32'd0;
        end else begin
            sched_valid_d <= sched_valid;
            sched_id_d    <= sched_id;
            sched_ts_d    <= sched_ts;

            pe_in_valid <= sched_valid_d && w_req0_valid;
            pe_in_id    <= sched_id_d;
            pe_in_ts    <= sched_ts_d;
            pe_in_mem   <= $signed(neuron_mem_dout_a);
            pe_in_syn   <= {{(STATE_WIDTH-8){w_req0_data[7]}}, w_req0_data};

            if (pe_out_valid) begin
                pe_op_count <= pe_op_count + 32'd1;
            end
        end
    end

    // ------------------------------------------------------------------------
    // LIF compute engine
    // ------------------------------------------------------------------------
    lif_neuron_pe #(
        .DATA_WIDTH(STATE_WIDTH),
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH),
        .LEAK_SHIFT(3),
        .THRESHOLD(16'sd64),
        .RESET_VALUE(16'sd0)
    ) u_lif_pe (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pe_in_valid),
        .in_neuron_id(pe_in_id),
        .in_timestamp(pe_in_ts),
        .membrane_in(pe_in_mem),
        .syn_input(pe_in_syn),
        .out_valid(pe_out_valid),
        .out_neuron_id(pe_out_id),
        .out_timestamp(pe_out_ts),
        .membrane_out(pe_out_mem),
        .out_spike(pe_out_spike)
    );

    assign out_valid = pe_out_valid;
    assign out_spike = pe_out_spike;
    assign out_neuron_id = pe_out_id;
    assign out_timestamp = pe_out_ts;

endmodule
