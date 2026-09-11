// =============================================================================
// Module: esfa_top_core.v
// Architecture: ES-FA Synthesizable RTL Top-Level Core
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Description: Fully synthesizable neuromorphic accelerator top-level core
//              integrating the 4-stage pipelined LIF PE, a dual-bank weight
//              memory driven through the bank arbiter (PE reads on port 0,
//              STDP writebacks on port 1), a per-neuron membrane state table,
//              and the on-chip STDP plasticity engine.
//
//              Weight reads are real synchronous BRAM reads: an accepted
//              event takes 4 cycles (request, grant, bank output, compute)
//              before the PE pipeline latches it. One event is in flight at a
//              time; producers must observe event_in_ready (AXI-Stream style
//              valid/ready handshake) and hold event_in_valid until accepted.
//
//              Documented simplifications:
//              - Membrane state is a MEMBRANE_ENTRIES-deep register table
//                indexed by the low bits of the neuron id. This is a state
//                cache sized for single-core demos; the full SoC couples this
//                port to the neuron state BRAM
//                (hardware/memory/neuron_bram.v) instead.
//              - Synaptic weights live in the two parity-interleaved banks
//                below (ADDR_WIDTH-1 bits of address per bank), initialized
//                from the SYN_INIT_WEIGHT parameter and updated in place by
//                the STDP engine, so learning persists across events.
//              - The STDP engine tracks the single most recent event's
//                synapse (single-active-synapse simplification). A production
//                core runs one engine per synapse row or banks several
//                engines.
// =============================================================================

`timescale 1ns / 1ps

module esfa_top_core #(
    parameter DATA_WIDTH     = 16,
    parameter WEIGHT_WIDTH   = 8,
    parameter NEURON_ID_W    = 8,
    parameter TS_WIDTH       = 16,
    parameter ADDR_WIDTH     = 8,      // Weight memory: 2 banks x 2^(ADDR_WIDTH-1) entries
    parameter MEMBRANE_ENTRIES = 64,   // Membrane state table depth (power of two)
    parameter signed THRESHOLD = 16'sd64,
    parameter signed [WEIGHT_WIDTH-1:0] SYN_INIT_WEIGHT = 8'sd18
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // Spike Event Ingestion Stream (valid/ready handshake)
    input  wire                         event_in_valid,
    input  wire [NEURON_ID_W-1:0]       event_in_id,
    input  wire [TS_WIDTH-1:0]          event_in_ts,
    output wire                         event_in_ready,

    // Spike Event Egress Stream
    output wire                         event_out_valid,
    output wire [NEURON_ID_W-1:0]       event_out_id,
    output wire [TS_WIDTH-1:0]          event_out_ts,
    output wire                         event_out_spike,

    // STDP Training Feedback Interface
    // When post_spike_trigger is high in the same cycle an event is accepted,
    // the engine treats that event as the pre-synaptic spike and evaluates
    // LTP/LTD against post_spike_timestamp.
    input  wire                         stdp_enable,
    input  wire                         post_spike_trigger,
    input  wire [TS_WIDTH-1:0]          post_spike_timestamp,

    // Telemetry Registers
    output reg  [31:0]                  total_spikes_fired,
    output reg  [31:0]                  total_cycles_active,
    output reg  [31:0]                  stdp_weight_updates
);

    localparam BANK_ENTRIES = (1 << (ADDR_WIDTH-1));
    localparam MEM_IDX_W    = $clog2(MEMBRANE_ENTRIES);

    // -------------------------------------------------------------------------
    // Weight read FSM (one event in flight)
    //   W_IDLE    : accept an event, latch id/timestamp
    //   W_REQ     : weight read request presented to the bank arbiter
    //   W_WAIT    : grant seen; the bank's synchronous output lands next cycle
    //   W_COMPUTE : arbiter data valid; PE stage 1 latched this cycle
    // -------------------------------------------------------------------------
    localparam [1:0] W_IDLE = 2'd0, W_REQ = 2'd1, W_WAIT = 2'd2, W_COMPUTE = 2'd3;
    reg [1:0]             w_state;
    reg [NEURON_ID_W-1:0] ev_id_q;
    reg [TS_WIDTH-1:0]    ev_ts_q;
    reg [WEIGHT_WIDTH-1:0] weight_q;   // most recently read synapse weight

    // -------------------------------------------------------------------------
    // Physical weight banks (parity interleaved on address LSB)
    // -------------------------------------------------------------------------
    reg signed [WEIGHT_WIDTH-1:0] weight_bank0 [0:BANK_ENTRIES-1];
    reg signed [WEIGHT_WIDTH-1:0] weight_bank1 [0:BANK_ENTRIES-1];
    reg  [WEIGHT_WIDTH-1:0]       bram0_dout_q;
    reg  [WEIGHT_WIDTH-1:0]       bram1_dout_q;

    wire        bram0_en, bram0_we;
    wire [ADDR_WIDTH-2:0] bram0_addr;
    wire [WEIGHT_WIDTH-1:0] bram0_din;
    wire        bram1_en, bram1_we;
    wire [ADDR_WIDTH-2:0] bram1_addr;
    wire [WEIGHT_WIDTH-1:0] bram1_din;

    always @(posedge clk) begin
        if (bram0_en) begin
            if (bram0_we) weight_bank0[bram0_addr] <= bram0_din;
            bram0_dout_q <= weight_bank0[bram0_addr];
        end
        if (bram1_en) begin
            if (bram1_we) weight_bank1[bram1_addr] <= bram1_din;
            bram1_dout_q <= weight_bank1[bram1_addr];
        end
    end

    // -------------------------------------------------------------------------
    // STDP writeback staging: the engine's single-cycle write pulse is latched
    // and issued on arbiter port 1 only while no PE read is in flight, so the
    // write can never lose an arbitration round to the read stream.
    // -------------------------------------------------------------------------
    wire                    weight_write_en;
    wire signed [WEIGHT_WIDTH-1:0] updated_weight;
    reg                     stdp_wr_pending;
    reg [NEURON_ID_W-1:0]   stdp_wr_addr;
    reg [WEIGHT_WIDTH-1:0]  stdp_wr_data;
    wire                    stdp_wr_issue = stdp_wr_pending && (w_state == W_IDLE);

    // -------------------------------------------------------------------------
    // Bank arbiter: port 0 = PE weight reads, port 1 = STDP writebacks
    // -------------------------------------------------------------------------
    wire                    arb_req0_ready;
    wire [WEIGHT_WIDTH-1:0] arb_req0_dout;
    wire                    arb_req0_dout_valid;
    wire                    arb_req1_ready;
    wire [WEIGHT_WIDTH-1:0] arb_req1_dout;
    wire                    arb_req1_dout_valid;

    bram_bank_arbiter #(
        .DATA_WIDTH(WEIGHT_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_weight_arbiter (
        .clk(clk),
        .rst_n(rst_n),
        .req0_valid(w_state == W_REQ),
        .req0_we(1'b0),
        .req0_addr(ev_id_q),
        .req0_din({WEIGHT_WIDTH{1'b0}}),
        .req0_dout(arb_req0_dout),
        .req0_ready(arb_req0_ready),
        .req0_dout_valid(arb_req0_dout_valid),
        .req1_valid(stdp_wr_issue),
        .req1_we(1'b1),
        .req1_addr(stdp_wr_addr),
        .req1_din(stdp_wr_data),
        .req1_dout(arb_req1_dout),
        .req1_ready(arb_req1_ready),
        .req1_dout_valid(arb_req1_dout_valid),
        .bram0_en(bram0_en),
        .bram0_we(bram0_we),
        .bram0_addr(bram0_addr),
        .bram0_din(bram0_din),
        .bram0_dout(bram0_dout_q),
        .bram1_en(bram1_en),
        .bram1_we(bram1_we),
        .bram1_addr(bram1_addr),
        .bram1_din(bram1_din),
        .bram1_dout(bram1_dout_q)
    );

    // -------------------------------------------------------------------------
    // Membrane state table (per-neuron membrane cache, read on event,
    // written back after each PE update)
    // -------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] membrane_tbl [0:MEMBRANE_ENTRIES-1];

    wire                          pe_in_valid = (w_state == W_COMPUTE);
    wire [NEURON_ID_W-1:0]        pe_in_id    = ev_id_q;
    wire [TS_WIDTH-1:0]           pe_in_ts    = ev_ts_q;
    wire signed [DATA_WIDTH-1:0]  pe_membrane_in = membrane_tbl[ev_id_q[MEM_IDX_W-1:0]];
    wire signed [WEIGHT_WIDTH-1:0] pe_weight_in  = arb_req0_dout;
    wire signed [DATA_WIDTH-1:0]  pe_syn_in =
        {{(DATA_WIDTH-WEIGHT_WIDTH){pe_weight_in[WEIGHT_WIDTH-1]}}, pe_weight_in};

    // LIF PE Core Instance
    wire                                pe_out_valid;
    wire [NEURON_ID_W-1:0]              pe_out_id;
    wire [TS_WIDTH-1:0]                 pe_out_ts;
    wire signed [DATA_WIDTH-1:0]        pe_membrane_out;
    wire                                pe_spike_out;

    lif_pe_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH),
        .LEAK_SHIFT(3),
        .THRESHOLD(THRESHOLD),
        .RESET_VALUE(16'sd0)
    ) u_lif_pe (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pe_in_valid),
        .in_neuron_id(pe_in_id),
        .in_timestamp(pe_in_ts),
        .membrane_in(pe_membrane_in),
        .syn_input(pe_syn_in),
        .out_valid(pe_out_valid),
        .out_neuron_id(pe_out_id),
        .out_timestamp(pe_out_ts),
        .membrane_out(pe_membrane_out),
        .out_spike(pe_spike_out)
    );

    // Membrane write-back: persist the PE result so accumulation works
    // across events targeting the same neuron.
    always @(posedge clk) begin
        if (pe_out_valid) begin
            membrane_tbl[pe_out_id[MEM_IDX_W-1:0]] <= pe_membrane_out;
        end
    end

    // STDP Learning Engine Instance
    stdp_weight_updater #(
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .TIME_WIDTH(TS_WIDTH),
        .ALPHA_PLUS(8'd4),
        .ALPHA_MINUS(8'd3),
        .TAU_WINDOW(16'd32)
    ) u_stdp (
        .clk(clk),
        .rst_n(rst_n),
        .pre_spike_valid(event_in_valid && stdp_enable && (w_state == W_IDLE)),
        .pre_spike_time(event_in_ts),
        .post_spike_valid(post_spike_trigger && stdp_enable),
        .post_spike_time(post_spike_timestamp),
        .current_weight(weight_q),
        .updated_weight(updated_weight),
        .weight_write_en(weight_write_en)
    );

    // -------------------------------------------------------------------------
    // Weight read FSM sequencing
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_state  <= W_IDLE;
            ev_id_q  <= {NEURON_ID_W{1'b0}};
            ev_ts_q  <= {TS_WIDTH{1'b0}};
            weight_q <= {WEIGHT_WIDTH{1'b0}};
        end else begin
            case (w_state)
                W_IDLE: begin
                    if (event_in_valid) begin
                        ev_id_q <= event_in_id;
                        ev_ts_q <= event_in_ts;
                        w_state <= W_REQ;
                    end
                end
                W_REQ: begin
                    // Grant observed this cycle; bank output lands next cycle.
                    if (arb_req0_ready) w_state <= W_WAIT;
                end
                W_WAIT: begin
                    w_state <= W_COMPUTE;
                end
                W_COMPUTE: begin
                    weight_q <= arb_req0_dout;
                    w_state  <= W_IDLE;
                end
                default: w_state <= W_IDLE;
            endcase
        end
    end

    // STDP writeback staging and telemetry
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stdp_wr_pending     <= 1'b0;
            stdp_wr_addr        <= {NEURON_ID_W{1'b0}};
            stdp_wr_data        <= {WEIGHT_WIDTH{1'b0}};
            stdp_weight_updates <= 32'd0;
        end else begin
            if (weight_write_en) begin
                // Stage the update against the event currently in flight
                // (ev_id_q is stable through the whole read sequence). If a
                // previous write is still staged it is overwritten: the
                // newest update wins for this single-synapse engine.
                stdp_wr_pending     <= 1'b1;
                stdp_wr_addr        <= ev_id_q;
                stdp_wr_data        <= updated_weight;
                stdp_weight_updates <= stdp_weight_updates + 32'd1;
            end else if (stdp_wr_issue && arb_req1_ready) begin
                stdp_wr_pending <= 1'b0;
            end
        end
    end

    // Direct assignment to output stream
    assign event_in_ready  = (w_state == W_IDLE);
    assign event_out_valid = pe_out_valid;
    assign event_out_id    = pe_out_id;
    assign event_out_ts    = pe_out_ts;
    assign event_out_spike = pe_spike_out;

    // Telemetry Statistics
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_spikes_fired  <= 32'd0;
            total_cycles_active <= 32'd0;
        end else begin
            if (event_in_valid && event_in_ready) begin
                total_cycles_active <= total_cycles_active + 1'b1;
            end
            if (pe_out_valid && pe_spike_out) begin
                total_spikes_fired <= total_spikes_fired + 1'b1;
            end
        end
    end

    // Power-on initialization: both weight banks start from the configured
    // baseline synapse strength; the membrane table starts at rest.
    integer wi;
    initial begin
        for (wi = 0; wi < BANK_ENTRIES; wi = wi + 1) begin
            weight_bank0[wi] = SYN_INIT_WEIGHT;
            weight_bank1[wi] = SYN_INIT_WEIGHT;
        end
        for (wi = 0; wi < MEMBRANE_ENTRIES; wi = wi + 1) begin
            membrane_tbl[wi] = {DATA_WIDTH{1'b0}};
        end
    end

endmodule
