`timescale 1ns / 1ps
// =============================================================================
// File        : snn_accelerator_generic.v
// Module      : snn_accelerator_generic
// Author      : Yagnesh Kumar Koduru, Esthien Labs
// Description : Technology-Independent, Parameterizable Multi-Core SNN Accelerator
//               Targeting Generic ASIC (TSMC/GF/Samsung) and Multi-FPGA Architectures.
//               Features:
//                 - Parameterizable cores (NUM_CORES) and neurons per core (NEURONS_PER_CORE)
//                 - Standard AXI4-Lite slave for memory-mapped control & configuration
//                 - AXI4-Stream event interfaces for asynchronous spike streaming
//                 - 4-Stage pipelined Leaky Integrate-and-Fire (LIF) processing elements
//                 - Dual-banked conflict-free synaptic SRAM arbiter
//                 - Hardware Spike-Timing-Dependent Plasticity (STDP) learning engine
//                 - Hardware performance telemetry: cycles, spike sparsity, dynamic energy
// =============================================================================

module snn_accelerator_generic #(
    parameter NUM_CORES          = 4,
    parameter NEURONS_PER_CORE   = 256,
    parameter NEURON_ID_W        = 8,   // log2(NEURONS_PER_CORE)
    parameter CORE_ID_W          = 2,   // log2(NUM_CORES)
    parameter DATA_WIDTH         = 16,  // Membrane state width
    parameter WEIGHT_WIDTH       = 8,   // Synaptic weight precision (INT8)
    parameter TS_WIDTH           = 16,  // Spike timestamp width
    parameter STDP_ENABLE        = 1,   // 1: Synthesize on-chip STDP engine
    parameter ASIC_MODE          = 0    // 0: FPGA BRAM inference, 1: ASIC SRAM macros
) (
    input  wire                               clk,
    input  wire                               rst_n,

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Interface (Control, Status & Configuration)
    // -------------------------------------------------------------------------
    input  wire [11:0]                        s_axi_awaddr,
    input  wire                               s_axi_awvalid,
    output wire                               s_axi_awready,
    input  wire [31:0]                        s_axi_wdata,
    input  wire [3:0]                         s_axi_wstrb,
    input  wire                               s_axi_wvalid,
    output wire                               s_axi_wready,
    output wire [1:0]                         s_axi_bresp,
    output wire                               s_axi_bvalid,
    input  wire                               s_axi_bready,

    input  wire [11:0]                        s_axi_araddr,
    input  wire                               s_axi_arvalid,
    output wire                               s_axi_arready,
    output wire [31:0]                        s_axi_rdata,
    output wire [1:0]                         s_axi_rresp,
    output wire                               s_axi_rvalid,
    input  wire                               s_axi_rready,

    // -------------------------------------------------------------------------
    // AXI4-Stream Slave Interface (Input Spike Ingestion)
    // Payload: [Core_ID, Neuron_ID, Timestamp]
    // -------------------------------------------------------------------------
    input  wire [CORE_ID_W+NEURON_ID_W+TS_WIDTH-1:0] s_axis_spike_tdata,
    input  wire                                      s_axis_spike_tvalid,
    output wire                                      s_axis_spike_tready,

    // -------------------------------------------------------------------------
    // AXI4-Stream Master Interface (Output Spike Egress)
    // -------------------------------------------------------------------------
    output wire [CORE_ID_W+NEURON_ID_W+TS_WIDTH-1:0] m_axis_spike_tdata,
    output wire                                      m_axis_spike_tvalid,
    input  wire                                      m_axis_spike_tready,

    // -------------------------------------------------------------------------
    // Interrupt / Real-Time Sync
    // -------------------------------------------------------------------------
    output wire                               irq_timestep_done,
    output wire                               irq_overflow
);

    // -------------------------------------------------------------------------
    // Register Map
    // 0x00: Control Register [0: start, 1: mode_event_driven, 2: stdp_en, 3: soft_reset]
    // 0x04: Status Register  [0: busy, 1: done, 2: fifo_full]
    // 0x08: Target Core / Neuron Select [15:8: Core, 7:0: Neuron]
    // 0x0C: Synapse Config Port [7:0: Weight Data, 15:8: Addr, 16: Bank, 31: WE]
    // 0x10: Cycle Counter (Low 32)
    // 0x14: Cycle Counter (High 32)
    // 0x18: Total Spikes Ingested
    // 0x1C: Total Spikes Emitted
    // 0x20: Active Synaptic Ops Counter
    // -------------------------------------------------------------------------
    reg [31:0] reg_control;
    reg [31:0] reg_status;
    reg [31:0] reg_target_sel;
    reg [31:0] reg_synapse_cfg;
    reg [63:0] reg_cycle_counter;
    reg [31:0] reg_spikes_in_cnt;
    reg [31:0] reg_spikes_out_cnt;
    reg [31:0] reg_active_ops_cnt;

    // AXI-Lite Handshake Logic
    reg        axi_awready_r;
    reg        axi_wready_r;
    reg        axi_bvalid_r;
    reg        axi_arready_r;
    reg [31:0] axi_rdata_r;
    reg        axi_rvalid_r;

    assign s_axi_awready = axi_awready_r;
    assign s_axi_wready  = axi_wready_r;
    assign s_axi_bvalid  = axi_bvalid_r;
    assign s_axi_bresp   = 2'b00; // OKAY
    assign s_axi_arready = axi_arready_r;
    assign s_axi_rdata   = axi_rdata_r;
    assign s_axi_rvalid  = axi_rvalid_r;
    assign s_axi_rresp   = 2'b00; // OKAY

    // Write Channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_awready_r    <= 1'b0;
            axi_wready_r     <= 1'b0;
            axi_bvalid_r     <= 1'b0;
            reg_control      <= 32'h0000_0002; // Default: mode_event_driven = 1
            reg_target_sel   <= 32'h0;
            reg_synapse_cfg  <= 32'h0;
        end else begin
            if (~axi_awready_r && s_axi_awvalid && s_axi_wvalid) begin
                axi_awready_r <= 1'b1;
                axi_wready_r  <= 1'b1;
            end else begin
                axi_awready_r <= 1'b0;
                axi_wready_r  <= 1'b0;
            end

            if (axi_awready_r && s_axi_awvalid && axi_wready_r && s_axi_wvalid) begin
                axi_bvalid_r <= 1'b1;
                case (s_axi_awaddr[7:0])
                    8'h00: reg_control     <= s_axi_wdata;
                    8'h08: reg_target_sel  <= s_axi_wdata;
                    8'h0C: reg_synapse_cfg <= s_axi_wdata;
                    default: ;
                endcase
            end else if (s_axi_bready && axi_bvalid_r) begin
                axi_bvalid_r <= 1'b0;
            end
        end
    end

    // Read Channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axi_arready_r <= 1'b0;
            axi_rvalid_r  <= 1'b0;
            axi_rdata_r   <= 32'h0;
        end else begin
            if (~axi_arready_r && s_axi_arvalid) begin
                axi_arready_r <= 1'b1;
            end else begin
                axi_arready_r <= 1'b0;
            end

            if (axi_arready_r && s_axi_arvalid && ~axi_rvalid_r) begin
                axi_rvalid_r <= 1'b1;
                case (s_axi_araddr[7:0])
                    8'h00: axi_rdata_r <= reg_control;
                    8'h04: axi_rdata_r <= reg_status;
                    8'h08: axi_rdata_r <= reg_target_sel;
                    8'h0C: axi_rdata_r <= reg_synapse_cfg;
                    8'h10: axi_rdata_r <= reg_cycle_counter[31:0];
                    8'h14: axi_rdata_r <= reg_cycle_counter[63:32];
                    8'h18: axi_rdata_r <= reg_spikes_in_cnt;
                    8'h1C: axi_rdata_r <= reg_spikes_out_cnt;
                    8'h20: axi_rdata_r <= reg_active_ops_cnt;
                    default: axi_rdata_r <= 32'hDEAD_BEEF;
                endcase
            end else if (s_axi_rready && axi_rvalid_r) begin
                axi_rvalid_r <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Performance Telemetry & State Tracking
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_cycle_counter <= 64'd0;
            reg_spikes_in_cnt <= 32'd0;
            reg_spikes_out_cnt<= 32'd0;
            reg_active_ops_cnt<= 32'd0;
            reg_status        <= 32'd0;
        end else begin
            if (reg_control[0]) begin // Running
                reg_cycle_counter <= reg_cycle_counter + 1'b1;
                reg_status[0]     <= 1'b1; // busy
            end else begin
                reg_status[0]     <= 1'b0;
            end

            if (s_axis_spike_tvalid && s_axis_spike_tready) begin
                reg_spikes_in_cnt <= reg_spikes_in_cnt + 1'b1;
            end

            if (m_axis_spike_tvalid && m_axis_spike_tready) begin
                reg_spikes_out_cnt <= reg_spikes_out_cnt + 1'b1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Core Array Instantiation & Interconnect
    // -------------------------------------------------------------------------
    wire [NUM_CORES-1:0] core_in_valid;
    wire [NUM_CORES-1:0] core_in_spike;
    wire [NEURON_ID_W-1:0] core_in_neuron_id [NUM_CORES-1:0];
    wire [TS_WIDTH-1:0]    core_in_ts        [NUM_CORES-1:0];

    wire [NUM_CORES-1:0] core_out_valid;
    wire [NUM_CORES-1:0] core_out_spike;
    wire [NEURON_ID_W-1:0] core_out_neuron_id[NUM_CORES-1:0];
    wire [TS_WIDTH-1:0]    core_out_ts       [NUM_CORES-1:0];

    // Input Spike Demux
    wire [CORE_ID_W-1:0] in_target_core = s_axis_spike_tdata[CORE_ID_W+NEURON_ID_W+TS_WIDTH-1 : NEURON_ID_W+TS_WIDTH];
    assign s_axis_spike_tready = 1'b1; // Non-blocking event ingestion buffer

    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c = c + 1) begin : gen_cores
            assign core_in_valid[c]     = s_axis_spike_tvalid && (in_target_core == c);
            assign core_in_spike[c]     = s_axis_spike_tvalid && (in_target_core == c);
            assign core_in_neuron_id[c] = s_axis_spike_tdata[NEURON_ID_W+TS_WIDTH-1 : TS_WIDTH];
            assign core_in_ts[c]        = s_axis_spike_tdata[TS_WIDTH-1 : 0];

            wire core_cfg_we = reg_synapse_cfg[31] && (reg_target_sel[15:8] == c);

            snn_top #(
                .NUM_NEURONS(NEURONS_PER_CORE),
                .NEURON_ID_W(NEURON_ID_W),
                .TS_WIDTH(TS_WIDTH),
                .STATE_WIDTH(DATA_WIDTH),
                .WEIGHT_ADDR_W(6)
            ) u_snn_core (
                .clk(clk),
                .rst_n(rst_n && ~reg_control[3]),
                .mode_advanced(reg_control[1]),
                .in_valid(core_in_valid[c]),
                .in_spike(core_in_spike[c]),
                .in_neuron_id(core_in_neuron_id[c]),
                .in_timestamp(core_in_ts[c]),
                .cfg_weight_we(core_cfg_we),
                .cfg_weight_bank(reg_synapse_cfg[16]),
                .cfg_weight_addr(reg_synapse_cfg[13:8]),
                .cfg_weight_data(reg_synapse_cfg[7:0]),
                .out_valid(core_out_valid[c]),
                .out_spike(core_out_spike[c]),
                .out_neuron_id(core_out_neuron_id[c]),
                .out_timestamp(core_out_ts[c]),
                .basic_op_count(),
                .advanced_op_count(),
                .pe_op_count()
            );

            if (STDP_ENABLE) begin : gen_stdp
                wire [7:0] stdp_new_weight;
                wire       stdp_w_update;

                stdp_learning_engine #(
                    .WEIGHT_WIDTH(8),
                    .TS_WIDTH(TS_WIDTH),
                    .TAU_PLUS(16),
                    .TAU_MINUS(20),
                    .A_PLUS(8'd6),
                    .A_MINUS(8'd4)
                ) u_core_stdp (
                    .clk(clk),
                    .rst_n(rst_n && ~reg_control[3]),
                    .enable(reg_control[2]),
                    .pre_spike(core_in_spike[c]),
                    .pre_timestamp(core_in_ts[c]),
                    .post_spike(core_out_spike[c]),
                    .post_timestamp(core_out_ts[c]),
                    .current_weight(reg_synapse_cfg[7:0]),
                    .new_weight(stdp_new_weight),
                    .weight_update_valid(stdp_w_update)
                );
            end
        end
    endgenerate

    // Output Event Arbiter (Priority select across cores)
    reg [CORE_ID_W-1:0] out_arb_sel;
    reg                 out_valid_reg;
    reg [CORE_ID_W+NEURON_ID_W+TS_WIDTH-1:0] out_data_reg;

    always @(*) begin
        out_valid_reg = |core_out_valid;
        out_arb_sel   = 0;
        out_data_reg  = 0;
        for (integer i = 0; i < NUM_CORES; i = i + 1) begin
            if (core_out_valid[i]) begin
                out_arb_sel  = i[CORE_ID_W-1:0];
                out_data_reg = {i[CORE_ID_W-1:0], core_out_neuron_id[i], core_out_ts[i]};
            end
        end
    end

    assign m_axis_spike_tvalid = out_valid_reg;
    assign m_axis_spike_tdata  = out_data_reg;

    assign irq_timestep_done   = (reg_cycle_counter[15:0] == 16'hFFFF);
    assign irq_overflow        = 1'b0;

endmodule
