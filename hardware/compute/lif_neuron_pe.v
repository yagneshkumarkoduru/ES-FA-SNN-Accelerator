`timescale 1ns / 1ps

// 4-stage pipelined LIF PE:
// 1) Read/Latch state
// 2) Integrate with leak
// 3) Threshold and spike generation
// 4) Output write-back payload
module lif_neuron_pe #(
    parameter DATA_WIDTH   = 16,
    parameter NEURON_ID_W  = 7,
    parameter TS_WIDTH     = 16,
    parameter LEAK_SHIFT   = 3,
    parameter signed THRESHOLD   = 16'sd64,
    parameter signed RESET_VALUE = 16'sd0
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    input  wire [NEURON_ID_W-1:0]            in_neuron_id,
    input  wire [TS_WIDTH-1:0]               in_timestamp,
    input  wire signed [DATA_WIDTH-1:0]      membrane_in,
    input  wire signed [DATA_WIDTH-1:0]      syn_input,

    output reg                               out_valid,
    output reg  [NEURON_ID_W-1:0]            out_neuron_id,
    output reg  [TS_WIDTH-1:0]               out_timestamp,
    output reg  signed [DATA_WIDTH-1:0]      membrane_out,
    output reg                               out_spike
);
    // Stage-1
    reg                                s1_valid;
    reg  [NEURON_ID_W-1:0]             s1_id;
    reg  [TS_WIDTH-1:0]                s1_ts;
    reg  signed [DATA_WIDTH-1:0]       s1_mem;
    reg  signed [DATA_WIDTH-1:0]       s1_syn;

    // Stage-2
    reg                                s2_valid;
    reg  [NEURON_ID_W-1:0]             s2_id;
    reg  [TS_WIDTH-1:0]                s2_ts;
    reg  signed [DATA_WIDTH-1:0]       s2_mem_u;

    // Stage-3
    reg                                s3_valid;
    reg  [NEURON_ID_W-1:0]             s3_id;
    reg  [TS_WIDTH-1:0]                s3_ts;
    reg  signed [DATA_WIDTH-1:0]       s3_mem_next;
    reg                                s3_spike;

    wire signed [DATA_WIDTH-1:0] s2_leaked = s1_mem - (s1_mem >>> LEAK_SHIFT);
    wire signed [DATA_WIDTH-1:0] s2_updated = s2_leaked + s1_syn;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid      <= 1'b0;
            s2_valid      <= 1'b0;
            s3_valid      <= 1'b0;
            out_valid     <= 1'b0;
            out_neuron_id <= {NEURON_ID_W{1'b0}};
            out_timestamp <= {TS_WIDTH{1'b0}};
            membrane_out  <= {DATA_WIDTH{1'b0}};
            out_spike     <= 1'b0;
        end else begin
            // Stage 1
            s1_valid <= in_valid;
            s1_id    <= in_neuron_id;
            s1_ts    <= in_timestamp;
            s1_mem   <= membrane_in;
            s1_syn   <= syn_input;

            // Stage 2
            s2_valid <= s1_valid;
            s2_id    <= s1_id;
            s2_ts    <= s1_ts;
            s2_mem_u <= s2_updated;

            // Stage 3
            s3_valid <= s2_valid;
            s3_id    <= s2_id;
            s3_ts    <= s2_ts;
            if (s2_mem_u >= THRESHOLD) begin
                s3_spike    <= 1'b1;
                s3_mem_next <= RESET_VALUE;
            end else begin
                s3_spike    <= 1'b0;
                s3_mem_next <= s2_mem_u;
            end

            // Stage 4 output
            out_valid     <= s3_valid;
            out_neuron_id <= s3_id;
            out_timestamp <= s3_ts;
            membrane_out  <= s3_mem_next;
            out_spike     <= s3_spike;
        end
    end

endmodule
