`timescale 1ns / 1ps

// Spike event router:
// - Dense mode: forwards every input event
// - Sparse mode: forwards only events where in_spike = 1
module spike_router #(
    parameter NEURON_ID_W = 7,
    parameter TS_WIDTH    = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     mode_dense,
    input  wire                     in_valid,
    input  wire                     in_spike,
    input  wire [NEURON_ID_W-1:0]   in_neuron_id,
    input  wire [TS_WIDTH-1:0]      in_timestamp,

    output reg                      out_valid,
    output reg                      out_spike,
    output reg  [NEURON_ID_W-1:0]   out_neuron_id,
    output reg  [TS_WIDTH-1:0]      out_timestamp
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            out_spike     <= 1'b0;
            out_neuron_id <= {NEURON_ID_W{1'b0}};
            out_timestamp <= {TS_WIDTH{1'b0}};
        end else begin
            out_valid     <= in_valid && (mode_dense || in_spike);
            out_spike     <= in_spike;
            out_neuron_id <= in_neuron_id;
            out_timestamp <= in_timestamp;
        end
    end
endmodule
