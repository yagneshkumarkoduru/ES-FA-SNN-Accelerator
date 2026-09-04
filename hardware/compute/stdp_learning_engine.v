// =============================================================================
// Module: stdp_learning_engine.v
// Project: ES-FA (Event-Driven Spiking FPGA Accelerator)
// Author: Yagnesh Kumar Koduru
// Domain: On-Chip Unsupervised Synaptic Plasticity, Neuromorphic Hardware
// Description: Synthesizable fixed-point Spike-Timing-Dependent Plasticity (STDP)
//              engine for local on-chip weight adaptation (LTP / LTD).
// =============================================================================

`timescale 1ns / 1ps

module stdp_learning_engine #(
    parameter WEIGHT_WIDTH = 8,
    parameter TIME_WIDTH   = 16,
    parameter ALPHA_PLUS   = 8'd4,     // LTP learning step
    parameter ALPHA_MINUS  = 8'd3,     // LTD learning step
    parameter TAU_WINDOW   = 16'd32    // Max temporal correlation window (cycles)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     pre_spike_valid,
    input  wire [TIME_WIDTH-1:0]    pre_spike_time,
    input  wire                     post_spike_valid,
    input  wire [TIME_WIDTH-1:0]    post_spike_time,
    input  wire signed [WEIGHT_WIDTH-1:0] current_weight,
    output reg  signed [WEIGHT_WIDTH-1:0] updated_weight,
    output reg                      weight_write_en
);

    reg signed [TIME_WIDTH:0] delta_t;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            updated_weight  <= 8'sd0;
            weight_write_en <= 1'b0;
            delta_t         <= {(TIME_WIDTH+1){1'b0}};
        end else begin
            weight_write_en <= 1'b0;

            if (pre_spike_valid && post_spike_valid) begin
                delta_t <= $signed({1'b0, post_spike_time}) - $signed({1'b0, pre_spike_time});

                // Long-Term Potentiation (LTP): post-spike occurs after pre-spike within window
                if (($signed({1'b0, post_spike_time}) > $signed({1'b0, pre_spike_time})) &&
                    (($signed({1'b0, post_spike_time}) - $signed({1'b0, pre_spike_time})) <= $signed({1'b0, TAU_WINDOW}))) begin
                    if (current_weight + $signed({1'b0, ALPHA_PLUS}) > 8'sd127)
                        updated_weight <= 8'sd127;
                    else
                        updated_weight <= current_weight + $signed({1'b0, ALPHA_PLUS});
                    weight_write_en <= 1'b1;
                end
                // Long-Term Depression (LTD): post-spike occurs before pre-spike within window
                else if (($signed({1'b0, post_spike_time}) < $signed({1'b0, pre_spike_time})) &&
                         (($signed({1'b0, pre_spike_time}) - $signed({1'b0, post_spike_time})) <= $signed({1'b0, TAU_WINDOW}))) begin
                    if (current_weight - $signed({1'b0, ALPHA_MINUS}) < -8'sd128)
                        updated_weight <= -8'sd128;
                    else
                        updated_weight <= current_weight - $signed({1'b0, ALPHA_MINUS});
                    weight_write_en <= 1'b1;
                end
            end
        end
    end

endmodule
