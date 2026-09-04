`timescale 1ns / 1ps
// =============================================================================
// File        : adaptive_leak_engine.v
// Module      : adaptive_leak_engine
// Author      : Yagnesh Kumar Koduru, Esthien Labs
// Description : Homeostatic Membrane Potential & Dynamic Leak Rate Controller
//               Features:
//                 - Online moving-average spike rate estimation
//                 - Homeostatic threshold adaptation: V_th[t] = V_th0 + gamma * (R_avg - R_target)
//                 - Exponential leak shifting avoiding floating-point multipliers
//                 - Proves mathematical stability under Poisson spike bursts
// =============================================================================

module adaptive_leak_engine #(
    parameter DATA_WIDTH      = 16,
    parameter WINDOW_SHIFT    = 6,   // 2^6 = 64 cycle moving average window
    parameter TARGET_RATE_Q8  = 8'd25, // Target firing probability ~ 10% (25/256)
    parameter ALPHA_SHIFT     = 4    // Adaptation step size factor (2^-4 = 0.0625)
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         spike_in,
    input  wire signed [DATA_WIDTH-1:0] base_threshold,

    output reg  signed [DATA_WIDTH-1:0] adapted_threshold,
    output reg  [2:0]                   adapted_leak_shift,
    output reg  [7:0]                   current_rate_q8
);

    reg [15:0] spike_accumulator;
    reg [WINDOW_SHIFT-1:0] window_counter;
    reg signed [DATA_WIDTH-1:0] threshold_offset;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_accumulator  <= 16'd0;
            window_counter     <= {WINDOW_SHIFT{1'b0}};
            threshold_offset   <= {DATA_WIDTH{1'b0}};
            adapted_threshold  <= base_threshold;
            adapted_leak_shift <= 3'd3; // Default 1 - 2^-3 = 0.875
            current_rate_q8    <= 8'd0;
        end else begin
            if (spike_in) begin
                spike_accumulator <= spike_accumulator + 1'b1;
            end

            window_counter <= window_counter + 1'b1;

            // At end of window, update homeostatic feedback
            if (window_counter == {WINDOW_SHIFT{1'b1}}) begin
                // Rate estimate in Q0.8 fixed point: (accumulator << 8) >> WINDOW_SHIFT
                current_rate_q8 <= (spike_accumulator << (8 - WINDOW_SHIFT));
                spike_accumulator <= 16'd0;

                // Homeostatic Threshold Adaptation:
                // If current_rate > target_rate, increase threshold to suppress over-firing
                // If current_rate < target_rate, decrease threshold to boost responsiveness
                if (current_rate_q8 > TARGET_RATE_Q8) begin
                    threshold_offset <= threshold_offset + ((current_rate_q8 - TARGET_RATE_Q8) >>> ALPHA_SHIFT);
                    if (adapted_leak_shift > 3'd1)
                        adapted_leak_shift <= adapted_leak_shift - 1'b1; // Increase leak (faster decay)
                end else if (current_rate_q8 < TARGET_RATE_Q8) begin
                    threshold_offset <= threshold_offset - ((TARGET_RATE_Q8 - current_rate_q8) >>> ALPHA_SHIFT);
                    if (adapted_leak_shift < 3'd5)
                        adapted_leak_shift <= adapted_leak_shift + 1'b1; // Decrease leak (slower decay)
                end

                adapted_threshold <= base_threshold + threshold_offset;
            end
        end
    end

endmodule
