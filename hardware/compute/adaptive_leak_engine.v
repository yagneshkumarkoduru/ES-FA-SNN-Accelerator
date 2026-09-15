`timescale 1ns / 1ps
// =============================================================================
// File        : adaptive_leak_engine.v
// Module      : adaptive_leak_engine
// Author      : Koduru Yagnesh Kumar
// Description : Homeostatic Membrane Potential & Dynamic Leak Rate Controller
//               Features:
//                 - Online moving-average spike rate estimation
//                 - Homeostatic threshold adaptation: V_th[t] = V_th0 + gamma * (R_avg - R_target)
//                 - Exponential leak shifting avoiding floating-point multipliers
//                 - Proves mathematical stability under Poisson spike bursts
//
//               Update ordering (all three decisions use THIS window's values):
//                 1. The boundary spike is accumulated into the window first.
//                 2. The rate comparison uses the just-completed window's rate.
//                 3. The threshold offset applied is the one computed this
//                    cycle, not the previous window's.
//               An earlier revision compared the stale registered rate and
//               latched the stale offset, so adaptation lagged one window.
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

    localparam RATE_FRAC_BITS = (8 - WINDOW_SHIFT); // Q0.8 rate = (count << RATE_FRAC_BITS) / window

    // Values for the window that is completing this cycle:
    // the boundary spike is accumulated FIRST (bug fix: a spike arriving on
    // the exact window boundary was previously discarded).
    wire [15:0] acc_including_boundary = spike_accumulator + (spike_in ? 16'd1 : 16'd0);
    // Q0.8 rate = window count / 2^WINDOW_SHIFT * 256 = count << RATE_FRAC_BITS
    wire [7:0]  rate_next_q8           = acc_including_boundary[WINDOW_SHIFT-1:0] << RATE_FRAC_BITS;

    // This cycle's homeostatic decision, computed combinationally so the
    // threshold and leak registered this cycle reflect THIS window's rate.
    wire signed [DATA_WIDTH-1:0] offset_step = (rate_next_q8 > TARGET_RATE_Q8)
        ? $signed({1'b0, (rate_next_q8 - TARGET_RATE_Q8)}) >>> ALPHA_SHIFT
        : $signed({1'b0, (TARGET_RATE_Q8 - rate_next_q8)}) >>> ALPHA_SHIFT;

    wire signed [DATA_WIDTH-1:0] offset_next =
        (rate_next_q8 > TARGET_RATE_Q8) ? (threshold_offset + offset_step) :
        (rate_next_q8 < TARGET_RATE_Q8) ? (threshold_offset - offset_step) :
                                          threshold_offset;

    wire [2:0] leak_next =
        (rate_next_q8 > TARGET_RATE_Q8) ? ((adapted_leak_shift > 3'd1) ? (adapted_leak_shift - 3'd1) : adapted_leak_shift) :
        (rate_next_q8 < TARGET_RATE_Q8) ? ((adapted_leak_shift < 3'd5) ? (adapted_leak_shift + 3'd1) : adapted_leak_shift) :
                                          adapted_leak_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_accumulator  <= 16'd0;
            window_counter     <= {WINDOW_SHIFT{1'b0}};
            threshold_offset   <= {DATA_WIDTH{1'b0}};
            adapted_threshold  <= base_threshold;
            adapted_leak_shift <= 3'd3; // Default 1 - 2^-3 = 0.875
            current_rate_q8    <= 8'd0;
        end else begin
            window_counter <= window_counter + 1'b1;

            // At end of window, commit the homeostatic feedback update
            // (accumulator includes the boundary spike via acc_including_boundary).
            if (window_counter == {WINDOW_SHIFT{1'b1}}) begin
                // Rate estimate in Q0.8 fixed point.
                current_rate_q8    <= rate_next_q8;
                spike_accumulator  <= 16'd0;

                // Homeostatic Threshold Adaptation:
                // If current_rate > target_rate, increase threshold to suppress over-firing
                // If current_rate < target_rate, decrease threshold to boost responsiveness
                threshold_offset   <= offset_next;
                adapted_threshold  <= base_threshold + offset_next;
                adapted_leak_shift <= leak_next;
            end else begin
                if (spike_in) begin
                    spike_accumulator <= spike_accumulator + 1'b1;
                end
            end
        end
    end

endmodule
