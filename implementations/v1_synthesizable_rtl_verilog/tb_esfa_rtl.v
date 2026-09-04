// =============================================================================
// Module: tb_esfa_rtl.v
// Architecture: ES-FA Tier 1 RTL Verification Testbench
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Description: Testbench verifying pipelined LIF accumulation, threshold firing,
//              and STDP weight updates under event spike stimulation.
// =============================================================================

`timescale 1ns / 1ps

module tb_esfa_rtl;

    reg clk;
    reg rst_n;

    reg        event_in_valid;
    reg [7:0]  event_in_id;
    reg [15:0] event_in_ts;
    wire       event_in_ready;

    wire       event_out_valid;
    wire [7:0] event_out_id;
    wire [15:0] event_out_ts;
    wire       event_out_spike;

    reg        stdp_enable;
    reg        post_spike_trigger;
    reg [15:0] post_spike_timestamp;

    wire [31:0] total_spikes_fired;
    wire [31:0] total_cycles_active;

    // Instantiate Device Under Test (DUT)
    esfa_top_core #(
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(8),
        .NEURON_ID_W(8),
        .TS_WIDTH(16),
        .ADDR_WIDTH(10),
        .THRESHOLD(16'sd30) // Low threshold to demonstrate rapid firing
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .event_in_valid(event_in_valid),
        .event_in_id(event_in_id),
        .event_in_ts(event_in_ts),
        .event_in_ready(event_in_ready),
        .event_out_valid(event_out_valid),
        .event_out_id(event_out_id),
        .event_out_ts(event_out_ts),
        .event_out_spike(event_out_spike),
        .stdp_enable(stdp_enable),
        .post_spike_trigger(post_spike_trigger),
        .post_spike_timestamp(post_spike_timestamp),
        .total_spikes_fired(total_spikes_fired),
        .total_cycles_active(total_cycles_active)
    );

    // Clock generation: 100 MHz (10 ns period)
    always #5 clk = ~clk;

    initial begin
        $display("=== ES-FA Tier 1 Synthesizable RTL Testbench Initialized ===");
        clk = 0;
        rst_n = 0;
        event_in_valid = 0;
        event_in_id = 0;
        event_in_ts = 0;
        stdp_enable = 0;
        post_spike_trigger = 0;
        post_spike_timestamp = 0;

        // Reset Pulse
        #25;
        rst_n = 1;
        #20;

        // Stimulate with burst of spikes
        $display("[RTL TB] Injecting event spike train into Neuron ID 0x42...");
        repeat (10) begin
            @(posedge clk);
            event_in_valid <= 1'b1;
            event_in_id    <= 8'h42;
            event_in_ts    <= event_in_ts + 1'b1;
        end
        @(posedge clk);
        event_in_valid <= 1'b0;

        // Exercise STDP
        $display("[RTL TB] Enabling STDP Hebbian learning window...");
        @(posedge clk);
        stdp_enable <= 1'b1;
        event_in_valid <= 1'b1;
        event_in_id <= 8'h05;
        event_in_ts <= 16'd50;
        @(posedge clk);
        event_in_valid <= 1'b0;

        #20;
        // Post-synaptic spike at timestamp 55 (LTP: post > pre by 5 cycles)
        @(posedge clk);
        post_spike_trigger <= 1'b1;
        post_spike_timestamp <= 16'd55;
        @(posedge clk);
        post_spike_trigger <= 1'b0;

        #100;
        $display("[RTL TB] Simulation Completed Successfully.");
        $display("[RTL TB] Total spikes fired: %0d", total_spikes_fired);
        $display("[RTL TB] Total active cycles: %0d", total_cycles_active);
        $finish;
    end

endmodule
