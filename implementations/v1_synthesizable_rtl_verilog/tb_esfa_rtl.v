// =============================================================================
// Module: tb_esfa_rtl.v
// Architecture: ES-FA Synthesizable RTL Verification Testbench
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Description: Testbench verifying pipelined LIF accumulation across events
//              (membrane state persistence), threshold firing through the
//              dual-bank weight arbiter, and STDP weight writeback under
//              event spike stimulation.
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
    wire [31:0] stdp_weight_updates;

    // Instantiate Device Under Test (DUT)
    esfa_top_core #(
        .DATA_WIDTH(16),
        .WEIGHT_WIDTH(8),
        .NEURON_ID_W(8),
        .TS_WIDTH(16),
        .ADDR_WIDTH(8),
        .THRESHOLD(16'sd30),          // Low threshold to demonstrate rapid firing
        .SYN_INIT_WEIGHT(8'sd18)      // Baseline synapse strength
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
        .total_cycles_active(total_cycles_active),
        .stdp_weight_updates(stdp_weight_updates)
    );

    // Clock generation: 100 MHz (10 ns period)
    always #5 clk = ~clk;

    // AXI-Stream style handshake: hold event_in_valid until event_in_ready.
    // After acceptance the stream is paced: the neuron-state
    // read-modify-write loop is 8 cycles (4-cycle weight read + 4-cycle PE
    // pipeline), so waiting 5 cycles after acceptance guarantees the next
    // event's membrane read observes the previous write-back. Event-driven
    // cores hide this latency by interleaving distinct neurons; this
    // single-PE demo core simply paces its input stream.
    task send_event(input [7:0] nid, input [15:0] ts);
        begin
            while (!event_in_ready) @(posedge clk);
            event_in_valid <= 1'b1;
            event_in_id    <= nid;
            event_in_ts    <= ts;
            @(posedge clk);
            event_in_valid <= 1'b0;
            repeat (5) @(posedge clk);
        end
    endtask

    integer k;

    initial begin
        $display("=== ES-FA Synthesizable RTL Testbench Initialized ===");
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
        // Consume the posedge pending in this timestep so the first
        // send_event handshake does not get its valid pulse cancelled by
        // the same edge it was scheduled on.
        @(posedge clk);

        // Stimulate with a burst of 10 events on Neuron ID 0x42.
        // With threshold 30 and baseline weight 18 the membrane traces
        // 18 -> 34 (spike, reset) -> 18 -> 34 ... so several spikes must fire.
        $display("[RTL TB] Injecting event spike train into Neuron ID 0x42...");
        for (k = 0; k < 10; k = k + 1) begin
            send_event(8'h42, k[15:0] + 16'd1);
        end

        repeat (10) @(posedge clk);
        $display("[RTL TB] Bursts complete. Spikes so far: %0d", total_spikes_fired);

        // Exercise STDP LTP. First event loads neuron 0x05's weight (18) into
        // the current-weight tracker; the second event is accepted while the
        // post-synaptic trigger asserts in the same cycle (pre ts=112,
        // post ts=124 -> dt=+12, inside the +/-32 cycle window), so the
        // engine writes back 18 + ALPHA_PLUS = 22 into the weight bank.
        $display("[RTL TB] Enabling STDP Hebbian learning window...");
        stdp_enable <= 1'b1;
        send_event(8'h05, 16'd100);
        repeat (6) @(posedge clk);

        while (!event_in_ready) @(posedge clk);
        event_in_valid      <= 1'b1;
        event_in_id         <= 8'h05;
        event_in_ts         <= 16'd112;
        post_spike_trigger  <= 1'b1;
        post_spike_timestamp<= 16'd124;
        @(posedge clk);
        event_in_valid      <= 1'b0;
        post_spike_trigger  <= 1'b0;

        // Allow the staged weight writeback (arbiter port 1) to land.
        repeat (20) @(posedge clk);
        // Neuron 0x05 (odd id) lives in bank 1 at address 0x05 >> 1 = 2.
        $display("[RTL TB] STDP-updated weight for neuron 0x05: %0d (expected 22)", $signed(dut.weight_bank1[2]));

        repeat (20) @(posedge clk);
        $display("[RTL TB] Simulation Completed.");
        $display("[RTL TB] Total spikes fired  : %0d", total_spikes_fired);
        $display("[RTL TB] Total active cycles : %0d", total_cycles_active);
        $display("[RTL TB] STDP weight updates : %0d", stdp_weight_updates);

        if (total_spikes_fired > 0 && stdp_weight_updates == 32'd1) begin
            $display("[RTL TB] PASS: membrane accumulation fired %0d spikes and STDP applied 1 weight update.", total_spikes_fired);
        end else begin
            $display("[RTL TB] FAIL: spikes=%0d (expected > 0), stdp_updates=%0d (expected 1)",
                     total_spikes_fired, stdp_weight_updates);
        end
        $finish;
    end

endmodule
