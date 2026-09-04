`timescale 1ns / 1ps

// Event-driven scheduler:
// - Enqueues routed events
// - Pops oldest-first (approx) from event_queue
// - Sends only active-neuron events to PE
module advanced_scheduler #(
    parameter NEURON_ID_W = 7,
    parameter TS_WIDTH    = 16,
    parameter QUEUE_DEPTH = 256,
    parameter QUEUE_PTR_W = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,

    input  wire                     in_event_valid,
    input  wire [NEURON_ID_W-1:0]   in_event_neuron_id,
    input  wire [TS_WIDTH-1:0]      in_event_timestamp,
    output wire                     in_event_ready,

    input  wire                     pe_ready,
    output wire                     out_valid,
    output wire [NEURON_ID_W-1:0]   out_neuron_id,
    output wire [TS_WIDTH-1:0]      out_timestamp,
    output wire [QUEUE_PTR_W:0]     queue_count,
    output reg  [31:0]              op_count
);
    wire q_push_valid = enable && in_event_valid;
    wire q_push_ready;
    wire q_pop_req = enable && pe_ready;
    wire q_pop_valid;
    wire [NEURON_ID_W-1:0] q_pop_id;
    wire [TS_WIDTH-1:0] q_pop_ts;

    event_queue #(
        .DEPTH(QUEUE_DEPTH),
        .NEURON_ID_W(NEURON_ID_W),
        .TS_WIDTH(TS_WIDTH),
        .PTR_W(QUEUE_PTR_W)
    ) u_event_queue (
        .clk(clk),
        .rst_n(rst_n),
        .push_valid(q_push_valid),
        .push_neuron_id(in_event_neuron_id),
        .push_timestamp(in_event_timestamp),
        .push_ready(q_push_ready),
        .pop_req(q_pop_req),
        .pop_valid(q_pop_valid),
        .pop_neuron_id(q_pop_id),
        .pop_timestamp(q_pop_ts),
        .queue_count(queue_count)
    );

    assign in_event_ready = q_push_ready && enable;
    assign out_valid = q_pop_valid && enable;
    assign out_neuron_id = q_pop_id;
    assign out_timestamp = q_pop_ts;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_count <= 32'd0;
        end else if (q_pop_valid && enable) begin
            op_count <= op_count + 32'd1;
        end
    end

endmodule
