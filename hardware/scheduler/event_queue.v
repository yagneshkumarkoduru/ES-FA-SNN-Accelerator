`timescale 1ns / 1ps

// Simplified BRAM-backed event queue:
// - Push into FIFO tail
// - On pop, compare first two entries by timestamp and select older one
//   (approximate priority without full heap complexity).
module event_queue #(
    parameter DEPTH       = 256,
    parameter NEURON_ID_W = 7,
    parameter TS_WIDTH    = 16,
    parameter PTR_W       = 8
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     push_valid,
    input  wire [NEURON_ID_W-1:0]   push_neuron_id,
    input  wire [TS_WIDTH-1:0]      push_timestamp,
    output wire                     push_ready,

    input  wire                     pop_req,
    output reg                      pop_valid,
    output reg  [NEURON_ID_W-1:0]   pop_neuron_id,
    output reg  [TS_WIDTH-1:0]      pop_timestamp,

    output wire [PTR_W:0]           queue_count
);
    (* ram_style = "block" *) reg [NEURON_ID_W-1:0] id_mem [0:DEPTH-1];
    (* ram_style = "block" *) reg [TS_WIDTH-1:0]    ts_mem [0:DEPTH-1];

    reg [PTR_W-1:0] head_ptr;
    reg [PTR_W-1:0] tail_ptr;
    reg [PTR_W:0]   count;

    wire full  = (count == DEPTH);
    wire empty = (count == 0);
    assign push_ready = !full;
    assign queue_count = count;

    wire do_push = push_valid && push_ready;
    wire do_pop = pop_req && !empty;

    wire [PTR_W-1:0] head_next = (head_ptr == DEPTH - 1) ? {PTR_W{1'b0}} : (head_ptr + 1'b1);
    wire second_valid = (count > 1);
    wire second_is_older = second_valid && (ts_mem[head_next] < ts_mem[head_ptr]);

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            id_mem[i] = {NEURON_ID_W{1'b0}};
            ts_mem[i] = {TS_WIDTH{1'b0}};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head_ptr      <= {PTR_W{1'b0}};
            tail_ptr      <= {PTR_W{1'b0}};
            count         <= {(PTR_W+1){1'b0}};
            pop_valid     <= 1'b0;
            pop_neuron_id <= {NEURON_ID_W{1'b0}};
            pop_timestamp <= {TS_WIDTH{1'b0}};
        end else begin
            pop_valid <= 1'b0;

            if (do_push) begin
                id_mem[tail_ptr] <= push_neuron_id;
                ts_mem[tail_ptr] <= push_timestamp;
                if (tail_ptr == DEPTH - 1) begin
                    tail_ptr <= {PTR_W{1'b0}};
                end else begin
                    tail_ptr <= tail_ptr + 1'b1;
                end
            end

            if (do_pop) begin
                pop_valid <= 1'b1;

                if (second_is_older) begin
                    // Remove second entry, keep head by moving it one slot forward.
                    pop_neuron_id <= id_mem[head_next];
                    pop_timestamp <= ts_mem[head_next];
                    id_mem[head_next] <= id_mem[head_ptr];
                    ts_mem[head_next] <= ts_mem[head_ptr];
                    head_ptr <= head_next;
                end else begin
                    pop_neuron_id <= id_mem[head_ptr];
                    pop_timestamp <= ts_mem[head_ptr];
                    head_ptr <= head_next;
                end
            end

            case ({do_push, do_pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule
