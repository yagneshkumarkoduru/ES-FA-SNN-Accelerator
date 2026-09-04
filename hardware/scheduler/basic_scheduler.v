`timescale 1ns / 1ps

// Basic round-robin scheduler. Processes one neuron per accepted input event.
module basic_scheduler #(
    parameter NUM_NEURONS = 128,
    parameter NEURON_ID_W = 7,
    parameter TS_WIDTH    = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     enable,
    input  wire                     in_valid,
    input  wire [TS_WIDTH-1:0]      in_timestamp,
    input  wire                     pe_ready,

    output reg                      out_valid,
    output reg  [NEURON_ID_W-1:0]   out_neuron_id,
    output reg  [TS_WIDTH-1:0]      out_timestamp,
    output reg  [31:0]              op_count
);
    reg [NEURON_ID_W-1:0] rr_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr_ptr        <= {NEURON_ID_W{1'b0}};
            out_valid     <= 1'b0;
            out_neuron_id <= {NEURON_ID_W{1'b0}};
            out_timestamp <= {TS_WIDTH{1'b0}};
            op_count      <= 32'd0;
        end else begin
            out_valid <= 1'b0;

            if (enable && in_valid && pe_ready) begin
                out_valid     <= 1'b1;
                out_neuron_id <= rr_ptr;
                out_timestamp <= in_timestamp;
                op_count      <= op_count + 32'd1;

                if (rr_ptr == NUM_NEURONS - 1) begin
                    rr_ptr <= {NEURON_ID_W{1'b0}};
                end else begin
                    rr_ptr <= rr_ptr + 1'b1;
                end
            end
        end
    end

endmodule
