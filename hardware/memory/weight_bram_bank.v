`timescale 1ns / 1ps

// Banked INT8 synaptic memory with simple 2-request arbitration.
// Requester 0 has priority when both request the same bank in the same cycle.
module weight_bram_bank #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 6,
    parameter NUM_BANKS  = 2,
    parameter BANK_SEL_W = 1
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // Optional config write port (used for initialization).
    input  wire                          wr_en,
    input  wire [BANK_SEL_W-1:0]         wr_bank,
    input  wire [ADDR_WIDTH-1:0]         wr_addr,
    input  wire [DATA_WIDTH-1:0]         wr_data,

    // Read requester 0
    input  wire                          req0_valid,
    input  wire [BANK_SEL_W-1:0]         req0_bank,
    input  wire [ADDR_WIDTH-1:0]         req0_addr,
    output reg                           req0_grant,
    output reg                           req0_data_valid,
    output reg  [DATA_WIDTH-1:0]         req0_data,

    // Read requester 1
    input  wire                          req1_valid,
    input  wire [BANK_SEL_W-1:0]         req1_bank,
    input  wire [ADDR_WIDTH-1:0]         req1_addr,
    output reg                           req1_grant,
    output reg                           req1_data_valid,
    output reg  [DATA_WIDTH-1:0]         req1_data
);
    localparam BANK_DEPTH  = (1 << ADDR_WIDTH);
    localparam TOTAL_DEPTH = NUM_BANKS * BANK_DEPTH;

    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:TOTAL_DEPTH-1];

    integer i;
    integer wr_idx;

    initial begin
        for (i = 0; i < TOTAL_DEPTH; i = i + 1) begin
            mem[i] = {DATA_WIDTH{1'b0}};
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req0_grant      <= 1'b0;
            req0_data_valid <= 1'b0;
            req0_data       <= {DATA_WIDTH{1'b0}};
            req1_grant      <= 1'b0;
            req1_data_valid <= 1'b0;
            req1_data       <= {DATA_WIDTH{1'b0}};
        end else begin
            req0_grant      <= 1'b0;
            req0_data_valid <= 1'b0;
            req1_grant      <= 1'b0;
            req1_data_valid <= 1'b0;

            if (wr_en) begin
                wr_idx = (wr_bank * BANK_DEPTH) + wr_addr;
                mem[wr_idx] <= wr_data;
            end

            if (req0_valid) begin
                req0_data <= mem[(req0_bank * BANK_DEPTH) + req0_addr];
                req0_data_valid <= 1'b1;
                req0_grant <= 1'b1;
            end

            if (req1_valid) begin
                if (!(req0_valid && (req0_bank == req1_bank))) begin
                    req1_data <= mem[(req1_bank * BANK_DEPTH) + req1_addr];
                    req1_data_valid <= 1'b1;
                    req1_grant <= 1'b1;
                end
            end
        end
    end

endmodule
