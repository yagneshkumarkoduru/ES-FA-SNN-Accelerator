// =============================================================================
// Module: bram_bank_arbiter.v
// Architecture: ES-FA Tier 1 Synthesizable RTL
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Description: Multi-bank synchronous BRAM arbiter resolving concurrent
//              access conflicts between spike-driven PE reads and STDP writebacks.
// =============================================================================

`timescale 1ns / 1ps

module bram_bank_arbiter #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10,
    parameter NUM_BANKS  = 2
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Port 0: Scheduled Spike Read Interface (PEs)
    input  wire                    req0_valid,
    input  wire                    req0_we,
    input  wire [ADDR_WIDTH-1:0]   req0_addr,
    input  wire [DATA_WIDTH-1:0]   req0_din,
    output reg  [DATA_WIDTH-1:0]   req0_dout,
    output reg                     req0_ready,

    // Port 1: STDP / Host Update Interface (Weight Adaptation)
    input  wire                    req1_valid,
    input  wire                    req1_we,
    input  wire [ADDR_WIDTH-1:0]   req1_addr,
    input  wire [DATA_WIDTH-1:0]   req1_din,
    output reg  [DATA_WIDTH-1:0]   req1_dout,
    output reg                     req1_ready,

    // Physical BRAM Bank 0 Interface
    output reg                     bram0_en,
    output reg                     bram0_we,
    output reg  [ADDR_WIDTH-2:0]   bram0_addr,
    output reg  [DATA_WIDTH-1:0]   bram0_din,
    input  wire [DATA_WIDTH-1:0]   bram0_dout,

    // Physical BRAM Bank 1 Interface
    output reg                     bram1_en,
    output reg                     bram1_we,
    output reg  [ADDR_WIDTH-2:0]   bram1_addr,
    output reg  [DATA_WIDTH-1:0]   bram1_din,
    input  wire [DATA_WIDTH-1:0]   bram1_dout
);

    wire target_bank0 = req0_addr[0];
    wire target_bank1 = req1_addr[0];

    // Priority Arbiter state for simultaneous bank collisions
    reg priority_toggle;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            priority_toggle <= 1'b0;
            req0_ready      <= 1'b0;
            req1_ready      <= 1'b0;
            req0_dout       <= {DATA_WIDTH{1'b0}};
            req1_dout       <= {DATA_WIDTH{1'b0}};
            bram0_en        <= 1'b0;
            bram0_we        <= 1'b0;
            bram0_addr      <= {(ADDR_WIDTH-1){1'b0}};
            bram0_din       <= {DATA_WIDTH{1'b0}};
            bram1_en        <= 1'b0;
            bram1_we        <= 1'b0;
            bram1_addr      <= {(ADDR_WIDTH-1){1'b0}};
            bram1_din       <= {DATA_WIDTH{1'b0}};
        end else begin
            bram0_en <= 1'b0;
            bram0_we <= 1'b0;
            bram1_en <= 1'b0;
            bram1_we <= 1'b0;
            req0_ready <= 1'b0;
            req1_ready <= 1'b0;

            // Collision check: both requesting the same bank
            if (req0_valid && req1_valid && (target_bank0 == target_bank1)) begin
                if (priority_toggle == 1'b0) begin
                    // Grant Port 0
                    req0_ready <= 1'b1;
                    if (target_bank0 == 1'b0) begin
                        bram0_en   <= 1'b1;
                        bram0_we   <= req0_we;
                        bram0_addr <= req0_addr[ADDR_WIDTH-1:1];
                        bram0_din  <= req0_din;
                    end else begin
                        bram1_en   <= 1'b1;
                        bram1_we   <= req0_we;
                        bram1_addr <= req0_addr[ADDR_WIDTH-1:1];
                        bram1_din  <= req0_din;
                    end
                end else begin
                    // Grant Port 1
                    req1_ready <= 1'b1;
                    if (target_bank1 == 1'b0) begin
                        bram0_en   <= 1'b1;
                        bram0_we   <= req1_we;
                        bram0_addr <= req1_addr[ADDR_WIDTH-1:1];
                        bram0_din  <= req1_din;
                    end else begin
                        bram1_en   <= 1'b1;
                        bram1_we   <= req1_we;
                        bram1_addr <= req1_addr[ADDR_WIDTH-1:1];
                        bram1_din  <= req1_din;
                    end
                end
                priority_toggle <= ~priority_toggle;
            end else begin
                // No collision: parallel access to different banks
                if (req0_valid) begin
                    req0_ready <= 1'b1;
                    if (target_bank0 == 1'b0) begin
                        bram0_en   <= 1'b1;
                        bram0_we   <= req0_we;
                        bram0_addr <= req0_addr[ADDR_WIDTH-1:1];
                        bram0_din  <= req0_din;
                    end else begin
                        bram1_en   <= 1'b1;
                        bram1_we   <= req0_we;
                        bram1_addr <= req0_addr[ADDR_WIDTH-1:1];
                        bram1_din  <= req0_din;
                    end
                end
                if (req1_valid) begin
                    req1_ready <= 1'b1;
                    if (target_bank1 == 1'b0) begin
                        bram0_en   <= 1'b1;
                        bram0_we   <= req1_we;
                        bram0_addr <= req1_addr[ADDR_WIDTH-1:1];
                        bram0_din  <= req1_din;
                    end else begin
                        bram1_en   <= 1'b1;
                        bram1_we   <= req1_we;
                        bram1_addr <= req1_addr[ADDR_WIDTH-1:1];
                        bram1_din  <= req1_din;
                    end
                end
            end

            // Forward synchronous read data
            req0_dout <= (target_bank0 == 1'b0) ? bram0_dout : bram1_dout;
            req1_dout <= (target_bank1 == 1'b0) ? bram0_dout : bram1_dout;
        end
    end

endmodule
