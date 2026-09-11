// =============================================================================
// Module: bram_bank_arbiter.v
// Architecture: ES-FA Synthesizable RTL
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Description: Dual-bank synchronous BRAM arbiter resolving concurrent access
//              between spike-driven PE weight reads (port 0) and STDP
//              writebacks (port 1). Bank assignment is fixed by address LSB
//              (parity interleaving), so requests aimed at different banks are
//              granted in the same cycle with no stall. When both ports target
//              the same bank, a flickering round-robin priority grants one
//              port per cycle; the losing port simply re-asserts its request
//              on the next cycle.
//
//              The design is a fixed two-bank fabric (bank = addr[0]); a
//              wider system instantiates one arbiter per core rather than
//              parameterizing bank count here, which keeps the grant logic
//              single-cycle and free of mux trees.
//
// Read timing contract (one outstanding request per port):
//   cycle 0: req*_valid asserted
//   cycle 1: req*_ready pulsed (granted; bank enable asserted for next cycle)
//   cycle 2: addressed bank latches its synchronous output
//   cycle 3: req*_dout registered and req*_dout_valid pulsed
// A new request on the same port must wait for the previous req*_dout_valid;
// issuing earlier would overwrite the in-flight tracker.
//
// Fix note: an earlier revision loaded req*_dout unconditionally every cycle,
// which meant the losing port of a same-bank collision observed corrupted
// data mid-flight. dout is now captured exactly once, only for a port whose
// read was actually granted, and is muxed by the bank captured at grant time.
// =============================================================================

`timescale 1ns / 1ps

module bram_bank_arbiter #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 10
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
    output reg                     req0_dout_valid,

    // Port 1: STDP / Host Update Interface (Weight Adaptation)
    input  wire                    req1_valid,
    input  wire                    req1_we,
    input  wire [ADDR_WIDTH-1:0]   req1_addr,
    input  wire [DATA_WIDTH-1:0]   req1_din,
    output reg  [DATA_WIDTH-1:0]   req1_dout,
    output reg                     req1_ready,
    output reg                     req1_dout_valid,

    // Physical BRAM Bank 0 Interface (even addresses)
    output reg                     bram0_en,
    output reg                     bram0_we,
    output reg  [ADDR_WIDTH-2:0]   bram0_addr,
    output reg  [DATA_WIDTH-1:0]   bram0_din,
    input  wire [DATA_WIDTH-1:0]   bram0_dout,

    // Physical BRAM Bank 1 Interface (odd addresses)
    output reg                     bram1_en,
    output reg                     bram1_we,
    output reg  [ADDR_WIDTH-2:0]   bram1_addr,
    output reg  [DATA_WIDTH-1:0]   bram1_din,
    input  wire [DATA_WIDTH-1:0]   bram1_dout
);

    wire target_bank0 = req0_addr[0];
    wire target_bank1 = req1_addr[0];

    // Round-robin priority state for same-bank collisions.
    reg priority_toggle;

    wire collision = req0_valid && req1_valid && (target_bank0 == target_bank1);
    wire grant0    = req0_valid && (!collision || (priority_toggle == 1'b0));
    wire grant1    = req1_valid && (!collision || (priority_toggle == 1'b1));

    // One-outstanding-read tracking per port. Stage 1 marks a granted read;
    // stage 2 marks the cycle in which the addressed bank's synchronous
    // output is valid, so dout is captured exactly once.
    reg req0_rd_pend1, req0_rd_pend2, req0_bank_pend;
    reg req1_rd_pend1, req1_rd_pend2, req1_bank_pend;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            priority_toggle <= 1'b0;
            req0_ready      <= 1'b0;
            req1_ready      <= 1'b0;
            req0_dout       <= {DATA_WIDTH{1'b0}};
            req1_dout       <= {DATA_WIDTH{1'b0}};
            req0_dout_valid <= 1'b0;
            req1_dout_valid <= 1'b0;
            req0_rd_pend1   <= 1'b0;
            req0_rd_pend2   <= 1'b0;
            req0_bank_pend  <= 1'b0;
            req1_rd_pend1   <= 1'b0;
            req1_rd_pend2   <= 1'b0;
            req1_bank_pend  <= 1'b0;
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
            req0_ready      <= 1'b0;
            req1_ready      <= 1'b0;
            req0_dout_valid <= 1'b0;
            req1_dout_valid <= 1'b0;

            // Advance the read trackers (defaults; grants override stage 1).
            req0_rd_pend2 <= req0_rd_pend1;
            req1_rd_pend2 <= req1_rd_pend1;
            req0_rd_pend1 <= 1'b0;
            req1_rd_pend1 <= 1'b0;

            if (collision) begin
                priority_toggle <= ~priority_toggle;
            end

            if (grant0) begin
                req0_ready <= 1'b1;
                if (!req0_we) begin
                    req0_rd_pend1  <= 1'b1;
                    req0_bank_pend <= target_bank0;
                end
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

            if (grant1) begin
                req1_ready <= 1'b1;
                if (!req1_we) begin
                    req1_rd_pend1  <= 1'b1;
                    req1_bank_pend <= target_bank1;
                end
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

            // Capture granted read data exactly once, muxed by the bank that
            // was latched at grant time (never by the current request's bank).
            if (req0_rd_pend2) begin
                req0_dout       <= req0_bank_pend ? bram1_dout : bram0_dout;
                req0_dout_valid <= 1'b1;
            end
            if (req1_rd_pend2) begin
                req1_dout       <= req1_bank_pend ? bram1_dout : bram0_dout;
                req1_dout_valid <= 1'b1;
            end
        end
    end

endmodule
