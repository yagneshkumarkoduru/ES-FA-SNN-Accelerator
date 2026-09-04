`timescale 1ns / 1ps

// Dual-port BRAM for neuron state (e.g., membrane potential).
module neuron_bram #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 7,
    parameter INIT_FILE = ""
) (
    input  wire                     clk,
    input  wire                     we_a,
    input  wire [ADDR_WIDTH-1:0]    addr_a,
    input  wire [DATA_WIDTH-1:0]    din_a,
    output reg  [DATA_WIDTH-1:0]    dout_a,
    input  wire                     we_b,
    input  wire [ADDR_WIDTH-1:0]    addr_b,
    input  wire [DATA_WIDTH-1:0]    din_b,
    output reg  [DATA_WIDTH-1:0]    dout_b
);
    localparam DEPTH = (1 << ADDR_WIDTH);

    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    integer i;
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end else begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem[i] = {DATA_WIDTH{1'b0}};
            end
        end
    end

    always @(posedge clk) begin
        if (we_a) begin
            mem[addr_a] <= din_a;
        end
        dout_a <= mem[addr_a];

        if (we_b) begin
            mem[addr_b] <= din_b;
        end
        dout_b <= mem[addr_b];
    end

endmodule
