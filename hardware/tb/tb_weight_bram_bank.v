`timescale 1ns / 1ps

module tb_weight_bram_bank;
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n;
    reg wr_en;
    reg [0:0] wr_bank;
    reg [5:0] wr_addr;
    reg [7:0] wr_data;

    reg req0_valid;
    reg [0:0] req0_bank;
    reg [5:0] req0_addr;
    wire req0_grant;
    wire req0_data_valid;
    wire [7:0] req0_data;

    reg req1_valid;
    reg [0:0] req1_bank;
    reg [5:0] req1_addr;
    wire req1_grant;
    wire req1_data_valid;
    wire [7:0] req1_data;

    weight_bram_bank #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(6),
        .NUM_BANKS(2),
        .BANK_SEL_W(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_bank(wr_bank),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .req0_valid(req0_valid),
        .req0_bank(req0_bank),
        .req0_addr(req0_addr),
        .req0_grant(req0_grant),
        .req0_data_valid(req0_data_valid),
        .req0_data(req0_data),
        .req1_valid(req1_valid),
        .req1_bank(req1_bank),
        .req1_addr(req1_addr),
        .req1_grant(req1_grant),
        .req1_data_valid(req1_data_valid),
        .req1_data(req1_data)
    );

    initial begin
        rst_n = 0;
        wr_en = 0; wr_bank = 0; wr_addr = 0; wr_data = 0;
        req0_valid = 0; req0_bank = 0; req0_addr = 0;
        req1_valid = 0; req1_bank = 0; req1_addr = 0;

        repeat (2) @(posedge clk);
        rst_n <= 1;

        // Write two banks.
        @(posedge clk);
        wr_en <= 1; wr_bank <= 0; wr_addr <= 6'd1; wr_data <= 8'h12;
        @(posedge clk);
        wr_bank <= 1; wr_addr <= 6'd1; wr_data <= 8'h34;
        @(posedge clk);
        wr_en <= 0;

        // Parallel read from distinct banks.
        @(posedge clk);
        req0_valid <= 1; req0_bank <= 0; req0_addr <= 6'd1;
        req1_valid <= 1; req1_bank <= 1; req1_addr <= 6'd1;
        @(posedge clk);
        req0_valid <= 0; req1_valid <= 0;
        $display("Distinct-bank reads: req0=0x%0h req1=0x%0h", req0_data, req1_data);

        // Conflict read: requester 0 should win.
        @(posedge clk);
        req0_valid <= 1; req0_bank <= 0; req0_addr <= 6'd1;
        req1_valid <= 1; req1_bank <= 0; req1_addr <= 6'd1;
        @(posedge clk);
        req0_valid <= 0; req1_valid <= 0;
        $display("Conflict read: req0_grant=%0d req1_grant=%0d", req0_grant, req1_grant);

        #20;
        $finish;
    end
endmodule
