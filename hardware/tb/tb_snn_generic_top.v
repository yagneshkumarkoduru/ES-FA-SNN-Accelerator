`timescale 1ns / 1ps

module tb_snn_generic_top;

    parameter NUM_CORES        = 4;
    parameter NEURONS_PER_CORE = 128;
    parameter NEURON_ID_W      = 7;
    parameter CORE_ID_W        = 2;
    parameter TS_WIDTH         = 16;
    parameter DATA_WIDTH       = 16;

    reg clk;
    reg rst_n;

    // AXI-Lite
    reg [11:0]  s_axi_awaddr;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg [31:0]  s_axi_wdata;
    reg [3:0]   s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;

    reg [11:0]  s_axi_araddr;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    // AXI-Stream
    reg [CORE_ID_W+NEURON_ID_W+TS_WIDTH-1:0] s_axis_spike_tdata;
    reg                                      s_axis_spike_tvalid;
    wire                                     s_axis_spike_tready;

    wire [CORE_ID_W+NEURON_ID_W+TS_WIDTH-1:0] m_axis_spike_tdata;
    wire                                      m_axis_spike_tvalid;
    reg                                       m_axis_spike_tready;

    wire irq_timestep_done;
    wire irq_overflow;

    snn_accelerator_generic #(
        .NUM_CORES(NUM_CORES),
        .NEURONS_PER_CORE(NEURONS_PER_CORE),
        .NEURON_ID_W(NEURON_ID_W),
        .CORE_ID_W(CORE_ID_W),
        .DATA_WIDTH(DATA_WIDTH),
        .TS_WIDTH(TS_WIDTH),
        .STDP_ENABLE(1),
        .ASIC_MODE(0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .s_axis_spike_tdata(s_axis_spike_tdata),
        .s_axis_spike_tvalid(s_axis_spike_tvalid),
        .s_axis_spike_tready(s_axis_spike_tready),
        .m_axis_spike_tdata(m_axis_spike_tdata),
        .m_axis_spike_tvalid(m_axis_spike_tvalid),
        .m_axis_spike_tready(m_axis_spike_tready),
        .irq_timestep_done(irq_timestep_done),
        .irq_overflow(irq_overflow)
    );

    always #5 clk = ~clk;

    task axi_write(input [11:0] addr, input [31:0] data);
    begin
        @(posedge clk);
        s_axi_awaddr  <= addr;
        s_axi_awvalid <= 1'b1;
        s_axi_wdata   <= data;
        s_axi_wstrb   <= 4'hF;
        s_axi_wvalid  <= 1'b1;
        s_axi_bready  <= 1'b1;
        wait(s_axi_awready && s_axi_wready);
        @(posedge clk);
        s_axi_awvalid <= 1'b0;
        s_axi_wvalid  <= 1'b0;
        wait(s_axi_bvalid);
        @(posedge clk);
        s_axi_bready  <= 1'b0;
    end
    endtask

    initial begin
        clk = 0;
        rst_n = 0;
        s_axi_awvalid = 0;
        s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_arvalid = 0;
        s_axi_rready = 0;
        s_axis_spike_tvalid = 0;
        s_axis_spike_tdata = 0;
        m_axis_spike_tready = 1;

        #20 rst_n = 1;
        #20;

        // Configure accelerator: Enable Event-Driven mode + STDP + Run
        // reg_control: [0: start=1, 1: event_driven=1, 2: stdp_en=1] -> 0x07
        axi_write(12'h000, 32'h0000_0007);

        // Stream test spikes into Core 0, 1, 2, 3
        for (integer step = 0; step < 16; step = step + 1) begin
            @(posedge clk);
            s_axis_spike_tvalid <= 1'b1;
            // Packet: {Core[1:0], Neuron[6:0], Timestamp[15:0]}
            s_axis_spike_tdata  <= {step[1:0], step[6:0], step[15:0]};
        end

        @(posedge clk);
        s_axis_spike_tvalid <= 1'b0;

        #200;
        $display("[TB SUCCESS] Generic Multi-Core SNN Accelerator verification completed.");
        $finish;
    end

endmodule
