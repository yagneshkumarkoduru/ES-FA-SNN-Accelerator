# Generic clock constraint for implementation-quality timing reports.
# Board-level pin constraints can be overlaid for final KV260 bitstream runs.
create_clock -name clk -period 5.000 [get_ports clk]
set_clock_uncertainty 0.20 [get_clocks clk]

