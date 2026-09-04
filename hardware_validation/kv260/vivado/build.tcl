proc arg_or_default {idx default} {
    if {$idx < [llength $::argv]} {
        return [lindex $::argv $idx]
    }
    return $default
}

set script_dir [file dirname [file normalize [info script]]]
set kv260_root [file normalize [file join $script_dir ".."]]
set project_root [file normalize [file join $kv260_root ".." ".."]]

set model_id [arg_or_default 0 "baseline_paper1"]
set scheduler_mode [arg_or_default 1 "dense"]
set out_dir [arg_or_default 2 [file join $project_root "results" "hardware_validation" $model_id $scheduler_mode "vivado_default"]]
set part_name [arg_or_default 3 "xczu3eg-sbva484-1-e"]
set top_name [arg_or_default 4 "snn_top"]
set clk_mhz [arg_or_default 5 "200.0"]

file mkdir $out_dir
set report_dir [file join $out_dir "reports"]
file mkdir $report_dir

set clk_period_ns [expr {1000.0 / double($clk_mhz)}]
set_param general.maxThreads 8

set rtl_files [list \
    [file join $project_root "hardware" "memory" "neuron_bram.v"] \
    [file join $project_root "hardware" "memory" "weight_bram_bank.v"] \
    [file join $project_root "hardware" "compute" "lif_neuron_pe.v"] \
    [file join $project_root "hardware" "routing" "spike_router.v"] \
    [file join $project_root "hardware" "scheduler" "basic_scheduler.v"] \
    [file join $project_root "hardware" "scheduler" "event_queue.v"] \
    [file join $project_root "hardware" "scheduler" "advanced_scheduler.v"] \
    [file join $project_root "hardware" "top" "snn_top.v"] \
]

read_verilog $rtl_files
read_xdc [file join $script_dir "constraints_kv260.xdc"]

# Out-of-context flow avoids board IO pin constraints while preserving
# architecture-level synthesis/implementation timing/utilization evidence.
synth_design -top $top_name -part $part_name -mode out_of_context -directive RuntimeOptimized
report_utilization -file [file join $report_dir "synth_utilization.rpt"]
report_timing_summary -delay_type max -file [file join $report_dir "synth_timing_summary.rpt"]
report_power -file [file join $report_dir "synth_power.rpt"]
write_checkpoint -force [file join $out_dir "synth_ooc.dcp"]

opt_design -directive RuntimeOptimized
place_design -directive RuntimeOptimized
phys_opt_design -directive RuntimeOptimized
route_design -directive RuntimeOptimized

report_utilization -file [file join $report_dir "impl_utilization.rpt"]
report_timing_summary -delay_type max -file [file join $report_dir "impl_timing_summary.rpt"]
report_power -file [file join $report_dir "impl_power.rpt"]
write_checkpoint -force [file join $out_dir "impl_ooc.dcp"]

set meta_fp [open [file join $out_dir "vivado_build_meta.txt"] w]
puts $meta_fp "model_id=$model_id"
puts $meta_fp "scheduler_mode=$scheduler_mode"
puts $meta_fp "part_name=$part_name"
puts $meta_fp "top_name=$top_name"
puts $meta_fp "clk_mhz=$clk_mhz"
puts $meta_fp "clk_period_ns=$clk_period_ns"
puts $meta_fp "flow_type=non_project_ooc"
close $meta_fp

exit

