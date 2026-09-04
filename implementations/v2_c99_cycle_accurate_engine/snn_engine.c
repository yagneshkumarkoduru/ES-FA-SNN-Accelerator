#include "snn_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STDP_TAU_PLUS   16
#define STDP_TAU_MINUS  20
#define STDP_A_PLUS     6
#define STDP_A_MINUS    4

void snn_system_init(SNNSystem *sys, uint8_t num_cores, uint16_t neurons_per_core, bool event_driven, bool stdp, float freq_mhz) {
    memset(sys, 0, sizeof(SNNSystem));
    sys->num_cores          = (num_cores > MAX_CORES) ? MAX_CORES : num_cores;
    sys->neurons_per_core   = (neurons_per_core > MAX_NEURONS_PER_CORE) ? MAX_NEURONS_PER_CORE : neurons_per_core;
    sys->mode_event_driven  = event_driven;
    sys->stdp_enabled       = stdp;
    sys->clock_freq_mhz     = (freq_mhz > 1.0f) ? freq_mhz : 250.0f;

    for (uint8_t c = 0; c < sys->num_cores; c++) {
        SNNCore *core = &sys->cores[c];
        core->core_id = c;
        core->num_neurons = sys->neurons_per_core;
        core->queue_head = 0;
        core->queue_tail = 0;
        core->queue_count = 0;

        for (uint16_t n = 0; n < core->num_neurons; n++) {
            LIFNeuron *nrn = &core->neurons[n];
            nrn->membrane_potential = 0;
            nrn->threshold = 64;       // 64 fixed-point LSBs
            nrn->reset_value = 0;
            nrn->leak_shift = 3;       // Leak decay factor: 1 - 2^-3 = 0.875
            nrn->refractory_counter = 0;
            nrn->last_spike_time = 0;
            nrn->total_spikes_emitted = 0;

            for (uint16_t dst = 0; dst < core->num_neurons; dst++) {
                // Initialize synthetic connection weights
                if ((n + dst) % 3 == 0) {
                    core->weights[n][dst].weight = (int8_t)((n * 7 + dst * 13) % 45 + 10);
                } else if ((n + dst) % 5 == 0) {
                    core->weights[n][dst].weight = (int8_t)-((n * 3 + dst * 5) % 30 + 5);
                } else {
                    core->weights[n][dst].weight = 0;
                }
                core->weights[n][dst].last_pre_ts = 0;
                core->weights[n][dst].last_post_ts = 0;
            }
        }
    }
}

bool snn_enqueue_spike(SNNSystem *sys, uint8_t core_id, uint16_t neuron_id, uint16_t timestamp) {
    if (core_id >= sys->num_cores || neuron_id >= sys->neurons_per_core) return false;
    SNNCore *core = &sys->cores[core_id];
    if (core->queue_count >= EVENT_QUEUE_CAPACITY) return false;

    core->queue[core->queue_tail].core_id = core_id;
    core->queue[core->queue_tail].neuron_id = neuron_id;
    core->queue[core->queue_tail].timestamp = timestamp;
    core->queue_tail = (core->queue_tail + 1) % EVENT_QUEUE_CAPACITY;
    core->queue_count++;
    sys->total_input_spikes++;
    return true;
}

static void apply_stdp(Synapse *syn, uint16_t pre_ts, uint16_t post_ts) {
    int32_t dt = (int32_t)post_ts - (int32_t)pre_ts;
    int32_t dw = 0;
    if (dt > 0 && dt <= 64) {
        // LTP (Long-Term Potentiation)
        dw = (int32_t)(STDP_A_PLUS * expf(-(float)dt / (float)STDP_TAU_PLUS));
        if (dw < 1) dw = 1;
    } else if (dt < 0 && dt >= -64) {
        // LTD (Long-Term Depression)
        int32_t abs_dt = -dt;
        dw = -(int32_t)(STDP_A_MINUS * expf(-(float)abs_dt / (float)STDP_TAU_MINUS));
        if (dw > -1) dw = -1;
    }

    int32_t new_w = (int32_t)syn->weight + dw;
    if (new_w > 127) new_w = 127;
    if (new_w < -128) new_w = -128;
    syn->weight = (int8_t)new_w;
}

void snn_step_cycle(SNNSystem *sys) {
    sys->global_cycle_count++;

    for (uint8_t c = 0; c < sys->num_cores; c++) {
        SNNCore *core = &sys->cores[c];

        if (sys->mode_event_driven) {
            // Event-Driven Execution: Only execute when spikes exist in event queue
            if (core->queue_count > 0) {
                SpikeEvent ev = core->queue[core->queue_head];
                core->queue_head = (core->queue_head + 1) % EVENT_QUEUE_CAPACITY;
                core->queue_count--;

                uint16_t src_nrn = ev.neuron_id;
                uint16_t ts      = ev.timestamp;

                // Propagate fanout synapses
                for (uint16_t dst = 0; dst < core->num_neurons; dst++) {
                    Synapse *syn = &core->weights[src_nrn][dst];
                    int8_t w = syn->weight;
                    if (w == 0) continue; // Zero weight skip

                    core->synaptic_ops++;
                    sys->total_synaptic_ops++;
                    sys->sram_read_accesses++;

                    // Bank contention model (Bank 0: even dst, Bank 1: odd dst)
                    uint8_t bank = dst & 1;
                    core->bank_accesses[bank]++;

                    LIFNeuron *post_nrn = &core->neurons[dst];
                    if (post_nrn->refractory_counter > 0) {
                        post_nrn->refractory_counter--;
                        continue;
                    }

                    // 4-Stage pipelined LIF integration
                    core->active_pe_cycles += 4;
                    int16_t v_leak = post_nrn->membrane_potential - (post_nrn->membrane_potential >> post_nrn->leak_shift);
                    int16_t v_next = v_leak + (int16_t)w;

                    if (v_next >= post_nrn->threshold) {
                        // Spike Fired!
                        post_nrn->membrane_potential = post_nrn->reset_value;
                        post_nrn->refractory_counter = 2;
                        post_nrn->last_spike_time = ts;
                        post_nrn->total_spikes_emitted++;
                        sys->total_output_spikes++;

                        // STDP adaptation
                        if (sys->stdp_enabled) {
                            apply_stdp(syn, ts, ts + 2);
                            sys->sram_write_accesses++;
                        }

                        // Inter-core or intra-core routing
                        if (dst % 7 == 0 && sys->num_cores > 1) {
                            uint8_t next_core = (c + 1) % sys->num_cores;
                            snn_enqueue_spike(sys, next_core, dst, ts + 4);
                            sys->router_hops++;
                        }
                    } else {
                        post_nrn->membrane_potential = v_next;
                    }
                }
            } else {
                // Clock-gated idle cycle
                core->idle_pe_cycles++;
            }
        } else {
            // Basic Round-Robin Synchronous Model: Evaluates all neurons every cycle
            for (uint16_t n = 0; n < core->num_neurons; n++) {
                core->active_pe_cycles++;
                LIFNeuron *nrn = &core->neurons[n];

                // Read synaptic row unconditionally
                sys->sram_read_accesses += 2;
                int16_t v_leak = nrn->membrane_potential - (nrn->membrane_potential >> nrn->leak_shift);
                nrn->membrane_potential = v_leak;

                if (core->queue_count > 0 && n == core->queue[core->queue_head].neuron_id) {
                    SpikeEvent ev = core->queue[core->queue_head];
                    (void)ev;
                    core->queue_head = (core->queue_head + 1) % EVENT_QUEUE_CAPACITY;
                    core->queue_count--;
                    core->synaptic_ops++;
                    sys->total_synaptic_ops++;
                    nrn->membrane_potential += 32;
                    if (nrn->membrane_potential >= nrn->threshold) {
                        nrn->membrane_potential = nrn->reset_value;
                        nrn->total_spikes_emitted++;
                        sys->total_output_spikes++;
                    }
                }
            }
        }
    }
}

void snn_run_timesteps(SNNSystem *sys, uint32_t num_timesteps, float input_spike_probability) {
    for (uint32_t t = 0; t < num_timesteps; t++) {
        // Inject random Poisson spikes based on input probability
        for (uint8_t c = 0; c < sys->num_cores; c++) {
            if (((float)rand() / (float)RAND_MAX) < input_spike_probability) {
                uint16_t nid = rand() % sys->neurons_per_core;
                snn_enqueue_spike(sys, c, nid, (uint16_t)(t & 0xFFFF));
            }
        }
        snn_step_cycle(sys);
    }
    snn_compute_telemetry(sys);
}

void snn_compute_telemetry(SNNSystem *sys) {
    double period_ns = 1000.0 / (double)sys->clock_freq_mhz;
    double total_time_s = (double)sys->global_cycle_count * period_ns * 1e-9;

    // Physical energy parameters (28nm standard cell / Kria UltraScale+)
    // Energy per active PE op: 1.25 pJ = 1.25e-3 nJ
    // Energy per idle PE cycle (clock gated): 0.04 pJ = 0.04e-3 nJ
    // Energy per un-gated PE cycle: 0.98 pJ = 0.98e-3 nJ
    // Energy per SRAM read (64-bit): 0.82 pJ = 0.82e-3 nJ
    // Energy per SRAM write: 0.95 pJ = 0.95e-3 nJ
    // Energy per router hop: 0.45 pJ = 0.45e-3 nJ
    // Static leakage power: 14.8 mW = 14.8e-3 W

    double e_pe = 0.0;
    for (uint8_t c = 0; c < sys->num_cores; c++) {
        if (sys->mode_event_driven) {
            e_pe += (double)sys->cores[c].active_pe_cycles * 1.25e-3;
            e_pe += (double)sys->cores[c].idle_pe_cycles * 0.04e-3;
        } else {
            e_pe += (double)sys->cores[c].active_pe_cycles * 0.98e-3;
        }
    }

    double e_sram = ((double)sys->sram_read_accesses * 0.82e-3) + ((double)sys->sram_write_accesses * 0.95e-3);
    double e_router = (double)sys->router_hops * 0.45e-3;

    sys->energy_dynamic_nj = e_pe + e_sram + e_router;
    sys->energy_static_nj  = 14.8 * (total_time_s * 1e9) * 1e-3; // mW * ns = pJ / 1000 = nJ
    sys->energy_total_nj   = sys->energy_dynamic_nj + sys->energy_static_nj;

    double energy_joules = sys->energy_total_nj * 1e-9;
    sys->energy_delay_product = energy_joules * total_time_s;

    if (total_time_s > 1e-12) {
        sys->throughput_gsops = ((double)sys->total_synaptic_ops / total_time_s) * 1e-9;
    } else {
        sys->throughput_gsops = 0.0;
    }
}

void snn_print_report(const SNNSystem *sys) {
    double period_ns = 1000.0 / (double)sys->clock_freq_mhz;
    double total_time_us = (double)sys->global_cycle_count * period_ns * 1e-3;
    double pJ_per_sop = (sys->total_synaptic_ops > 0) ? (sys->energy_dynamic_nj * 1e3 / (double)sys->total_synaptic_ops) : 0.0;

    printf("\n==================================================================\n");
    printf("     ES-FA NEUROMORPHIC ACCELERATOR -- EXECUTION TELEMETRY REPORT\n");
    printf("==================================================================\n");
    printf(" Architecture Config  : %u Cores x %u Neurons/Core (%u Total Neurons)\n",
           sys->num_cores, sys->neurons_per_core, sys->num_cores * sys->neurons_per_core);
    printf(" Execution Mode       : %s\n", sys->mode_event_driven ? "EVENT-DRIVEN (ES-FA)" : "BASIC ROUND-ROBIN");
    printf(" STDP Plasticity      : %s\n", sys->stdp_enabled ? "ENABLED (On-Chip Learning)" : "DISABLED (Inference Only)");
    printf(" Clock Frequency      : %.2f MHz (Period: %.2f ns)\n", sys->clock_freq_mhz, period_ns);
    printf(" Total Elapsed Cycles : %llu cycles (%.3f us)\n", (unsigned long long)sys->global_cycle_count, total_time_us);
    printf(" Input Spikes Ingested: %llu\n", (unsigned long long)sys->total_input_spikes);
    printf(" Output Spikes Fired  : %llu\n", (unsigned long long)sys->total_output_spikes);
    printf(" Total Synaptic Ops   : %llu SOPs\n", (unsigned long long)sys->total_synaptic_ops);
    printf(" Peak Throughput      : %.4f GSOP/s\n", sys->throughput_gsops);
    printf(" SRAM Read / Write    : %llu / %llu accesses\n", (unsigned long long)sys->sram_read_accesses, (unsigned long long)sys->sram_write_accesses);
    printf(" Inter-Core Router Hops: %llu\n", (unsigned long long)sys->router_hops);
    printf(" Dynamic Energy       : %.4f nJ (%.2f pJ / Synaptic OP)\n", sys->energy_dynamic_nj, pJ_per_sop);
    printf(" Static Leakage Energy: %.4f nJ\n", sys->energy_static_nj);
    printf(" Total Energy (E)     : %.4f nJ (%.4f uJ)\n", sys->energy_total_nj, sys->energy_total_nj * 1e-3);
    printf(" Energy-Delay Product : %.6e J*s\n", sys->energy_delay_product);
    printf("==================================================================\n\n");
}
