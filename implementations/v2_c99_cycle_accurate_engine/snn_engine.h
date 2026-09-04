#ifndef SNN_ENGINE_H
#define SNN_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CORES 16
#define MAX_NEURONS_PER_CORE 512
#define EVENT_QUEUE_CAPACITY 4096

typedef struct {
    uint16_t timestamp;
    uint8_t  core_id;
    uint16_t neuron_id;
} SpikeEvent;

typedef struct {
    int16_t  membrane_potential;
    int16_t  threshold;
    int16_t  reset_value;
    uint8_t  leak_shift;
    uint8_t  refractory_counter;
    uint16_t last_spike_time;
    uint32_t total_spikes_emitted;
} LIFNeuron;

typedef struct {
    int8_t   weight;
    uint16_t last_pre_ts;
    uint16_t last_post_ts;
} Synapse;

typedef struct {
    uint8_t    core_id;
    uint16_t   num_neurons;
    LIFNeuron  neurons[MAX_NEURONS_PER_CORE];
    Synapse    weights[MAX_NEURONS_PER_CORE][MAX_NEURONS_PER_CORE];

    // Banked SRAM Contention Tracking
    uint32_t   bank_accesses[2];
    uint32_t   bank_conflicts;

    // Local Event Queue
    SpikeEvent queue[EVENT_QUEUE_CAPACITY];
    int        queue_head;
    int        queue_tail;
    int        queue_count;

    // Activity metrics
    uint64_t   active_pe_cycles;
    uint64_t   idle_pe_cycles;
    uint64_t   synaptic_ops;
} SNNCore;

typedef struct {
    uint8_t    num_cores;
    uint16_t   neurons_per_core;
    bool       mode_event_driven; // 1: Event-driven ES-FA, 0: Basic Round-Robin
    bool       stdp_enabled;
    float      clock_freq_mhz;

    SNNCore    cores[MAX_CORES];

    // Global performance & energy telemetry
    uint64_t   global_cycle_count;
    uint64_t   total_input_spikes;
    uint64_t   total_output_spikes;
    uint64_t   total_synaptic_ops;
    uint64_t   sram_read_accesses;
    uint64_t   sram_write_accesses;
    uint64_t   router_hops;

    // Physical energy modeling (derived from 28nm/Kria characterization)
    double     energy_dynamic_nj;
    double     energy_static_nj;
    double     energy_total_nj;
    double     energy_delay_product; // Joules * Seconds
    double     throughput_gsops;
} SNNSystem;

// API Functions
void snn_system_init(SNNSystem *sys, uint8_t num_cores, uint16_t neurons_per_core, bool event_driven, bool stdp, float freq_mhz);
bool snn_enqueue_spike(SNNSystem *sys, uint8_t core_id, uint16_t neuron_id, uint16_t timestamp);
void snn_step_cycle(SNNSystem *sys);
void snn_run_timesteps(SNNSystem *sys, uint32_t num_timesteps, float input_spike_probability);
void snn_compute_telemetry(SNNSystem *sys);
void snn_print_report(const SNNSystem *sys);

#ifdef __cplusplus
}
#endif

#endif // SNN_ENGINE_H
