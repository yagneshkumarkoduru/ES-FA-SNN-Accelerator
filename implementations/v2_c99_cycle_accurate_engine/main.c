#include "snn_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    uint8_t  cores       = 4;
    uint16_t neurons     = 128;
    uint32_t steps       = 10000;
    float    sparsity    = 0.85f; // 85% sparse (15% spike probability)
    float    freq_mhz    = 250.0f;
    const char *json_out = "c_benchmark_results.json";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cores") == 0 && i + 1 < argc) {
            cores = (uint8_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--neurons") == 0 && i + 1 < argc) {
            neurons = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc) {
            sparsity = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }

    srand(42); // Deterministic seed for reproducible research evaluation

    printf("[ES-FA Neuromorphic Simulation Engine]\n");
    printf("Initializing research benchmark: %u Cores, %u Neurons/Core, %u Timesteps, Sparsity: %.1f%%\n",
           cores, neurons, steps, sparsity * 100.0f);

    float spike_prob = 1.0f - sparsity;

    // Allocate systems on Heap to avoid stack overflow
    SNNSystem *sys_basic = (SNNSystem*)malloc(sizeof(SNNSystem));
    SNNSystem *sys_event = (SNNSystem*)malloc(sizeof(SNNSystem));
    SNNSystem *sys_stdp  = (SNNSystem*)malloc(sizeof(SNNSystem));

    if (!sys_basic || !sys_event || !sys_stdp) {
        fprintf(stderr, "Error: Memory allocation failed for SNN systems\n");
        return 1;
    }

    // 1. Basic Round-Robin Run
    snn_system_init(sys_basic, cores, neurons, false, false, freq_mhz);
    snn_run_timesteps(sys_basic, steps, spike_prob);

    // 2. Event-Driven ES-FA Run (Inference)
    snn_system_init(sys_event, cores, neurons, true, false, freq_mhz);
    snn_run_timesteps(sys_event, steps, spike_prob);

    // 3. Event-Driven ES-FA Run with STDP On-Chip Learning
    snn_system_init(sys_stdp, cores, neurons, true, true, freq_mhz);
    snn_run_timesteps(sys_stdp, steps, spike_prob);

    // Print reports
    printf("\n>>> BASELINE SYNCHRONOUS ROUND-ROBIN ACCELERATOR <<<\n");
    snn_print_report(sys_basic);

    printf("\n>>> ES-FA EVENT-DRIVEN ACCELERATOR (INFERENCE ONLY) <<<\n");
    snn_print_report(sys_event);

    printf("\n>>> ES-FA EVENT-DRIVEN ACCELERATOR (ON-CHIP STDP LEARNING) <<<\n");
    snn_print_report(sys_stdp);

    // Compute comparative metrics
    double energy_savings_pct = (1.0 - (sys_event->energy_dynamic_nj / sys_basic->energy_dynamic_nj)) * 100.0;
    double edp_reduction_x    = sys_basic->energy_delay_product / sys_event->energy_delay_product;

    printf("==================================================================\n");
    printf("                COMPARATIVE RESEARCH BREAKTHROUGHS                \n");
    printf("==================================================================\n");
    printf(" Dynamic Energy Savings : %.2f%%\n", energy_savings_pct);
    printf(" EDP Reduction Factor   : %.2fx\n", edp_reduction_x);
    printf(" Baseline EDP (J*s)     : %.6e\n", sys_basic->energy_delay_product);
    printf(" ES-FA EDP (J*s)        : %.6e\n", sys_event->energy_delay_product);
    printf(" ES-FA + STDP EDP (J*s) : %.6e\n", sys_stdp->energy_delay_product);
    printf("==================================================================\n\n");

    // Export JSON file for paper figures & tables
    FILE *fp = fopen(json_out, "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"cores\": %u,\n", cores);
        fprintf(fp, "  \"neurons_per_core\": %u,\n", neurons);
        fprintf(fp, "  \"timesteps\": %u,\n", steps);
        fprintf(fp, "  \"sparsity_pct\": %.2f,\n", sparsity * 100.0f);
        fprintf(fp, "  \"clock_freq_mhz\": %.2f,\n", freq_mhz);
        fprintf(fp, "  \"basic\": {\n");
        fprintf(fp, "    \"cycles\": %llu,\n", (unsigned long long)sys_basic->global_cycle_count);
        fprintf(fp, "    \"dynamic_energy_nj\": %.4f,\n", sys_basic->energy_dynamic_nj);
        fprintf(fp, "    \"static_energy_nj\": %.4f,\n", sys_basic->energy_static_nj);
        fprintf(fp, "    \"total_energy_nj\": %.4f,\n", sys_basic->energy_total_nj);
        fprintf(fp, "    \"edp_js\": %.6e\n", sys_basic->energy_delay_product);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"es_fa_event\": {\n");
        fprintf(fp, "    \"cycles\": %llu,\n", (unsigned long long)sys_event->global_cycle_count);
        fprintf(fp, "    \"dynamic_energy_nj\": %.4f,\n", sys_event->energy_dynamic_nj);
        fprintf(fp, "    \"static_energy_nj\": %.4f,\n", sys_event->energy_static_nj);
        fprintf(fp, "    \"total_energy_nj\": %.4f,\n", sys_event->energy_total_nj);
        fprintf(fp, "    \"edp_js\": %.6e,\n", sys_event->energy_delay_product);
        fprintf(fp, "    \"energy_savings_pct\": %.2f,\n", energy_savings_pct);
        fprintf(fp, "    \"edp_reduction_x\": %.2f\n", edp_reduction_x);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"es_fa_stdp\": {\n");
        fprintf(fp, "    \"cycles\": %llu,\n", (unsigned long long)sys_stdp->global_cycle_count);
        fprintf(fp, "    \"dynamic_energy_nj\": %.4f,\n", sys_stdp->energy_dynamic_nj);
        fprintf(fp, "    \"static_energy_nj\": %.4f,\n", sys_stdp->energy_static_nj);
        fprintf(fp, "    \"total_energy_nj\": %.4f,\n", sys_stdp->energy_total_nj);
        fprintf(fp, "    \"edp_js\": %.6e\n", sys_stdp->energy_delay_product);
        fprintf(fp, "  }\n");
        fprintf(fp, "}\n");
        fclose(fp);
        printf("[INFO] Benchmark metrics exported to %s successfully.\n", json_out);
    }

    free(sys_basic);
    free(sys_event);
    free(sys_stdp);

    return 0;
}
