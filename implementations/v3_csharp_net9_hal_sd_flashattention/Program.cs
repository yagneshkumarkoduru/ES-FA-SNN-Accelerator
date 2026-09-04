// =============================================================================
// File: Program.cs
// Project: ES-FA Neuromorphic Accelerator (Tier 3 Implementation)
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Architecture: .NET 9 Main Benchmark Console Runner
// =============================================================================

using System;
using ESFA.Tier3.Driver;

namespace ESFA.Tier3
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("====================================================================");
            Console.WriteLine("  ES-FA TIER 3: .NET 9 HAL DRIVER & SD-FLASHATTENTION BENCHMARK     ");
            Console.WriteLine("  Author: Yagnesh Kumar Koduru | Esthien Labs                       ");
            Console.WriteLine("====================================================================");

            // Part 1: HAL DMA Streaming Stress Test
            using var driver = new EsfaDriverNet9(numCores: 8, neuronsPerCore: 512);
            driver.Start();

            var telemetry = driver.BenchmarkThroughput(packetCount: 1_000_000);
            Console.WriteLine("\n--- [HAL DMA Ring Buffer Telemetry] ---");
            Console.WriteLine($"  Total Packets Streamed : {telemetry.TotalPacketsStreamed:N0}");
            Console.WriteLine($"  Total Spikes Received  : {telemetry.TotalSpikesReceived:N0}");
            Console.WriteLine($"  Streaming Throughput   : {telemetry.ThroughputMpps:F2} Million Packets/sec");
            Console.WriteLine($"  Average Dispatch Lat   : {telemetry.AverageLatencyNs:F1} ns");
            Console.WriteLine($"  Energy-Delay Product   : {telemetry.EnergyDelayProductJs:E2} J*s");

            // Part 2: Spike-Driven FlashAttention SIMD Kernel
            Console.WriteLine("\n--- [Spike-Driven FlashAttention Kernel Execution] ---");
            var sdFa = new SpikeDrivenFlashAttention();
            var (elapsedMs, bypassed, executed, reduction) = sdFa.RunBenchmark();

            Console.WriteLine($"  Kernel Execution Time  : {elapsedMs:F3} ms");
            Console.WriteLine($"  Operations Bypassed    : {bypassed:N0} ({(double)bypassed / (bypassed + executed) * 100.0:F2}%)");
            Console.WriteLine($"  Operations Executed    : {executed:N0}");
            Console.WriteLine($"  Theoretical Speedup    : {reduction:F1}x Energy Efficiency Gain");
            Console.WriteLine("====================================================================");
        }
    }
}
