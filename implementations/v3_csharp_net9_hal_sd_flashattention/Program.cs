// =============================================================================
// File: Program.cs
// Project: ES-FA Neuromorphic Accelerator (.NET 9 HAL & SD-FlashAttention)
// Author: Yagnesh Kumar Koduru (Esthien Labs)
// Architecture: .NET 9 Main Benchmark Console Runner
// =============================================================================

using System;
using ESFA.Driver;

namespace ESFA
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("====================================================================");
            Console.WriteLine("  ES-FA .NET 9 HAL DRIVER & SD-FLASHATTENTION BENCHMARK             ");
            Console.WriteLine("  Author: Yagnesh Kumar Koduru | Esthien Labs                       ");
            Console.WriteLine("====================================================================");

            // Part 1: HAL DMA Streaming Stress Test
            using var driver = new EsfaDriverNet9(numCores: 8, neuronsPerCore: 512);
            driver.Start();

            var telemetry = driver.BenchmarkThroughput(packetCount: 1_000_000);
            Console.WriteLine("\n--- [HAL DMA Ring Buffer Telemetry] ---");
            Console.WriteLine($"  Total Packets Streamed   : {telemetry.TotalPacketsStreamed:N0}");
            Console.WriteLine($"  Total Spikes Received    : {telemetry.TotalSpikesReceived:N0}");
            Console.WriteLine($"  Streaming Throughput     : {telemetry.ThroughputMpps:F2} Million Packets/sec");
            Console.WriteLine($"  Mean Inter-Packet Gap    : {telemetry.MeanInterPacketGapNs:F1} ns (includes queue wait)");
            Console.WriteLine($"  Mean Dispatch Latency    : {telemetry.MeanDispatchLatencyNs:F1} ns (measured around the dispatch call)");
            Console.WriteLine($"  Energy-Delay Product     : {telemetry.EnergyDelayProductModelJs:E2} J*s (MODEL estimate: 3.89 pJ/packet constant x dispatch latency)");

            // Part 2: Spike-Driven FlashAttention Kernel
            Console.WriteLine("\n--- [Spike-Driven FlashAttention Kernel Execution] ---");
            var sdFa = new SpikeDrivenFlashAttention();
            var (elapsedMs, bypassed, executed, reduction) = sdFa.RunBenchmark();

            Console.WriteLine($"  Kernel Execution Time    : {elapsedMs:F3} ms");
            Console.WriteLine($"  Operations Bypassed      : {bypassed:N0} ({(double)bypassed / (bypassed + executed) * 100.0:F2}%)");
            Console.WriteLine($"  Operations Executed      : {executed:N0}");
            Console.WriteLine($"  Theoretical Speedup      : {reduction:F1}x Energy Efficiency Gain");
            Console.WriteLine("====================================================================");
        }
    }
}
