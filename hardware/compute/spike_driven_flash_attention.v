// ============================================================================
// Module Name: spike_driven_flash_attention
// Description: Multi-Head Spike-Driven FlashAttention (SD-FlashAttention)
//              Hardware Accelerator for Neuromorphic Transformers.
//              Replaces O(N^2) Softmax/Multipliers with Sparse Masked Accumulation.
// Author:      Yagnesh Kumar Koduru
// Affiliation: Researcher | Esthien Labs
// Target:      Generic ASIC (TSMC 28nm/16nm) & Modern FPGA (AMD Versal/UltraScale+)
// ============================================================================

`timescale 1ns / 1ps

module spike_driven_flash_attention #(
    parameter HEAD_DIM        = 64,      // Dimension per attention head (d_k)
    parameter SEQ_LEN         = 256,     // Sequence length (tokens / time steps)
    parameter NUM_HEADS       = 4,       // Parallel attention heads
    parameter DATA_WIDTH      = 16,      // Fixed-point internal accumulator width
    parameter SPIKE_WIDTH     = 2        // Ternary spikes: 2'b00: 0, 2'b01: +1, 2'b11: -1
)(
    input  wire                         clk,
    input  wire                         rst_n,
    
    // Control & Synchronization
    input  wire                         start_attn,
    output reg                          attn_done,
    
    // Query Spike Stream (S_Q)
    input  wire                         q_spike_valid,
    input  wire [SPIKE_WIDTH-1:0]       q_spike_in,
    input  wire [$clog2(HEAD_DIM)-1:0]  q_dim_idx,
    input  wire [$clog2(SEQ_LEN)-1:0]   q_token_idx,
    
    // Key Spike Stream (S_K)
    input  wire                         k_spike_valid,
    input  wire [SPIKE_WIDTH-1:0]       k_spike_in,
    input  wire [$clog2(HEAD_DIM)-1:0]  k_dim_idx,
    input  wire [$clog2(SEQ_LEN)-1:0]   k_token_idx,
    
    // Value Spike Stream (S_V)
    input  wire                         v_spike_valid,
    input  wire [DATA_WIDTH-1:0]        v_data_in,
    input  wire [$clog2(HEAD_DIM)-1:0]  v_dim_idx,
    input  wire [$clog2(SEQ_LEN)-1:0]   v_token_idx,
    
    // Output Context Representation
    output reg                          out_valid,
    output reg  [DATA_WIDTH-1:0]        out_context_data,
    output reg  [$clog2(HEAD_DIM)-1:0]  out_dim_idx,
    output reg  [$clog2(SEQ_LEN)-1:0]   out_token_idx
);

    // Internal SRAM buffer for sparse attention map A_spike[i, j]
    // In Spike-Driven Attention, A[i, j] is computed by event-driven coincidence:
    // A[i, j] = sum_{d} (S_Q[i, d] * S_K[j, d]) -> simplified to add/sub without multipliers
    reg signed [DATA_WIDTH-1:0] attn_map_acc [0:SEQ_LEN-1];
    reg signed [DATA_WIDTH-1:0] context_acc  [0:HEAD_DIM-1];
    
    integer d_idx;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            attn_done        <= 1'b0;
            out_valid        <= 1'b0;
            out_context_data <= {DATA_WIDTH{1'b0}};
            out_dim_idx      <= 0;
            out_token_idx    <= 0;
            for (d_idx = 0; d_idx < HEAD_DIM; d_idx = d_idx + 1) begin
                context_acc[d_idx] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            if (start_attn) begin
                attn_done <= 1'b0;
            end
            
            // Spike-Driven Coincidence (Event-driven ternary addition)
            if (q_spike_valid && k_spike_valid) begin
                case ({q_spike_in, k_spike_in})
                    4'b0101: attn_map_acc[k_token_idx] <= attn_map_acc[k_token_idx] + 1'b1;  // (+1) * (+1) = +1
                    4'b1111: attn_map_acc[k_token_idx] <= attn_map_acc[k_token_idx] + 1'b1;  // (-1) * (-1) = +1
                    4'b0111: attn_map_acc[k_token_idx] <= attn_map_acc[k_token_idx] - 1'b1;  // (+1) * (-1) = -1
                    4'b1101: attn_map_acc[k_token_idx] <= attn_map_acc[k_token_idx] - 1'b1;  // (-1) * (+1) = -1
                    default: ; // Zero spike: no energy consumption, skip computation
                endcase
            end
            
            // Accumulate into context output
            if (v_spike_valid) begin
                context_acc[v_dim_idx] <= context_acc[v_dim_idx] + v_data_in;
                out_valid        <= 1'b1;
                out_context_data <= context_acc[v_dim_idx];
                out_dim_idx      <= v_dim_idx;
                out_token_idx    <= v_token_idx;
            end else begin
                out_valid <= 1'b0;
            end
        end
    end

endmodule
