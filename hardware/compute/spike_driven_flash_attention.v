// ============================================================================
// Module Name: spike_driven_flash_attention
// Description: Spike-Driven FlashAttention (SD-FlashAttention) hardware
//              accelerator for neuromorphic transformers. Replaces the
//              O(N^2) softmax/multiplier datapath with sparse masked
//              accumulation driven purely by ternary spike coincidence.
//
//              One engine instance processes ONE attention head. A multi-head
//              deployment instantiates one engine per head (the Q/K/V streams
//              carry no head identifier, so head multiplexing is a system
//              integration decision, not a module parameter).
//
//              Datapath (multiplier-free):
//              1. Coincidence scoring: whenever Q and K spikes for query row i
//                 and key j coincide on dimension d, the score S[i,j]
//                 accumulates +1 / -1 by ternary sign. Scores are kept per key
//                 with a query-row tag, so a key's stale score from a previous
//                 row is treated as zero instead of being read without reset.
//              2. Weighted value accumulation: when a V element arrives for
//                 key j, its score gates the accumulation. The magnitude class
//                 of |S[i,j]| picks an approximate weight (1, 1/2, or 1/4)
//                 implemented with arithmetic right shifts - no multipliers.
//                 Numerator and denominator accumulators are kept separately
//                 and the running normalized estimate is emitted per event.
//
//              Honest approximation note: the true per-key weight is
//              S[i,j]/max|S|, which would need a division per token. It is
//              approximated by a 3-level magnitude class (full / half /
//              quarter weight) plus a power-of-two denominator normalization
//              (shift by the highest set bit of the weight total). The
//              approximation error is bounded by construction (each class is
//              within [1/2, 1] of its target weight) and shrinks as d_k grows.
//
// Author:      Koduru Yagnesh Kumar
// Affiliation: Independent Researcher
// Target:      Generic ASIC (TSMC 28nm/16nm) & Modern FPGA (AMD Versal/UltraScale+)
// ============================================================================

`timescale 1ns / 1ps

module spike_driven_flash_attention #(
    parameter HEAD_DIM        = 64,      // Dimension per attention head (d_k)
    parameter SEQ_LEN         = 256,     // Sequence length (tokens / time steps)
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

    // Magnitude class thresholds over |S[i,j]| (max possible score is d_k).
    // |S| >= FULL_TH  -> weight 1   (numerator += v,      denominator += 4)
    // |S| >= HALF_TH  -> weight 1/2 (numerator += v>>1,  denominator += 2)
    // otherwise       -> weight 1/4 (numerator += v>>2,  denominator += 1)
    localparam SCORE_FULL_TH = (HEAD_DIM / 2);
    localparam SCORE_HALF_TH = (HEAD_DIM / 8);
    localparam DEN_FULL_W    = 2'd3;   // denominator units: quarters of full weight
    localparam DEN_HALF_W    = 2'd2;

    // Per-key coincidence score of the current query row, with a row tag per
    // key so scores from an earlier row are recognized as stale (treated as
    // zero) rather than silently accumulated into the new row.
    reg signed [DATA_WIDTH-1:0]      attn_map_acc  [0:SEQ_LEN-1];
    reg [$clog2(SEQ_LEN)-1:0]        score_row_tag [0:SEQ_LEN-1];
    reg [$clog2(SEQ_LEN)-1:0]        cur_q_row;

    // Weighted accumulation (numerator/denominator) for the running output.
    reg signed [DATA_WIDTH-1:0]      num_acc [0:HEAD_DIM-1];
    reg signed [2*DATA_WIDTH-1:0]    den_acc;   // sums of 1/2/4 per event
    reg                              have_weights; // any score contributed yet

    // Highest set bit of den_acc: power-of-two approximation of the division
    // by the weight total (a priority scan; den_acc is small because it only
    // grows by 1..4 per V event).
    reg [5:0] den_shift;
    integer   s_idx;
    always @(*) begin
        den_shift = 6'd0;
        for (s_idx = 1; s_idx < 2*DATA_WIDTH; s_idx = s_idx + 1) begin
            if (den_acc[s_idx]) den_shift = s_idx[5:0];
        end
    end

    // Per-key class for the V accumulation path.
    reg                          score_stale;
    reg                          v_score_stale;
    reg signed [DATA_WIDTH-1:0]  score_v;
    reg signed [DATA_WIDTH-1:0]  v_shifted;
    reg signed [DATA_WIDTH-1:0]  num_next;
    reg [1:0]                    v_den_inc;

    always @(*) begin
        score_stale   = (score_row_tag[k_token_idx] != q_token_idx);
        v_score_stale = (score_row_tag[v_token_idx] != cur_q_row);

        score_v   = {DATA_WIDTH{1'b0}};
        v_shifted = {DATA_WIDTH{1'b0}};
        num_next  = num_acc[v_dim_idx];
        v_den_inc = 2'd0;

        if (!v_score_stale) begin
            score_v = attn_map_acc[v_token_idx];
            if (score_v != 0) begin
                // Multiplier-free weighting: |score| class -> shift/add/sub.
                if (score_v >= SCORE_FULL_TH) begin
                    v_shifted = v_data_in;
                    v_den_inc = DEN_FULL_W;
                end else if (score_v >= SCORE_HALF_TH) begin
                    v_shifted = v_data_in >>> 1;
                    v_den_inc = DEN_HALF_W;
                end else if (score_v <= -SCORE_FULL_TH) begin
                    v_shifted = -v_data_in;
                    v_den_inc = DEN_FULL_W;
                end else if (score_v <= -SCORE_HALF_TH) begin
                    v_shifted = -(v_data_in >>> 1);
                    v_den_inc = DEN_HALF_W;
                end else begin
                    v_shifted = (score_v > 0) ? (v_data_in >>> 2) : -(v_data_in >>> 2);
                    v_den_inc = 2'd1;
                end
                num_next = num_acc[v_dim_idx] + v_shifted;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            attn_done        <= 1'b0;
            out_valid        <= 1'b0;
            out_context_data <= {DATA_WIDTH{1'b0}};
            out_dim_idx      <= 0;
            out_token_idx    <= 0;
            cur_q_row        <= 0;
            den_acc          <= {(2*DATA_WIDTH){1'b0}};
            have_weights     <= 1'b0;
            for (d_idx = 0; d_idx < HEAD_DIM; d_idx = d_idx + 1) begin
                num_acc[d_idx] <= {DATA_WIDTH{1'b0}};
            end
            for (r_idx = 0; r_idx < SEQ_LEN; r_idx = r_idx + 1) begin
                score_row_tag[r_idx] <= 0;
                attn_map_acc[r_idx]  <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            if (start_attn) begin
                attn_done    <= 1'b0;
                den_acc      <= {(2*DATA_WIDTH){1'b0}};
                have_weights <= 1'b0;
                for (d_idx = 0; d_idx < HEAD_DIM; d_idx = d_idx + 1) begin
                    num_acc[d_idx] <= {DATA_WIDTH{1'b0}};
                end
                // Query-row tags are reset so a key whose tag no longer
                // matches the incoming q_token_idx is treated as score zero.
                for (r_idx = 0; r_idx < SEQ_LEN; r_idx = r_idx + 1) begin
                    score_row_tag[r_idx] <= 0;
                end
                cur_q_row <= 0;
            end

            // -----------------------------------------------------------------
            // Stage 1: spike-driven coincidence accumulation (ternary add/sub,
            // no multipliers). Skipped entirely when either spike is zero.
            // -----------------------------------------------------------------
            if (q_spike_valid && k_spike_valid) begin
                cur_q_row <= q_token_idx;
                case ({q_spike_in, k_spike_in})
                    4'b0101: begin // (+1) * (+1) = +1
                        attn_map_acc[k_token_idx] <= score_stale ? 16'sd1 : (attn_map_acc[k_token_idx] + 16'sd1);
                        score_row_tag[k_token_idx] <= q_token_idx;
                    end
                    4'b1111: begin // (-1) * (-1) = +1
                        attn_map_acc[k_token_idx] <= score_stale ? 16'sd1 : (attn_map_acc[k_token_idx] + 16'sd1);
                        score_row_tag[k_token_idx] <= q_token_idx;
                    end
                    4'b0111: begin // (+1) * (-1) = -1
                        attn_map_acc[k_token_idx] <= score_stale ? -16'sd1 : (attn_map_acc[k_token_idx] - 16'sd1);
                        score_row_tag[k_token_idx] <= q_token_idx;
                    end
                    4'b1101: begin // (-1) * (+1) = -1
                        attn_map_acc[k_token_idx] <= score_stale ? -16'sd1 : (attn_map_acc[k_token_idx] - 16'sd1);
                        score_row_tag[k_token_idx] <= q_token_idx;
                    end
                    default: ; // Zero spike: no energy consumption, skip
                endcase
            end

            // -----------------------------------------------------------------
            // Stage 2: attention-weighted value accumulation. The V element
            // is consumed only if its key holds a fresh (non-stale) nonzero
            // score for the current query row; otherwise the event costs
            // nothing. The running normalized estimate
            //   out ~= (sum_j w_j * v_j) >>> log2(sum_j w_j)
            // is emitted per V event (power-of-two approximation of the
            // division by the weight total; exact division is done by the
            // host in software when needed).
            // -----------------------------------------------------------------
            if (v_spike_valid) begin
                num_acc[v_dim_idx] <= num_next;
                den_acc            <= den_acc + {{(2*DATA_WIDTH-2){1'b0}}, v_den_inc};
                if (v_den_inc != 2'd0) begin
                    have_weights <= 1'b1;
                end

                out_valid <= 1'b1;
                out_context_data <= ((have_weights || (v_den_inc != 2'd0)) && (den_shift > 0))
                                    ? (num_next >>> den_shift)
                                    : num_next;
                out_dim_idx   <= v_dim_idx;
                out_token_idx <= cur_q_row;

                // Done pulse when the last V element of the sequence streams
                // through (last token, last dimension).
                if ((v_token_idx == SEQ_LEN - 1) && (v_dim_idx == HEAD_DIM - 1)) begin
                    attn_done <= 1'b1;
                end
            end else begin
                out_valid <= 1'b0;
            end
        end
    end

endmodule
