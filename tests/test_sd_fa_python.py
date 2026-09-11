"""Unit tests for the pure-NumPy SD-FlashAttention reference (sd_flashattention_engine).

Verifies that:
  - the kernel runs on small inputs and returns finite, correctly-shaped output;
  - the attention weighting really uses the coincidence scores (a query with
    all-zero spikes produces zero output; identical keys produce the mean value;
    a dominant coincidence key dominates the output);
  - the implementation matches an independently written manual reference.
"""
import numpy as np
import pytest

import sd_flashattention_engine as sd


def make_inputs(seq_len=8, head_dim=4, num_heads=1, seed=0):
    rng = np.random.default_rng(seed)
    s_q = rng.integers(-1, 2, size=(num_heads, seq_len, head_dim)).astype(np.int8)
    s_k = rng.integers(-1, 2, size=(num_heads, seq_len, head_dim)).astype(np.int8)
    values = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    return s_q, s_k, values


def manual_reference(s_q, s_k, values, head_dim):
    """Independent (loop-based) implementation of the documented formula."""
    num_heads, seq_len, _ = s_q.shape
    out = np.zeros_like(values, dtype=np.float64)
    for h in range(num_heads):
        for i in range(seq_len):
            for d in range(head_dim):
                acc = 0.0
                for j in range(seq_len):
                    score = int(np.sum(s_q[h, i] * s_k[h, j]))
                    acc += (score / head_dim) * values[h, j, d]
                out[h, i, d] = acc
    return out


class TestSdFlashAttention:
    def test_output_shape_and_finiteness(self):
        s_q, s_k, values = make_inputs()
        out = sd.sd_attention(s_q, s_k, values, head_dim=s_q.shape[-1])
        assert out.shape == values.shape
        assert np.isfinite(out).all()

    def test_matches_manual_reference(self):
        s_q, s_k, values = make_inputs(seed=3)
        out = sd.sd_attention(s_q, s_k, values, head_dim=s_q.shape[-1])
        ref = manual_reference(s_q, s_k, values, s_q.shape[-1])
        assert np.allclose(out, ref, atol=1e-5)

    def test_zero_coincidence_gives_zero_output(self):
        # All-zero query spikes: no coincidence mass anywhere -> zero output.
        s_q = np.zeros((1, 8, 4), dtype=np.int8)
        _, s_k, values = make_inputs(seed=1)
        out = sd.sd_attention(s_q, s_k, values, head_dim=4)
        assert np.count_nonzero(out) == 0

    def test_identical_keys_give_uniform_attention(self):
        # If every key is identical, each query attends uniformly: the output
        # is proportional to the mean value vector (score is constant over j).
        head_dim = 4
        s_q = np.ones((1, 4, head_dim), dtype=np.int8)
        key = np.ones((1, 1, head_dim), dtype=np.int8)
        s_k = np.repeat(key, 5, axis=1)
        values = np.stack([np.arange(1, 6, dtype=np.float32),
                           np.arange(10, 15, dtype=np.float32),
                           np.arange(20, 25, dtype=np.float32),
                           np.arange(30, 35, dtype=np.float32)]).reshape(1, 5, head_dim)
        out = sd.sd_attention(s_q, s_k, values, head_dim=head_dim)
        mean_v = values.mean(axis=1, keepdims=True)
        scale = s_q.shape[-1] * 1.0 / head_dim * s_k.shape[1]  # score/head_dim * N_j
        assert np.allclose(out, scale * mean_v, atol=1e-4)

    def test_dominant_coincidence_key_dominates_output(self):
        # Zero-coincidence keys contribute NOTHING (gating by score): with
        # only key 0 having a maximal coincidence, the output is exactly V[0].
        head_dim = 8
        s_q = np.ones((1, 1, head_dim), dtype=np.int8)
        s_k = np.zeros((1, 4, head_dim), dtype=np.int8)
        s_k[0, 0, :] = 1  # score(head_dim) for key 0; keys 1..3 score 0
        rng = np.random.default_rng(7)
        values = rng.standard_normal((1, 4, head_dim)).astype(np.float32)
        out = sd.sd_attention(s_q, s_k, values, head_dim=head_dim)
        assert np.allclose(out[0, 0], values[0, 0], atol=1e-5)

        # With the other keys also fully coincident, each key contributes
        # score/head_dim = 1 times its value, so the (unnormalized,
        # softmax-free) output is the plain SUM of the four value vectors.
        s_k[:] = 1
        out = sd.sd_attention(s_q, s_k, values, head_dim=head_dim)
        assert np.allclose(out[0, 0], values[0].sum(axis=0), atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
