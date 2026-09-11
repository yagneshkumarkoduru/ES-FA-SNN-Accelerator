"""Reference-math tests for the ES-FA LIF fixed-point update.

The C99 engine and the synthesizable RTL both compute the membrane update as
integer arithmetic with a shift-based leak:

    V_leak = V - (V >> LEAK_SHIFT)        (arithmetic shift, LEAK_SHIFT = 3)
    V_next = V_leak + W*S
    spike if V_next >= THRESHOLD (64 in the C engine), hard reset to RESET_VALUE

This suite mirrors that arithmetic in numpy and checks hand-computed traces so
the reference model and the hardware stay aligned. The training-side surrogate
model (beta = 0.875 float) is compared only for bounded drift; the training
code itself is untouched.
"""
import numpy as np
import pytest

LEAK_SHIFT = 3
THRESHOLD = 64


def lif_fixed_point_step(v, w):
    """One integer LIF update exactly as implemented by the C engine / RTL."""
    v_leak = v - (v >> LEAK_SHIFT)
    v_next = v_leak + w
    if v_next >= THRESHOLD:
        return 0, 1  # spike: hard reset to 0
    return v_next, 0


def run_lif_trace(w, steps, v0=0):
    v = v0
    trace, spikes = [], []
    for _ in range(steps):
        v, s = lif_fixed_point_step(v, int(w))
        trace.append(v)
        spikes.append(s)
    return trace, spikes


class TestLifFixedPoint:
    def test_hand_computed_trace_w18(self):
        # V: 0 -> 18 -> 34 -> 48 -> 60 -> 71(spike, reset) ...
        # 18 >> 3 = 2, 34 >> 3 = 4, 48 >> 3 = 6, 60 >> 3 = 7
        trace, spikes = run_lif_trace(18, 10)
        assert trace[:5] == [18, 34, 48, 60, 0]
        assert spikes == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    def test_period_5_limit_cycle_w18(self):
        # After the reset the pattern repeats every 5 steps: 18,34,48,60,spike.
        trace, _ = run_lif_trace(18, 15)
        assert trace[5:10] == trace[0:5]
        assert trace[10:15] == trace[0:5]

    def test_negative_weight_converges_to_stable_point(self):
        # Negative weight with a floor-based arithmetic shift: the fixed point
        # solves V - floor(V/8) + W = V, i.e. floor(V/8) = -W = 10, so any
        # V in [-80, -73] is a fixed point. The iteration from rest lands at
        # -73. (Note: for negative V the shift leak decays SLOWER than the
        # float beta = 0.875 model because floor(V/8) <= V/8.)
        trace, spikes = run_lif_trace(-10, 60)
        assert not any(spikes)
        assert trace[-1] == -73
        assert (np.diff(trace) <= 0).all()      # monotonically decreasing
        assert min(trace) >= -73

    def test_negative_shift_is_arithmetic(self):
        # -16 >> 3 = -2 (floor), so the leak of -16 is -14, not -12.
        assert (-16) >> LEAK_SHIFT == -2
        v, s = lif_fixed_point_step(-16, 0)
        assert v == -14 and s == 0

    def test_threshold_boundary_is_inclusive(self):
        # V_next == THRESHOLD must fire (>= comparison, as in Verilog >=).
        v, s = lif_fixed_point_step(0, THRESHOLD)
        assert s == 1 and v == 0

    def test_just_below_threshold_does_not_fire(self):
        v, s = lif_fixed_point_step(0, THRESHOLD - 1)
        assert s == 0 and v == THRESHOLD - 1

    def test_saturation_bounded_int16(self):
        # Weights saturate at INT8 range; membrane stays inside INT16.
        trace, _ = run_lif_trace(127, 8)
        assert all(-32768 <= t <= 32767 for t in trace)
        trace, _ = run_lif_trace(-128, 8)
        assert all(-32768 <= t <= 32767 for t in trace)

    def test_leak_operator_matches_beta_0875_within_one_lsb(self):
        # The shift leak realizes beta = 1 - 2^-3 = 0.875: for any V,
        # |(V - (V >> 3)) - 0.875*V| < 1 LSB (the error is the fractional
        # part of V/8). Verified over a wide signed range.
        rng = np.random.default_rng(42)
        for v in rng.integers(-3000, 3001, size=200):
            leaked = int(v) - (int(v) >> LEAK_SHIFT)
            assert abs(leaked - 0.875 * int(v)) < 1.0

    def test_leak_is_exact_for_values_divisible_by_eight(self):
        # For V divisible by 8 the shift leak is EXACT: V - (V>>3) == 0.875*V.
        for v in (-64, -16, 0, 8, 32, 64, 128):
            assert int(v) - (int(v) >> LEAK_SHIFT) == int(0.875 * int(v))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
