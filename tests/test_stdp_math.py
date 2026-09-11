"""Mirror tests for stdp_weight_updater.v (fixed-step STDP logic).

The synthesizable STDP engine implements, per coincidence event:
    dt = t_post - t_pre
    LTP:  0 < dt <= TAU_WINDOW   ->  w + ALPHA_PLUS   (saturate at +127)
    LTD: -TAU_WINDOW <= dt < 0   ->  w - ALPHA_MINUS  (saturate at -128)
    otherwise (dt == 0 or |dt| > window) -> no update, no write pulse

The numpy mirror below reproduces the exact integer rules and verifies the
branch coverage, including that the LTD branch is reachable (dt < 0).
"""
import numpy as np
import pytest

ALPHA_PLUS = 4
ALPHA_MINUS = 3
TAU_WINDOW = 32
W_MAX, W_MIN = 127, -128


def stdp_fixed_step(w, pre_t, post_t):
    """Exact mirror of stdp_weight_updater.v for one coincidence cycle."""
    dt = int(post_t) - int(pre_t)
    if 0 < dt <= TAU_WINDOW:
        w_new = W_MAX if w + ALPHA_PLUS > W_MAX else w + ALPHA_PLUS
        return w_new, 1
    if -TAU_WINDOW <= dt < 0:
        w_new = W_MIN if w - ALPHA_MINUS < W_MIN else w - ALPHA_MINUS
        return w_new, 1
    return w, 0


class TestStdpFixedStep:
    def test_ltp_within_window(self):
        w, we = stdp_fixed_step(18, pre_t=100, post_t=105)
        assert (w, we) == (22, 1)  # 18 + ALPHA_PLUS

    def test_ltd_within_window_is_reachable(self):
        w, we = stdp_fixed_step(18, pre_t=105, post_t=100)
        assert (w, we) == (15, 1)  # 18 - ALPHA_MINUS
        assert w < 18

    def test_dt_zero_is_no_op(self):
        w, we = stdp_fixed_step(18, pre_t=100, post_t=100)
        assert (w, we) == (18, 0)

    def test_outside_window_is_no_op(self):
        w, we = stdp_fixed_step(18, pre_t=100, post_t=100 + TAU_WINDOW + 1)
        assert (w, we) == (18, 0)
        w, we = stdp_fixed_step(18, pre_t=100, post_t=100 - TAU_WINDOW - 1)
        assert (w, we) == (18, 0)

    def test_window_boundary_dt32_potentiates(self):
        w, we = stdp_fixed_step(18, pre_t=100, post_t=132)
        assert (w, we) == (22, 1)

    def test_saturation_plus(self):
        w, we = stdp_fixed_step(126, pre_t=0, post_t=5)
        assert (w, we) == (W_MAX, 1)
        w, _ = stdp_fixed_step(W_MAX, pre_t=0, post_t=5)
        assert w == W_MAX

    def test_saturation_minus(self):
        w, we = stdp_fixed_step(-127, pre_t=5, post_t=0)
        assert (w, we) == (W_MIN, 1)
        w, _ = stdp_fixed_step(W_MIN, pre_t=5, post_t=0)
        assert w == W_MIN

    def test_repeated_ltd_drives_weight_negative(self):
        w = 18
        ws = []
        for k in range(20):
            w, _ = stdp_fixed_step(w, pre_t=100 + 2 * k, post_t=99 + 2 * k)
            ws.append(w)
        arr = np.array(ws)
        assert arr[-1] <= arr[0] - ALPHA_MINUS
        assert (np.diff(arr) <= 0).all()
        assert arr[-1] == W_MIN or arr[-1] < 0

    def test_alternating_ltp_ltd_is_stable(self):
        # 1 LTP (+4) followed by 1 LTD (-3) per round: the pair nets +1 per
        # round, so the trace is exactly linear and the drift stays bounded.
        w = 18
        history = []
        for k in range(10):
            w, _ = stdp_fixed_step(w, pre_t=100 * k, post_t=100 * k + 5)        # LTP
            w, _ = stdp_fixed_step(w, pre_t=100 * k + 10, post_t=100 * k + 7)   # LTD
            history.append(w)
        arr = np.array(history)
        assert (arr == 18 + np.arange(1, 11)).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
