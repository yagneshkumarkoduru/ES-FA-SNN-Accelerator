"""Aggregation-logic tests for the multi-seed reproducibility harness.

experiments/multiseed.py is written so the numpy-only aggregation path can be
tested without torch (which is not part of the CI dependency set). These tests
exercise the mean/std/n aggregation over synthetic per-seed metric
dictionaries and the torch-missing skip path.
"""
import numpy as np
import pytest

from experiments.multiseed import (
    _report_missing_torch,
    _try_import_torch,
    aggregate_seed_metrics,
)


class TestAggregateSeedMetrics:
    def test_mean_and_sample_std_match_numpy_reference(self):
        per_seed = [
            {"accuracy": 0.90, "spike_sparsity": 0.50, "energy_proxy": 100.0},
            {"accuracy": 0.94, "spike_sparsity": 0.60, "energy_proxy": 200.0},
            {"accuracy": 0.91, "spike_sparsity": 0.55, "energy_proxy": 150.0},
        ]
        summary = aggregate_seed_metrics(per_seed)
        acc = np.asarray([0.90, 0.94, 0.91])
        spar = np.asarray([0.50, 0.55, 0.60])
        energy = np.asarray([100.0, 150.0, 200.0])
        assert summary["accuracy"]["mean"] == pytest.approx(float(acc.mean()))
        assert summary["accuracy"]["std"] == pytest.approx(float(acc.std(ddof=1)))
        assert summary["spike_sparsity"]["mean"] == pytest.approx(float(spar.mean()))
        assert summary["spike_sparsity"]["std"] == pytest.approx(float(spar.std(ddof=1)))
        assert summary["energy_proxy"]["mean"] == pytest.approx(float(energy.mean()))
        assert summary["energy_proxy"]["std"] == pytest.approx(float(energy.std(ddof=1)))
        assert all(summary[key]["n"] == 3 for key in ("accuracy", "spike_sparsity", "energy_proxy"))

    def test_hand_computed_two_sample_case(self):
        # accuracy = [0.80, 0.90]: mean 0.85, sample std = sqrt(0.005) ~= 0.0707107
        summary = aggregate_seed_metrics([{"accuracy": 0.80}, {"accuracy": 0.90}])
        assert summary["accuracy"]["mean"] == pytest.approx(0.85)
        assert summary["accuracy"]["std"] == pytest.approx(0.070710678, rel=1e-6)
        assert summary["accuracy"]["n"] == 2

    def test_single_seed_yields_zero_std(self):
        summary = aggregate_seed_metrics([{"accuracy": 0.957, "energy_proxy": 4592863.1728}])
        assert summary["accuracy"]["mean"] == pytest.approx(0.957)
        assert summary["accuracy"]["std"] == 0.0
        assert summary["accuracy"]["n"] == 1
        assert summary["energy_proxy"]["n"] == 1

    def test_partial_metrics_kept_with_own_sample_count(self):
        # A seed that produced only a subset of metrics must not erase the
        # others: each metric aggregates over the seeds that reported it.
        per_seed = [
            {"accuracy": 0.90, "latency_proxy": 1.0},
            {"accuracy": 0.92},
        ]
        summary = aggregate_seed_metrics(per_seed)
        assert summary["accuracy"]["n"] == 2
        assert summary["accuracy"]["mean"] == pytest.approx(0.91)
        assert summary["latency_proxy"]["n"] == 1
        assert summary["latency_proxy"]["std"] == 0.0

    def test_empty_input_returns_empty_summary(self):
        assert aggregate_seed_metrics([]) == {}

    def test_integer_values_accepted(self):
        summary = aggregate_seed_metrics([{"accuracy": 1}, {"accuracy": 0}])
        assert summary["accuracy"]["mean"] == pytest.approx(0.5)
        assert summary["accuracy"]["std"] == pytest.approx(0.70710678, rel=1e-6)  # sqrt(0.5), ddof=1


class TestTorchGuard:
    def test_probe_returns_bool(self):
        assert isinstance(_try_import_torch(), bool)

    def test_missing_torch_message_and_exit_code(self, capsys):
        # Exercises the skip path directly so the message/exit contract is
        # verified regardless of whether torch is installed locally.
        exit_code = _report_missing_torch()
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "torch is not installed" in captured.out
        assert "pip install torch" in captured.out
