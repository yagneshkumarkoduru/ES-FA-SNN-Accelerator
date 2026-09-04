"""Helpers for structured iteration construction (v1/v2/v3)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_results(results_root: Path) -> List[Dict[str, Any]]:
    exp_dir = results_root / "experiments"
    runs = []
    for metrics_path in sorted(exp_dir.glob("*/metrics.json")):
        data = _load_json(metrics_path)
        data["_metrics_path"] = str(metrics_path)
        runs.append(data)
    return runs


def _normalized(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def rank_experiments(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    accs = [float(r["metrics"]["accuracy"]) for r in results]
    energies = [float(r["metrics"]["energy_proxy"]) for r in results]
    latencies = [float(r["metrics"]["latency_proxy"]) for r in results]
    sparsities = [float(r["metrics"]["spike_sparsity"]) for r in results]

    e_norm = _normalized(energies)
    l_norm = _normalized(latencies)
    s_norm = _normalized(sparsities)

    ranked = []
    for idx, r in enumerate(results):
        score = (
            0.70 * accs[idx]
            + 0.20 * s_norm[idx]
            - 0.07 * e_norm[idx]
            - 0.03 * l_norm[idx]
        )
        item = deepcopy(r)
        item["iteration_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x["iteration_score"], reverse=True)
    return ranked


def find_experiment_config(experiments_root: Path, run_name: str) -> Path:
    for config_path in experiments_root.glob("*/config.json"):
        cfg = _load_json(config_path)
        if cfg.get("run_name") == run_name:
            return config_path
    raise FileNotFoundError(f"Experiment config not found for run_name={run_name}")


def _load_config(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def _merge_for_v2(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(cfg_a)
    out["run_name"] = "iteration_v2"
    out["category"] = "iteration"
    out["proposal_mapping"] = "v2 combine top-2 proposal ideas"

    out["epochs"] = int(max(cfg_a.get("epochs", 5), cfg_b.get("epochs", 5)) + 1)
    out["learning_rate"] = float(min(cfg_a.get("learning_rate", 1e-3), cfg_b.get("learning_rate", 1e-3)))
    out["seed"] = 202

    for key in ("hardware", "sparsity", "latency"):
        out["loss_weights"][key] = float(cfg_a["loss_weights"].get(key, 0.0) + cfg_b["loss_weights"].get(key, 0.0))

    out["sparsity_target"] = float(
        (cfg_a.get("sparsity_target", 0.90) + cfg_b.get("sparsity_target", 0.90)) / 2.0
    )

    q_a = cfg_a.get("quantization", {})
    q_b = cfg_b.get("quantization", {})
    out["quantization"]["enabled"] = bool(q_a.get("enabled", False) or q_b.get("enabled", False))
    out["quantization"]["weight_bits"] = int(min(q_a.get("weight_bits", 8), q_b.get("weight_bits", 8)))
    out["quantization"]["act_bits"] = int(min(q_a.get("act_bits", 8), q_b.get("act_bits", 8)))

    hw_a = cfg_a.get("hardware_estimator", {})
    hw_b = cfg_b.get("hardware_estimator", {})
    modes = [hw_a.get("dataflow_mode", "dense"), hw_b.get("dataflow_mode", "dense")]
    if "adaptive" in modes:
        mode = "adaptive"
    elif "event" in modes:
        mode = "event"
    else:
        mode = "dense"
    out["hardware_estimator"]["dataflow_mode"] = mode
    out["hardware_estimator"]["temporal_multiplexing_factor"] = int(
        max(hw_a.get("temporal_multiplexing_factor", 1), hw_b.get("temporal_multiplexing_factor", 1))
    )
    out["hardware_estimator"]["density_threshold"] = float(
        min(hw_a.get("density_threshold", 0.1), hw_b.get("density_threshold", 0.1))
    )
    out["hardware_estimator"]["a"] = float((hw_a.get("a", 1.0) + hw_b.get("a", 1.0)) / 2.0)
    out["hardware_estimator"]["b"] = float((hw_a.get("b", 0.2) + hw_b.get("b", 0.2)) / 2.0)
    out["hardware_estimator"]["queue_pressure_scale"] = float(
        (hw_a.get("queue_pressure_scale", 0.2) + hw_b.get("queue_pressure_scale", 0.2)) / 2.0
    )
    return out


def build_v1_config(results_root: Path, experiments_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ranked = rank_experiments(load_experiment_results(results_root))
    if not ranked:
        raise RuntimeError("No experiment metrics found. Run experiments first.")
    best_exp = ranked[0]
    best_run_name = best_exp["run_name"]
    cfg_path = find_experiment_config(experiments_root, best_run_name)
    cfg = _load_config(cfg_path)
    cfg["run_name"] = "iteration_v1"
    cfg["category"] = "iteration"
    cfg["proposal_mapping"] = f"v1 best single experiment selected from {best_run_name}"
    cfg["seed"] = 101
    cfg["epochs"] = int(cfg.get("epochs", 5) + 1)
    return cfg, best_exp


def build_v2_config(results_root: Path, experiments_root: Path) -> Tuple[Dict[str, Any], List[str]]:
    ranked = rank_experiments(load_experiment_results(results_root))
    if len(ranked) < 2:
        raise RuntimeError("Need at least two experiments for v2 combination.")
    top1, top2 = ranked[0], ranked[1]
    cfg1 = _load_config(find_experiment_config(experiments_root, top1["run_name"]))
    cfg2 = _load_config(find_experiment_config(experiments_root, top2["run_name"]))
    cfg = _merge_for_v2(cfg1, cfg2)
    return cfg, [top1["run_name"], top2["run_name"]]


def build_v3_config(v2_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(v2_config)
    cfg["run_name"] = "iteration_v3"
    cfg["category"] = "iteration"
    cfg["proposal_mapping"] = "v3 refined/tuned variant of v2"
    cfg["seed"] = 303
    cfg["epochs"] = int(cfg.get("epochs", 6) + 1)
    cfg["learning_rate"] = float(cfg.get("learning_rate", 1e-3) * 0.7)
    cfg["sparsity_target"] = float(min(0.97, cfg.get("sparsity_target", 0.90) + 0.01))

    cfg["loss_weights"]["hardware"] = float(cfg["loss_weights"].get("hardware", 0.0) * 1.10)
    cfg["loss_weights"]["sparsity"] = float(cfg["loss_weights"].get("sparsity", 0.0) * 1.05)
    cfg["loss_weights"]["latency"] = float(cfg["loss_weights"].get("latency", 0.0) * 0.90)
    return cfg


def compare_metric_sets(curr: Dict[str, Any], prev: Dict[str, Any]) -> Dict[str, Any]:
    cm = curr["metrics"]
    pm = prev["metrics"]
    return {
        "current_run": curr["run_name"],
        "previous_run": prev["run_name"],
        "accuracy_delta": float(cm["accuracy"] - pm["accuracy"]),
        "sparsity_delta": float(cm["spike_sparsity"] - pm["spike_sparsity"]),
        "energy_delta": float(cm["energy_proxy"] - pm["energy_proxy"]),
        "latency_delta": float(cm["latency_proxy"] - pm["latency_proxy"]),
    }

