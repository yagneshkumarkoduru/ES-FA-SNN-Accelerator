"""Shared training core for baseline, experiments, and iterations."""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from p1_training.export_utils import export_int8_weights, export_spike_stats
from p1_training.model import ProposalAlignedSNN
from p2_hardware_model.estimators import (
    estimate_energy_proxy,
    estimate_latency_proxy,
    estimate_memory_access,
    estimate_spike_cost,
)


def default_config() -> Dict[str, Any]:
    return {
        "run_name": "baseline",
        "category": "baseline",
        "proposal_mapping": "Paper1 baseline training + Paper2 estimator integration",
        "run_metadata": {
            "model_id": "paper1_mlp_784_128_64_10",
            "dataset_split": "mnist_test",
            "scheduler_mode": "dense",
            "execution_mode": "cpu",
            "quantization_mode": "fp32",
            "clock_frequency_mhz": 0,
        },
        "seed": 42,
        "dataset": "MNIST",
        "device": "auto",
        "epochs": 5,
        "batch_size": 64,
        "num_workers": 2,
        "learning_rate": 1e-3,
        "time_steps": 16,
        "encoder": "poisson",
        "beta": 0.9,
        "threshold": 1.0,
        "surrogate_slope": 5.0,
        "loss_weights": {
            "hardware": 0.0,
            "sparsity": 0.0,
            "latency": 0.0,
        },
        "sparsity_target": 0.90,
        "quantization": {
            "enabled": False,
            "weight_bits": 8,
            "act_bits": 8,
        },
        "hardware_estimator": {
            "a": 1.0,
            "b": 0.2,
            "dataflow_mode": "dense",  # dense | event | adaptive
            "density_threshold": 0.10,
            "temporal_multiplexing_factor": 1,
            "queue_pressure_scale": 0.20,
        },
    }


def deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = default_config()
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as f:
            from_file = json.load(f)
        cfg = deep_merge_dict(cfg, from_file)
    if overrides is not None:
        cfg = deep_merge_dict(cfg, overrides)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_loaders(data_root: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def _batch_hardware_metrics(batch_stats: Dict[str, object], batch_size: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    spike_counts = batch_stats["spike_counts"]  # type: ignore[index]
    possible = batch_stats["possible_spikes"]  # type: ignore[index]
    tsteps = int(cfg["time_steps"])
    hw_cfg = cfg["hardware_estimator"]

    spike_cost = estimate_spike_cost(spike_counts)
    mem = estimate_memory_access(
        spike_counts=spike_counts,
        batch_size=batch_size,
        time_steps=tsteps,
        dataflow_mode=str(hw_cfg["dataflow_mode"]),
        density_threshold=float(hw_cfg["density_threshold"]),
        temporal_multiplex_factor=int(hw_cfg["temporal_multiplexing_factor"]),
    )

    hidden_spikes = float(spike_counts["layer1"] + spike_counts["layer2"])
    hidden_possible = max(1.0, float(possible["layer1"] + possible["layer2"]))
    hidden_sparsity = 1.0 - (hidden_spikes / hidden_possible)
    hidden_density = 1.0 - hidden_sparsity
    layer1_density = float(spike_counts["layer1"]) / max(1.0, float(possible["layer1"]))
    layer2_density = float(spike_counts["layer2"]) / max(1.0, float(possible["layer2"]))
    layer1_sparsity = 1.0 - layer1_density
    layer2_sparsity = 1.0 - layer2_density

    queue_pressure = float(hw_cfg["queue_pressure_scale"]) * (mem["event_total_accesses"] / max(mem["dense_total_accesses"], 1.0))
    latency = estimate_latency_proxy(
        spike_sparsity=hidden_sparsity,
        temporal_multiplex_factor=int(hw_cfg["temporal_multiplexing_factor"]),
        queue_pressure=queue_pressure,
        base_latency=1.0,
    )

    energy = estimate_energy_proxy(
        total_spikes=spike_cost["total_spikes"],
        total_memory_accesses=mem["adjusted_total_accesses"],
        a=float(hw_cfg["a"]),
        b=float(hw_cfg["b"]),
    )
    dense_energy_ref = estimate_energy_proxy(
        total_spikes=spike_cost["total_spikes"],
        total_memory_accesses=mem["dense_total_accesses"],
        a=float(hw_cfg["a"]),
        b=float(hw_cfg["b"]),
    )

    return {
        "total_spikes": spike_cost["total_spikes"],
        "hidden_sparsity": hidden_sparsity,
        "hidden_density": hidden_density,
        "layer1_density": layer1_density,
        "layer2_density": layer2_density,
        "layer1_sparsity": layer1_sparsity,
        "layer2_sparsity": layer2_sparsity,
        "input_density": mem["input_density"],
        "memory_total": mem["total_accesses"],
        "memory_adjusted_total": mem["adjusted_total_accesses"],
        "dense_memory_reference": mem["dense_total_accesses"],
        "event_memory_reference": mem["event_total_accesses"],
        "energy_proxy": energy,
        "dense_energy_reference": dense_energy_ref,
        "latency_proxy": latency,
    }


def run_epoch(
    model: ProposalAlignedSNN,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: Dict[str, Any],
    capture_raster: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    total_ce = 0.0
    total_energy = 0.0
    total_latency = 0.0
    total_memory = 0.0
    total_spikes = 0.0
    total_hidden_density = 0.0
    total_hidden_sparsity = 0.0
    total_input_density = 0.0
    total_layer1_density = 0.0
    total_layer2_density = 0.0
    total_layer1_sparsity = 0.0
    total_layer2_sparsity = 0.0

    captured_raster = None
    loss_w = cfg["loss_weights"]
    sparsity_target = float(cfg["sparsity_target"])

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits, batch_stats = model(images, capture_raster=(capture_raster and batch_idx == 0))
        hw = _batch_hardware_metrics(batch_stats, batch_size=batch_size, cfg=cfg)

        ce_loss = F.cross_entropy(logits, labels)

        # Hardware-aware objective components.
        energy_norm = max(1.0, hw["dense_energy_reference"])
        latency_norm = max(1.0, float(cfg["hardware_estimator"]["temporal_multiplexing_factor"]))
        sparsity_penalty = max(0.0, sparsity_target - hw["hidden_sparsity"])

        loss = ce_loss
        loss = loss + float(loss_w["hardware"]) * (hw["energy_proxy"] / energy_norm)
        loss = loss + float(loss_w["latency"]) * (hw["latency_proxy"] / latency_norm)
        loss = loss + float(loss_w["sparsity"]) * sparsity_penalty

        if is_train:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        correct = int((preds == labels).sum().item())

        total_samples += batch_size
        total_correct += correct
        total_loss += float(loss.item()) * batch_size
        total_ce += float(ce_loss.item()) * batch_size
        total_energy += hw["energy_proxy"] * batch_size
        total_latency += hw["latency_proxy"] * batch_size
        total_memory += hw["memory_adjusted_total"] * batch_size
        total_spikes += hw["total_spikes"] * batch_size
        total_hidden_density += hw["hidden_density"] * batch_size
        total_hidden_sparsity += hw["hidden_sparsity"] * batch_size
        total_input_density += hw["input_density"] * batch_size
        total_layer1_density += hw["layer1_density"] * batch_size
        total_layer2_density += hw["layer2_density"] * batch_size
        total_layer1_sparsity += hw["layer1_sparsity"] * batch_size
        total_layer2_sparsity += hw["layer2_sparsity"] * batch_size

        if capture_raster and batch_idx == 0 and "raster" in batch_stats:
            captured_raster = batch_stats["raster"]  # type: ignore[assignment]

    denom = max(1, total_samples)
    metrics = {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "accuracy": total_correct / denom,
        "energy_proxy": total_energy / denom,
        "latency_proxy": total_latency / denom,
        "memory_accesses": total_memory / denom,
        "total_spikes": total_spikes / denom,
        "hidden_spike_density": total_hidden_density / denom,
        "hidden_spike_sparsity": total_hidden_sparsity / denom,
        "input_spike_density": total_input_density / denom,
        "layer1_spike_density": total_layer1_density / denom,
        "layer2_spike_density": total_layer2_density / denom,
        "layer1_spike_sparsity": total_layer1_sparsity / denom,
        "layer2_spike_sparsity": total_layer2_sparsity / denom,
    }
    return metrics, captured_raster


def _json_dump(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _build_run_manifest(cfg: Dict[str, Any], output_dir: Path, data_root: Path, device: torch.device) -> Dict[str, Any]:
    meta = cfg.get("run_metadata", {})
    quant = cfg.get("quantization", {})
    hw = cfg.get("hardware_estimator", {})
    return {
        "run_name": cfg.get("run_name"),
        "category": cfg.get("category"),
        "proposal_mapping": cfg.get("proposal_mapping"),
        "dataset": cfg.get("dataset"),
        "seed": cfg.get("seed"),
        "model_id": meta.get("model_id", "paper1_mlp_784_128_64_10"),
        "dataset_split": meta.get("dataset_split", "mnist_test"),
        "quantization": {
            "enabled": bool(quant.get("enabled", False)),
            "weight_bits": int(quant.get("weight_bits", 8)),
            "act_bits": int(quant.get("act_bits", 8)),
            "mode": meta.get("quantization_mode", "int8" if bool(quant.get("enabled", False)) else "fp32"),
        },
        "scheduler_mode": meta.get("scheduler_mode", hw.get("dataflow_mode", "dense")),
        "execution_mode": meta.get("execution_mode", "cpu"),
        "clock_frequency_mhz": int(meta.get("clock_frequency_mhz", 0)),
        "paths": {
            "output_dir": str(output_dir.resolve()),
            "data_root": str(data_root.resolve()),
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def train_from_config(
    cfg: Dict[str, Any],
    output_dir: Path,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))

    device = resolve_device(str(cfg["device"]))
    if data_root is None:
        data_root = Path(__file__).resolve().parents[1] / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(
        data_root=data_root,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )

    model = ProposalAlignedSNN(
        time_steps=int(cfg["time_steps"]),
        encoder=str(cfg["encoder"]),
        beta=float(cfg["beta"]),
        threshold=float(cfg["threshold"]),
        surrogate_slope=float(cfg["surrogate_slope"]),
        quant_enabled=bool(cfg["quantization"]["enabled"]),
        weight_bits=int(cfg["quantization"]["weight_bits"]),
        act_bits=int(cfg["quantization"]["act_bits"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _json_dump(output_dir / "config_used.json", cfg)
    _json_dump(output_dir / "run_manifest.json", _build_run_manifest(cfg, output_dir, data_root, device))

    history = []
    best_acc = -1.0
    best_epoch = -1
    best_eval_metrics: Dict[str, float] = {}
    best_raster: Optional[Dict[str, np.ndarray]] = None

    start_time = time.time()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            capture_raster=False,
        )
        eval_metrics, raster = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            cfg=cfg,
            capture_raster=True,
        )

        row = {"epoch": epoch, "train": train_metrics, "val": eval_metrics}
        history.append(row)

        print(
            f"[{cfg['run_name']}] epoch={epoch:02d} "
            f"val_acc={eval_metrics['accuracy']*100:.2f}% "
            f"sparsity={eval_metrics['hidden_spike_sparsity']*100:.2f}% "
            f"energy={eval_metrics['energy_proxy']:.2f} "
            f"latency={eval_metrics['latency_proxy']:.4f}"
        )

        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            best_epoch = epoch
            best_eval_metrics = dict(eval_metrics)
            best_raster = raster
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metrics": eval_metrics,
                },
                ckpt_dir / "best_model.pt",
            )

    elapsed = time.time() - start_time
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "epoch": int(cfg["epochs"]),
            "val_metrics": history[-1]["val"] if history else {},
        },
        ckpt_dir / "last_model.pt",
    )

    _json_dump(output_dir / "history.json", {"history": history})

    # Export artifacts for hardware handoff.
    export_dir = output_dir / "exports"
    weight_meta = export_int8_weights(model, export_dir)
    spike_summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "hidden_spike_sparsity": best_eval_metrics.get("hidden_spike_sparsity", 0.0),
        "hidden_spike_density": best_eval_metrics.get("hidden_spike_density", 0.0),
        "layer1_spike_sparsity": best_eval_metrics.get("layer1_spike_sparsity", 0.0),
        "layer2_spike_sparsity": best_eval_metrics.get("layer2_spike_sparsity", 0.0),
        "layer1_spike_density": best_eval_metrics.get("layer1_spike_density", 0.0),
        "layer2_spike_density": best_eval_metrics.get("layer2_spike_density", 0.0),
        "total_spikes": best_eval_metrics.get("total_spikes", 0.0),
    }
    export_spike_stats(spike_summary, export_dir)
    layer_activity = {
        "best_epoch": best_epoch,
        "best_layer_activity": {
            "layer1_spike_density": best_eval_metrics.get("layer1_spike_density", 0.0),
            "layer2_spike_density": best_eval_metrics.get("layer2_spike_density", 0.0),
            "layer1_spike_sparsity": best_eval_metrics.get("layer1_spike_sparsity", 0.0),
            "layer2_spike_sparsity": best_eval_metrics.get("layer2_spike_sparsity", 0.0),
        },
        "history": [
            {
                "epoch": int(h["epoch"]),
                "val_layer1_spike_density": float(h["val"].get("layer1_spike_density", 0.0)),
                "val_layer2_spike_density": float(h["val"].get("layer2_spike_density", 0.0)),
                "val_layer1_spike_sparsity": float(h["val"].get("layer1_spike_sparsity", 0.0)),
                "val_layer2_spike_sparsity": float(h["val"].get("layer2_spike_sparsity", 0.0)),
            }
            for h in history
        ],
    }
    _json_dump(export_dir / "layer_activity.json", layer_activity)
    if best_raster is not None:
        np.savez(
            export_dir / "spike_raster.npz",
            layer1=best_raster["layer1"],
            layer2=best_raster["layer2"],
        )

    final_eval = history[-1]["val"] if history else {}
    result = {
        "run_name": cfg["run_name"],
        "category": cfg["category"],
        "proposal_mapping": cfg["proposal_mapping"],
        "best_epoch": best_epoch,
        "best_val_accuracy": best_acc,
        "final_val_accuracy": final_eval.get("accuracy", 0.0),
        "training_time_sec": elapsed,
        "metrics": {
            "accuracy": best_acc,
            "spike_activity": best_eval_metrics.get("hidden_spike_density", 0.0),
            "spike_sparsity": best_eval_metrics.get("hidden_spike_sparsity", 0.0),
            "energy_proxy": best_eval_metrics.get("energy_proxy", 0.0),
            "latency_proxy": best_eval_metrics.get("latency_proxy", 0.0),
            "memory_accesses": best_eval_metrics.get("memory_accesses", 0.0),
        },
        "config": cfg,
        "weight_export": weight_meta,
        "artifacts": {
            "best_checkpoint": str((ckpt_dir / "best_model.pt").resolve()),
            "last_checkpoint": str((ckpt_dir / "last_model.pt").resolve()),
            "history": str((output_dir / "history.json").resolve()),
            "run_manifest": str((output_dir / "run_manifest.json").resolve()),
            "export_dir": str(export_dir.resolve()),
        },
    }
    _json_dump(output_dir / "metrics.json", result)
    return result
