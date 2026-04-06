from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from core.config import CheckpointConfig, LogConfig


class CheckpointManager:
    """Saves and loads model checkpoints."""

    def __init__(self, cfg: CheckpointConfig) -> None:
        self.cfg = cfg
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._saved: List[Tuple[str, float]] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
        }
        path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, path)
        self._saved.append((str(path), metrics.get("val_loss", float("inf"))))

        if is_best and self.cfg.save_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(state, best_path)

        self._prune()
        return str(path)

    def _prune(self) -> None:
        if len(self._saved) > self.cfg.keep_last:
            to_remove = self._saved[: -self.cfg.keep_last]
            for path, _ in to_remove:
                if os.path.exists(path) and "best_model" not in path:
                    os.remove(path)
            self._saved = self._saved[-self.cfg.keep_last :]

    @staticmethod
    def load(
        path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint


class TrainingLogger:
    """Logs metrics to console and TensorBoard."""

    def __init__(self, cfg: LogConfig, experiment_name: str = "experiment") -> None:
        self.cfg = cfg
        self.log_dir = Path(cfg.log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._history: Dict[str, List[float]] = {}
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[WARN] TensorBoard not available, falling back to file logging.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if tag not in self._history:
            self._history[tag] = []
        self._history[tag].append(value)
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        for tag, val in tag_scalar_dict.items():
            self.log_scalar(f"{main_tag}/{tag}", val, step)
        if self._writer:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: Tensor, step: int) -> None:
        if self._writer:
            self._writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        if self._writer:
            self._writer.add_text(tag, text, step)

    def flush(self) -> None:
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self._history, f, indent=2)

    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history


class GradientInspector:
    """Monitors gradient stats for debugging."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                stats[name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "norm": grad.norm().item(),
                }
        return stats

    def print_gradient_summary(self) -> None:
        stats = self.get_gradient_stats()
        print("\n" + "=" * 80)
        print("GRADIENT INSPECTION")
        print("=" * 80)
        print(f"{'Parameter':50s} {'Norm':>10s} {'Mean':>10s} {'Std':>10s}")
        print("-" * 80)
        for name, s in stats.items():
            short_name = name[-48:] if len(name) > 48 else name
            print(f"{short_name:50s} {s['norm']:10.6f} {s['mean']:10.6f} {s['std']:10.6f}")
        print("=" * 80)

    def check_anomalies(self, threshold: float = 100.0) -> List[str]:
        anomalies: List[str] = []
        stats = self.get_gradient_stats()
        for name, s in stats.items():
            if s["norm"] > threshold:
                anomalies.append(f"EXPLODING gradient in {name}: norm={s['norm']:.4f}")
            if s["norm"] < 1e-8 and s["std"] < 1e-10:
                anomalies.append(f"VANISHING gradient in {name}: norm={s['norm']:.10f}")
        return anomalies


class ActivationStatsHook:
    """Tracks activation stats via forward hooks."""

    def __init__(self, model: nn.Module, module_types: Optional[Tuple[type, ...]] = None) -> None:
        self.stats: Dict[str, Dict[str, float]] = {}
        self._hooks: List[Any] = []
        types = module_types or (nn.Linear, nn.Conv2d)
        for name, module in model.named_modules():
            if isinstance(module, types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, Tensor):
                self.stats[name] = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "min": output.min().item(),
                    "max": output.max().item(),
                    "frac_zero": (output == 0).float().mean().item(),
                }
        return hook

    def print_summary(self) -> None:
        print("\n" + "=" * 90)
        print("ACTIVATION STATISTICS")
        print("=" * 90)
        print(f"{'Layer':50s} {'Mean':>8s} {'Std':>8s} {'%Zero':>8s}")
        print("-" * 90)
        for name, s in self.stats.items():
            short = name[-48:] if len(name) > 48 else name
            print(f"{short:50s} {s['mean']:8.4f} {s['std']:8.4f} {s['frac_zero']:8.4f}")
        print("=" * 90)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class PerformanceProfiler:
    """Times training phases."""

    def __init__(self) -> None:
        self._timers: Dict[str, float] = {}
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def start(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = time.perf_counter() - self._timers.get(name, time.perf_counter())
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1
        return elapsed

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for name in self._totals:
            total = self._totals[name]
            count = self._counts[name]
            result[name] = {
                "total_seconds": total,
                "count": count,
                "avg_seconds": total / max(count, 1),
            }
        return result

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILE")
        print("=" * 60)
        print(f"{'Phase':30s} {'Total(s)':>10s} {'Count':>8s} {'Avg(s)':>10s}")
        print("-" * 60)
        for name, info in s.items():
            print(
                f"{name:30s} {info['total_seconds']:10.3f} "
                f"{int(info['count']):8d} {info['avg_seconds']:10.5f}"
            )
        print("=" * 60)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping curve plots.")
        return

    loss_keys = [k for k in history if "loss" in k.lower()]
    acc_keys = [k for k in history if "accuracy" in k.lower()]
    f1_keys = [k for k in history if "f1" in k.lower()]

    num_plots = sum(1 for g in [loss_keys, acc_keys, f1_keys] if g)
    if num_plots == 0:
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    idx = 0
    if loss_keys:
        ax = axes[idx]
        for k in loss_keys:
            ax.plot(history[k], label=k.replace("/", " "))
        ax.set_title("Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        idx += 1

    if acc_keys:
        ax = axes[idx]
        for k in acc_keys:
            ax.plot(history[k], label=k.replace("/", " "))
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        idx += 1

    if f1_keys:
        ax = axes[idx]
        for k in f1_keys:
            ax.plot(history[k], label=k.replace("/", " "))
        ax.set_title("F1 Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {save_path}")
