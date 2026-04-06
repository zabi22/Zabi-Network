from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from config import ExperimentConfig
from losses import ConditionalGateLoss, build_loss_fn
from metrics import MetricsAccumulator
from utils import (
    CheckpointManager,
    TrainingLogger,
    GradientInspector,
    ActivationStatsHook,
    PerformanceProfiler,
    get_device,
)


class WarmupScheduler:
    """Linear warmup then base scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float,
        after_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.after_scheduler = after_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        if self._step_count <= self.warmup_epochs:
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                alpha = self._step_count / max(self.warmup_epochs, 1)
                pg["lr"] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        elif self.after_scheduler is not None:
            self.after_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "step_count": self._step_count,
            "warmup_epochs": self.warmup_epochs,
            "warmup_start_lr": self.warmup_start_lr,
        }
        if self.after_scheduler:
            state["after_scheduler"] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._step_count = state["step_count"]
        if self.after_scheduler and "after_scheduler" in state:
            self.after_scheduler.load_state_dict(state["after_scheduler"])


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    tc = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]
    if tc.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, betas=tc.betas, eps=tc.eps)
    elif tc.optimizer == "adam":
        return torch.optim.Adam(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, betas=tc.betas, eps=tc.eps)
    elif tc.optimizer == "sgd":
        return torch.optim.SGD(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, momentum=tc.momentum)
    elif tc.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=tc.learning_rate, weight_decay=tc.weight_decay, eps=tc.eps)
    raise ValueError(f"Unknown optimizer: {tc.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: ExperimentConfig) -> WarmupScheduler:
    tc = cfg.train
    if tc.scheduler == "cosine_warm_restarts":
        base = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=tc.scheduler_t0, T_mult=tc.scheduler_t_mult, eta_min=tc.scheduler_eta_min
        )
    elif tc.scheduler == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.epochs, eta_min=tc.scheduler_eta_min)
    elif tc.scheduler == "step":
        base = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif tc.scheduler == "plateau":
        base = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        base = None
    return WarmupScheduler(optimizer, tc.warmup_epochs, tc.warmup_start_lr, base)


class Trainer:
    """Training loop with mixed precision and multi-GPU support."""

    def __init__(self, model: nn.Module, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.distributed.local_rank)

        self.model = model.to(self.device)
        if cfg.distributed.enabled and torch.cuda.device_count() > 1:
            if cfg.distributed.use_data_parallel:
                self.model = nn.DataParallel(self.model)
            else:
                torch.distributed.init_process_group(
                    backend=cfg.distributed.backend,
                    init_method=cfg.distributed.init_method,
                    world_size=cfg.distributed.world_size,
                    rank=cfg.distributed.rank,
                )
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[cfg.distributed.local_rank]
                )

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.loss_fn = build_loss_fn(
            cfg.train.loss_fn, cfg.train.label_smoothing, cfg.train.focal_gamma, cfg.train.focal_alpha
        ).to(self.device)
        self.gate_loss_fn = ConditionalGateLoss() if cfg.model.conditional_execution else None

        self.scaler = torch.amp.GradScaler("cuda") if cfg.train.mixed_precision and self.device.type == "cuda" else None

        self.ckpt_mgr = CheckpointManager(cfg.checkpoint)
        self.logger = TrainingLogger(cfg.log, cfg.experiment_name)
        self.profiler = PerformanceProfiler() if cfg.log.profile else None
        self.grad_inspector = GradientInspector(self.model) if cfg.log.gradient_inspection else None
        self.activation_hook: Optional[ActivationStatsHook] = None
        if cfg.log.activation_stats:
            self.activation_hook = ActivationStatsHook(self.model)

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.global_step = 0

        if cfg.checkpoint.resume:
            self._resume(cfg.checkpoint.resume)

    def _resume(self, path: str) -> None:
        print(f"Resuming from {path}")
        ckpt = CheckpointManager.load(path, self.model, self.optimizer, self.scheduler, self.scaler)
        self.start_epoch = ckpt["epoch"] + 1
        metrics = ckpt.get("metrics", {})
        self.best_val_loss = metrics.get("val_loss", float("inf"))
        print(f"Resumed at epoch {self.start_epoch}, best_val_loss={self.best_val_loss:.4f}")

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        accumulator = MetricsAccumulator(self.cfg.model.num_classes)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, targets) in enumerate(loader):
            if self.profiler:
                self.profiler.start("batch_forward")

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            amp_enabled = self.scaler is not None
            amp_dtype = torch.float16 if amp_enabled else torch.float32
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits, info = self.model(images)
                loss = self.loss_fn(logits, targets)
                if self.gate_loss_fn and "gate_probs" in info:
                    loss = loss + self.gate_loss_fn(info["gate_probs"])
                loss = loss / self.cfg.train.accumulation_steps

            if self.profiler:
                self.profiler.stop("batch_forward")
                self.profiler.start("batch_backward")

            if amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.cfg.train.accumulation_steps == 0:
                if self.cfg.train.gradient_clip_norm > 0:
                    if amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.gradient_clip_norm)
                if self.cfg.train.gradient_clip_value:
                    if amp_enabled and not hasattr(self, "_unscaled"):
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.model.parameters(), self.cfg.train.gradient_clip_value)

                if amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.profiler:
                self.profiler.stop("batch_backward")

            preds = logits.argmax(dim=-1)
            real_loss = loss.item() * self.cfg.train.accumulation_steps
            accumulator.update(preds, targets, real_loss, images.size(0))
            self.global_step += 1

            if self.global_step % self.cfg.log.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_scalar("train/loss_step", real_loss, self.global_step)
                self.logger.log_scalar("train/lr", lr, self.global_step)
                print(
                    f"  [Epoch {epoch} | Step {batch_idx + 1}/{len(loader)}] "
                    f"loss={real_loss:.4f}  lr={lr:.2e}"
                )

            if self.grad_inspector and self.global_step % (self.cfg.log.log_every * 5) == 0:
                anomalies = self.grad_inspector.check_anomalies()
                for a in anomalies:
                    print(f"  [GRADIENT ALERT] {a}")

        metrics = accumulator.compute_all()
        return {
            "train_loss": float(metrics["loss"]),
            "train_accuracy": float(metrics["accuracy"]),
            "train_macro_f1": float(metrics["macro_f1"]),
        }

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        accumulator = MetricsAccumulator(self.cfg.model.num_classes)

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            amp_enabled = self.scaler is not None
            amp_dtype = torch.float16 if amp_enabled else torch.float32
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, targets)

            preds = logits.argmax(dim=-1)
            accumulator.update(preds, targets, loss.item(), images.size(0))

        metrics = accumulator.compute_all()
        print(f"\n{'=' * 40} {prefix.upper()} {'=' * 40}")
        print(accumulator.summary_string())
        print(f"\nConfusion Matrix:\n{accumulator.confusion_matrix_string()}")

        return {
            f"{prefix}_loss": float(metrics["loss"]),
            f"{prefix}_accuracy": float(metrics["accuracy"]),
            f"{prefix}_macro_precision": float(metrics["macro_precision"]),
            f"{prefix}_macro_recall": float(metrics["macro_recall"]),
            f"{prefix}_macro_f1": float(metrics["macro_f1"]),
            f"{prefix}_weighted_f1": float(metrics["weighted_f1"]),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        print(f"\nTraining on device: {self.device}")
        print(f"Total epochs: {self.cfg.train.epochs}, starting from epoch {self.start_epoch}")

        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            if self.profiler:
                self.profiler.start("epoch")

            print(f"\n{'#' * 60}")
            print(f"  EPOCH {epoch + 1}/{self.cfg.train.epochs}")
            print(f"{'#' * 60}")

            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch(train_loader, epoch)
            val_metrics = self._evaluate(val_loader, "val")

            self.scheduler.step()

            all_metrics = {**train_metrics, **val_metrics}
            for k, v in all_metrics.items():
                self.logger.log_scalar(k, v, epoch)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            if (epoch + 1) % self.cfg.checkpoint.save_every == 0 or is_best:
                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch, all_metrics, is_best
                )

            if self.activation_hook:
                self.activation_hook.print_summary()

            if self.profiler:
                self.profiler.stop("epoch")

        if self.profiler:
            self.profiler.print_summary()

        self.logger.close()
        return self.logger.history

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        print("\nRunning final evaluation on test set...")
        return self._evaluate(test_loader, "test")
