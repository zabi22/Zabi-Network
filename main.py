from __future__ import annotations

import copy
import random
from typing import Dict, List

import torch

from config import ExperimentConfig, ModelConfig, load_config_from_cli
from data import build_dataloaders
from model import build_model
from trainer import Trainer
from utils import plot_training_curves, set_seed


def run_train(cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed)
    effective_cfg = cfg.model
    if cfg.model.width_multiplier != 1.0 or cfg.model.depth_multiplier != 1.0:
        cfg = cfg.apply_width_depth_multipliers()
        effective_cfg = cfg.model

    model = build_model(effective_cfg)
    if cfg.log.print_model_summary:
        model.print_summary()

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data, effective_cfg, distributed=cfg.distributed.enabled, seed=cfg.seed
    )

    trainer = Trainer(model, cfg)
    history = trainer.train(train_loader, val_loader)

    test_metrics = trainer.test(test_loader)
    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if cfg.log.plot_curves and history:
        plot_training_curves(history, str(trainer.logger.log_dir / "training_curves.png"))

    cfg.save(str(trainer.logger.log_dir / "final_config.json"))


def run_eval(cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed)
    if cfg.model.width_multiplier != 1.0 or cfg.model.depth_multiplier != 1.0:
        cfg = cfg.apply_width_depth_multipliers()

    model = build_model(cfg.model)
    _, _, test_loader = build_dataloaders(cfg.data, cfg.model, seed=cfg.seed)

    trainer = Trainer(model, cfg)
    test_metrics = trainer.test(test_loader)
    print("\nEvaluation Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


def run_nas(cfg: ExperimentConfig) -> None:
    """Random NAS search."""
    set_seed(cfg.seed)
    nas = cfg.nas
    best_score = -float("inf")
    best_trial: Dict = {}

    print(f"\nStarting NAS with {nas.max_trials} trials")
    print(f"Search space: channels={nas.search_space_channels}, "
          f"layers={nas.search_space_layers}, heads={nas.search_space_heads}")

    for trial in range(nas.max_trials):
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg.model.cnn_channels = sorted(
            random.choices(nas.search_space_channels, k=len(cfg.model.cnn_channels))
        )
        trial_cfg.model.attn_num_layers = random.choice(nas.search_space_layers)
        trial_cfg.model.rnn_num_layers = random.choice(nas.search_space_layers)
        trial_cfg.model.attn_num_heads = random.choice(nas.search_space_heads)
        trial_cfg.model.rnn_type = random.choice(nas.search_space_rnn)
        trial_cfg.model.attn_embed_dim = max(
            trial_cfg.model.attn_num_heads,
            random.choice([128, 256, 512])
        )
        trial_cfg.model.attn_embed_dim -= trial_cfg.model.attn_embed_dim % trial_cfg.model.attn_num_heads

        trial_cfg.train.epochs = min(5, cfg.train.epochs)
        trial_cfg.experiment_name = f"nas_trial_{trial}"

        print(f"\n{'=' * 40} NAS Trial {trial + 1}/{nas.max_trials} {'=' * 40}")
        print(f"  CNN channels: {trial_cfg.model.cnn_channels}")
        print(f"  Attn layers: {trial_cfg.model.attn_num_layers}, heads: {trial_cfg.model.attn_num_heads}")
        print(f"  RNN type: {trial_cfg.model.rnn_type}, layers: {trial_cfg.model.rnn_num_layers}")

        try:
            model = build_model(trial_cfg.model)
            train_loader, val_loader, _ = build_dataloaders(
                trial_cfg.data, trial_cfg.model, seed=cfg.seed + trial
            )
            trainer = Trainer(model, trial_cfg)
            trainer.train(train_loader, val_loader)
            val_metrics = trainer._evaluate(val_loader, "val")
            score = val_metrics.get("val_macro_f1", 0.0)
            print(f"  Trial score (val_macro_f1): {score:.4f}")

            if score > best_score:
                best_score = score
                best_trial = {
                    "trial": trial,
                    "score": score,
                    "config": trial_cfg.to_dict(),
                }
        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            continue

    if best_trial:
        print(f"\nBest NAS trial: {best_trial['trial']} with score {best_trial['score']:.4f}")
        best_cfg = ExperimentConfig._from_dict(best_trial["config"])
        best_cfg.save("./nas_best_config.json")
        print("Best config saved to ./nas_best_config.json")


def run_profile(cfg: ExperimentConfig) -> None:
    """Profile forward/backward pass."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.width_multiplier != 1.0 or cfg.model.depth_multiplier != 1.0:
        cfg = cfg.apply_width_depth_multipliers()

    model = build_model(cfg.model).to(device)
    model.print_summary()

    x = torch.randn(cfg.data.batch_size, cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width).to(device)
    targets = torch.randint(0, cfg.model.num_classes, (cfg.data.batch_size,)).to(device)

    model.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits, info = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        print(f"\nForward + Backward: {elapsed:.2f} ms")
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory: {mem:.1f} MB")
    else:
        import time
        start = time.perf_counter()
        logits, info = model(x)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\nForward + Backward: {elapsed:.2f} ms")

    print(f"Output shape: {logits.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def main() -> None:
    cfg = load_config_from_cli()
    mode = getattr(cfg, "_mode", "train")

    print(f"\nExperiment: {cfg.experiment_name}")
    print(f"Mode: {mode}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if mode == "train":
        run_train(cfg)
    elif mode == "eval":
        run_eval(cfg)
    elif mode == "nas":
        run_nas(cfg)
    elif mode == "profile":
        run_profile(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
