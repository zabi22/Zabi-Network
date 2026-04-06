from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ModelConfig:
    num_classes: int = 10
    input_channels: int = 3
    input_height: int = 32
    input_width: int = 32
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    cnn_padding: List[int] = field(default_factory=lambda: [1, 1, 1])
    cnn_pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    rnn_type: str = "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_dropout: float = 0.1
    attn_embed_dim: int = 256
    attn_num_heads: int = 8
    attn_num_layers: int = 4
    attn_ff_dim: int = 1024
    attn_dropout: float = 0.1
    attn_use_gating: bool = True
    use_residual: bool = True
    global_dropout: float = 0.2
    width_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    conditional_execution: bool = False
    conditional_threshold: float = 0.5


@dataclass
class DataConfig:
    dataset_type: str = "synthetic"
    data_dir: str = "./data"
    num_train_samples: int = 50000
    num_val_samples: int = 10000
    num_test_samples: int = 10000
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    augmentation: bool = True
    augmentation_strength: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 100
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: str = "cosine_warm_restarts"
    scheduler_t0: int = 10
    scheduler_t_mult: int = 2
    scheduler_eta_min: float = 1e-6
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-7
    gradient_clip_norm: float = 1.0
    gradient_clip_value: Optional[float] = None
    mixed_precision: bool = True
    label_smoothing: float = 0.1
    loss_fn: str = "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    accumulation_steps: int = 1


@dataclass
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    use_data_parallel: bool = True


@dataclass
class CheckpointConfig:
    save_dir: str = "./checkpoints"
    save_every: int = 5
    keep_last: int = 3
    save_best: bool = True
    resume: Optional[str] = None


@dataclass
class LogConfig:
    log_dir: str = "./logs"
    log_every: int = 50
    use_tensorboard: bool = True
    print_model_summary: bool = True
    profile: bool = False
    gradient_inspection: bool = False
    activation_stats: bool = False
    plot_curves: bool = True


@dataclass
class NASConfig:
    enabled: bool = False
    search_space_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    search_space_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    search_space_heads: List[int] = field(default_factory=lambda: [2, 4, 8])
    search_space_rnn: List[str] = field(default_factory=lambda: ["lstm", "gru"])
    max_trials: int = 20


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: LogConfig = field(default_factory=LogConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    seed: int = 42
    experiment_name: str = "hybrid_net_experiment"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        data = self.to_dict()
        with open(path, "w") as f:
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                except ImportError:
                    json.dump(data, f, indent=2)
            else:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        with open(path, "r") as f:
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    data = json.load(f)
            else:
                data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        cfg = cls()
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "data" in data:
            for k, v in data["data"].items():
                if hasattr(cfg.data, k):
                    setattr(cfg.data, k, v)
        if "train" in data:
            for k, v in data["train"].items():
                if k == "betas" and isinstance(v, list):
                    v = tuple(v)
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)
        if "distributed" in data:
            for k, v in data["distributed"].items():
                if hasattr(cfg.distributed, k):
                    setattr(cfg.distributed, k, v)
        if "checkpoint" in data:
            for k, v in data["checkpoint"].items():
                if hasattr(cfg.checkpoint, k):
                    setattr(cfg.checkpoint, k, v)
        if "log" in data:
            for k, v in data["log"].items():
                if hasattr(cfg.log, k):
                    setattr(cfg.log, k, v)
        if "nas" in data:
            for k, v in data["nas"].items():
                if hasattr(cfg.nas, k):
                    setattr(cfg.nas, k, v)
        if "seed" in data:
            cfg.seed = data["seed"]
        if "experiment_name" in data:
            cfg.experiment_name = data["experiment_name"]
        return cfg

    def apply_width_depth_multipliers(self) -> "ExperimentConfig":
        cfg = copy.deepcopy(self)
        wm = cfg.model.width_multiplier
        dm = cfg.model.depth_multiplier
        cfg.model.cnn_channels = [max(1, int(c * wm)) for c in cfg.model.cnn_channels]
        cfg.model.rnn_hidden_size = max(1, int(cfg.model.rnn_hidden_size * wm))
        cfg.model.attn_embed_dim = max(1, int(cfg.model.attn_embed_dim * wm))
        cfg.model.attn_ff_dim = max(1, int(cfg.model.attn_ff_dim * wm))
        cfg.model.rnn_num_layers = max(1, int(cfg.model.rnn_num_layers * dm))
        cfg.model.attn_num_layers = max(1, int(cfg.model.attn_num_layers * dm))
        return cfg


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Neural Network Training System")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (YAML/JSON)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "nas", "profile"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--distributed", action="store_true", default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser


def load_config_from_cli(args: Optional[List[str]] = None) -> ExperimentConfig:
    parser = build_cli_parser()
    parsed = parser.parse_args(args)
    if parsed.config and os.path.exists(parsed.config):
        cfg = ExperimentConfig.load(parsed.config)
    else:
        cfg = ExperimentConfig()
    if parsed.resume is not None:
        cfg.checkpoint.resume = parsed.resume
    if parsed.epochs is not None:
        cfg.train.epochs = parsed.epochs
    if parsed.batch_size is not None:
        cfg.data.batch_size = parsed.batch_size
    if parsed.lr is not None:
        cfg.train.learning_rate = parsed.lr
    if parsed.seed is not None:
        cfg.seed = parsed.seed
    if parsed.experiment_name is not None:
        cfg.experiment_name = parsed.experiment_name
    if parsed.num_workers is not None:
        cfg.data.num_workers = parsed.num_workers
    if parsed.mixed_precision is not None:
        cfg.train.mixed_precision = parsed.mixed_precision
    if parsed.distributed is not None:
        cfg.distributed.enabled = parsed.distributed
    if parsed.local_rank is not None:
        cfg.distributed.local_rank = parsed.local_rank
    if parsed.save_dir is not None:
        cfg.checkpoint.save_dir = parsed.save_dir
    if parsed.log_dir is not None:
        cfg.log.log_dir = parsed.log_dir
    cfg._mode = parsed.mode
    return cfg
