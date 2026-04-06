from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.layers import (
    ConvBlock,
    RecurrentEncoder,
    TransformerEncoderBlock,
    SinusoidalPositionalEncoding,
    ManualLayerNorm,
    ManualDropout,
    ConditionalExecutionGate,
)
from core.config import ModelConfig


class CNNFeatureExtractor(nn.Module):
    """Stack of convolutional blocks producing spatial feature maps."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        channels = [cfg.input_channels] + cfg.cnn_channels
        blocks: List[nn.Module] = []
        for i in range(len(cfg.cnn_channels)):
            blocks.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=cfg.cnn_kernel_sizes[i],
                    stride=cfg.cnn_strides[i],
                    padding=cfg.cnn_padding[i],
                    pool_size=cfg.cnn_pool_sizes[i],
                    use_residual=cfg.use_residual,
                    dropout=cfg.global_dropout * 0.5,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class SpatialToSequence(nn.Module):
    """Reshapes CNN spatial features (B, C, H, W) into a sequence (B, H*W, C)."""

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.size()
        return x.view(b, c, h * w).permute(0, 2, 1)


class SequenceProjection(nn.Module):
    """Projects sequence features to a target embedding dimension."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class TransformerStack(nn.Module):
    """Stack of transformer encoder blocks with positional encoding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(cfg.attn_embed_dim, dropout=cfg.attn_dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=cfg.attn_embed_dim,
                    num_heads=cfg.attn_num_heads,
                    ff_dim=cfg.attn_ff_dim,
                    dropout=cfg.attn_dropout,
                    use_gating=cfg.attn_use_gating,
                    use_residual=cfg.use_residual,
                )
                for _ in range(cfg.attn_num_layers)
            ]
        )
        self.norm = ManualLayerNorm(cfg.attn_embed_dim)
        self.conditional = cfg.conditional_execution
        if self.conditional:
            self.gates = nn.ModuleList(
                [
                    ConditionalExecutionGate(cfg.attn_embed_dim, cfg.conditional_threshold)
                    for _ in range(cfg.attn_num_layers)
                ]
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        x = self.pos_enc(x)
        all_attn: List[Tensor] = []
        gate_probs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            if self.conditional:
                gate, gp = self.gates[i](x)
                gate_probs.append(gp)
                layer_out, attn_w = layer(x)
                x = gate.unsqueeze(1).unsqueeze(2) * layer_out + (1.0 - gate.unsqueeze(1).unsqueeze(2)) * x
            else:
                x, attn_w = layer(x)
            all_attn.append(attn_w)
        x = self.norm(x)
        info = {"attention_weights": all_attn}
        if gate_probs:
            info["gate_probs"] = gate_probs
        return x, info


class HybridNet(nn.Module):
    """Hybrid CNN + RNN + Transformer for image classification."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.cnn = CNNFeatureExtractor(cfg)
        self.spatial_to_seq = SpatialToSequence()

        cnn_out_channels = cfg.cnn_channels[-1]
        test_input = torch.zeros(1, cfg.input_channels, cfg.input_height, cfg.input_width)
        with torch.no_grad():
            test_out = self.cnn(test_input)
        _, c_out, h_out, w_out = test_out.shape
        seq_len = h_out * w_out
        cnn_feat_dim = c_out

        rnn_input_size = cnn_feat_dim
        self.rnn = RecurrentEncoder(
            input_size=rnn_input_size,
            hidden_size=cfg.rnn_hidden_size,
            num_layers=cfg.rnn_num_layers,
            rnn_type=cfg.rnn_type,
            bidirectional=cfg.rnn_bidirectional,
            dropout=cfg.rnn_dropout,
        )
        rnn_out_dim = cfg.rnn_hidden_size * (2 if cfg.rnn_bidirectional else 1)

        self.seq_proj = SequenceProjection(rnn_out_dim, cfg.attn_embed_dim)

        self.transformer = TransformerStack(cfg)

        self.pool_attention = nn.Linear(cfg.attn_embed_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.attn_embed_dim, cfg.attn_embed_dim // 2),
            nn.GELU(),
            ManualDropout(cfg.global_dropout),
            nn.Linear(cfg.attn_embed_dim // 2, cfg.num_classes),
        )

        self._activation_stats: Dict[str, Dict[str, float]] = {}

    def _attention_pool(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.pool_attention(x).squeeze(-1), dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        cnn_out = self.cnn(x)
        seq = self.spatial_to_seq(cnn_out)

        rnn_out = self.rnn(seq)

        projected = self.seq_proj(rnn_out)

        transformer_out, info = self.transformer(projected)

        pooled = self._attention_pool(transformer_out)

        logits = self.classifier(pooled)

        if self.training:
            self._activation_stats = {
                "cnn_out": {"mean": cnn_out.mean().item(), "std": cnn_out.std().item()},
                "rnn_out": {"mean": rnn_out.mean().item(), "std": rnn_out.std().item()},
                "transformer_out": {
                    "mean": transformer_out.mean().item(),
                    "std": transformer_out.std().item(),
                },
                "pooled": {"mean": pooled.mean().item(), "std": pooled.std().item()},
            }
            info["activation_stats"] = self._activation_stats

        return logits, info

    def get_param_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

    def print_summary(self) -> None:
        counts = self.get_param_count()
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        for name, count in counts.items():
            label = name.upper() if name in ("total", "trainable") else name
            print(f"  {label:30s} : {count:>12,} params")
        print("=" * 60 + "\n")


def build_model(cfg: ModelConfig) -> HybridNet:
    effective_cfg = cfg
    return HybridNet(effective_cfg)
