from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ManualLayerNorm(nn.Module):
    """Layer norm from scratch."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm


class ManualDropout(nn.Module):
    """Dropout from scratch."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1.0 - self.p)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = ManualDropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with manual projections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.w_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.w_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.w_v = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.w_o = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_q = nn.Parameter(torch.zeros(embed_dim))
        self.b_k = nn.Parameter(torch.zeros(embed_dim))
        self.b_v = nn.Parameter(torch.zeros(embed_dim))
        self.b_o = nn.Parameter(torch.zeros(embed_dim))

        self._init_weights()
        self.attention = ScaledDotProductAttention(dropout)

    def _init_weights(self) -> None:
        for w in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(w)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch, seq_len, _ = x.size()

        q = torch.matmul(x, self.w_q) + self.b_q
        k = torch.matmul(x, self.w_k) + self.b_k
        v = torch.matmul(x, self.w_v) + self.b_v

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out, attn_weights = self.attention(q, k, v, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = torch.matmul(attn_out, self.w_o) + self.b_o
        return output, attn_weights


class AttentionGate(nn.Module):
    """Gates attention output with learned sigmoid."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_gate = nn.Parameter(torch.zeros(embed_dim))
        self.w_skip = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_skip = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_skip)

    def forward(self, x: Tensor, attn_out: Tensor) -> Tensor:
        gate = torch.sigmoid(torch.matmul(x, self.w_gate) + self.b_gate)
        skip = torch.matmul(x, self.w_skip) + self.b_skip
        return gate * attn_out + (1.0 - gate) * skip


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward with GELU."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.b1 = nn.Parameter(torch.zeros(ff_dim))
        self.w2 = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.b2 = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        self.dropout = ManualDropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.matmul(x, self.w1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.matmul(h, self.w2) + self.b2
        return h


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 8192, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = ManualDropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with optional gating and residuals."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_gating = use_gating
        self.use_residual = use_residual
        self.norm1 = ManualLayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.gate = AttentionGate(embed_dim) if use_gating else None
        self.dropout1 = ManualDropout(dropout)
        self.norm2 = ManualLayerNorm(embed_dim)
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.dropout2 = ManualDropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, mask)
        if self.use_gating:
            attn_out = self.gate(x_norm, attn_out)
        attn_out = self.dropout1(attn_out)
        if self.use_residual:
            x = residual + attn_out
        else:
            x = attn_out

        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        ff_out = self.dropout2(ff_out)
        if self.use_residual:
            x = residual + ff_out
        else:
            x = ff_out
        return x, attn_weights


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> Pool with residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut: nn.Module
        if use_residual:
            if in_channels != out_channels or stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        if self.use_residual:
            shortcut = self.shortcut(identity)
            if shortcut.shape[2:] != out.shape[2:]:
                shortcut = F.adaptive_avg_pool2d(shortcut, out.shape[2:])
            out = out + shortcut
        out = self.act(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class RecurrentEncoder(nn.Module):
    """Bidirectional LSTM or GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = ManualLayerNorm(hidden_size * (2 if bidirectional else 1))

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.rnn(x)
        output = self.norm(output)
        return output


class ConditionalExecutionGate(nn.Module):
    """Learns per-sample gate to skip layers."""

    def __init__(self, embed_dim: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        gate_logit = self.fc(x.mean(dim=1))
        gate_prob = torch.sigmoid(gate_logit).squeeze(-1)
        if self.training:
            gate = (gate_prob > self.threshold).float() + gate_prob - gate_prob.detach()
        else:
            gate = (gate_prob > self.threshold).float()
        return gate, gate_prob
