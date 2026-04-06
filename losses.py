from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        num_classes = logits.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * log_probs * smooth_targets

        if self.alpha is not None:
            alpha_tensor = torch.full_like(loss, self.alpha)
            loss = alpha_tensor * loss

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """CE loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """Combined hard and soft label loss."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_loss = LabelSmoothingCrossEntropy(label_smoothing)

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, targets: Tensor
    ) -> Tensor:
        hard = self.hard_loss(student_logits, targets)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
            self.temperature ** 2
        )
        return self.alpha * soft + (1.0 - self.alpha) * hard


class ConditionalGateLoss(nn.Module):
    """Regularizer for conditional execution gates."""

    def __init__(self, target_rate: float = 0.5, weight: float = 0.01) -> None:
        super().__init__()
        self.target_rate = target_rate
        self.weight = weight

    def forward(self, gate_probs: list[Tensor]) -> Tensor:
        loss = torch.tensor(0.0, device=gate_probs[0].device)
        for gp in gate_probs:
            mean_gate = gp.mean()
            loss = loss + (mean_gate - self.target_rate) ** 2
        return self.weight * loss / max(len(gate_probs), 1)


def build_loss_fn(
    loss_name: str,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None,
) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
    elif loss_name == "label_smoothing_ce":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
