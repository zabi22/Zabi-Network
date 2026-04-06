from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor


class MetricsAccumulator:
    """Tracks predictions and computes metrics."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._all_preds: List[Tensor] = []
        self._all_targets: List[Tensor] = []
        self._running_loss: float = 0.0
        self._num_samples: int = 0
        self._num_batches: int = 0

    def update(self, preds: Tensor, targets: Tensor, loss: float, batch_size: int) -> None:
        self._all_preds.append(preds.detach().cpu())
        self._all_targets.append(targets.detach().cpu())
        self._running_loss += loss * batch_size
        self._num_samples += batch_size
        self._num_batches += 1

    @property
    def avg_loss(self) -> float:
        return self._running_loss / max(self._num_samples, 1)

    def _gather(self) -> tuple[Tensor, Tensor]:
        preds = torch.cat(self._all_preds, dim=0)
        targets = torch.cat(self._all_targets, dim=0)
        return preds, targets

    def compute_confusion_matrix(self) -> Tensor:
        preds, targets = self._gather()
        cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        for t, p in zip(targets, preds):
            cm[t.long(), p.long()] += 1
        return cm

    def compute_accuracy(self) -> float:
        preds, targets = self._gather()
        correct = (preds == targets).sum().item()
        return correct / max(len(targets), 1)

    def compute_precision_recall_f1(self) -> Dict[str, Dict[str, float]]:
        cm = self.compute_confusion_matrix()
        results: Dict[str, Dict[str, float]] = {}
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        supports: List[int] = []

        for c in range(self.num_classes):
            tp = cm[c, c].item()
            fp = cm[:, c].sum().item() - tp
            fn = cm[c, :].sum().item() - tp
            support = cm[c, :].sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[f"class_{c}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

        total_support = sum(supports)
        macro_precision = sum(precisions) / max(self.num_classes, 1)
        macro_recall = sum(recalls) / max(self.num_classes, 1)
        macro_f1 = sum(f1s) / max(self.num_classes, 1)

        weighted_precision = (
            sum(p * s for p, s in zip(precisions, supports)) / max(total_support, 1)
        )
        weighted_recall = (
            sum(r * s for r, s in zip(recalls, supports)) / max(total_support, 1)
        )
        weighted_f1 = (
            sum(f * s for f, s in zip(f1s, supports)) / max(total_support, 1)
        )

        results["macro"] = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        }
        results["weighted"] = {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        }
        return results

    def compute_all(self) -> Dict[str, object]:
        accuracy = self.compute_accuracy()
        prf = self.compute_precision_recall_f1()
        cm = self.compute_confusion_matrix()
        return {
            "loss": self.avg_loss,
            "accuracy": accuracy,
            "macro_precision": prf["macro"]["precision"],
            "macro_recall": prf["macro"]["recall"],
            "macro_f1": prf["macro"]["f1"],
            "weighted_precision": prf["weighted"]["precision"],
            "weighted_recall": prf["weighted"]["recall"],
            "weighted_f1": prf["weighted"]["f1"],
            "per_class": {
                k: v for k, v in prf.items() if k.startswith("class_")
            },
            "confusion_matrix": cm,
        }

    def summary_string(self) -> str:
        m = self.compute_all()
        lines = [
            f"  Loss:               {m['loss']:.4f}",
            f"  Accuracy:           {m['accuracy']:.4f}",
            f"  Macro Precision:    {m['macro_precision']:.4f}",
            f"  Macro Recall:       {m['macro_recall']:.4f}",
            f"  Macro F1:           {m['macro_f1']:.4f}",
            f"  Weighted Precision: {m['weighted_precision']:.4f}",
            f"  Weighted Recall:    {m['weighted_recall']:.4f}",
            f"  Weighted F1:        {m['weighted_f1']:.4f}",
        ]
        return "\n".join(lines)

    def confusion_matrix_string(self) -> str:
        cm = self.compute_confusion_matrix()
        n = cm.size(0)
        header = "     " + "".join(f"{i:>6}" for i in range(n))
        lines = [header]
        for i in range(n):
            row = f"{i:>4} " + "".join(f"{cm[i, j].item():>6}" for j in range(n))
            lines.append(row)
        return "\n".join(lines)
