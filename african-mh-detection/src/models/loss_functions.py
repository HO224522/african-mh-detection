"""
loss_functions.py
=================
Loss functions spécialisées pour l'entraînement multi-tâches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss pour classes déséquilibrées.
    Utile pour la détection de détresse (cas positifs rares).

    Reference: Lin et al. 2017 (RetinaNet)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss pour optimiser rappel > précision.
    Pénalise plus les faux négatifs que les faux positifs.

    Utile en santé mentale : manquer un signal = danger.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Clip négatifs pour stabilité
        probs_neg = (probs + self.clip).clamp(max=1)

        # Loss asymétrique
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        loss = loss_pos * (1 - probs) ** self.gamma_pos + loss_neg * probs_neg ** self.gamma_neg
        return -loss.mean()
