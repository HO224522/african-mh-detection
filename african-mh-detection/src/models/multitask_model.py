"""
multitask_model.py
==================
Architecture NLP multi-tâches pour la détection de détresse psychologique.

Modèle: XLM-RoBERTa fine-tuné avec 3 têtes de prédiction simultanées:
  - Task 1: Détection signal binaire (détresse / pas détresse)
  - Task 2: Catégorisation (dépression, anxiété, idées noires, isolement, épuisement)
  - Task 3: Contexte (facteurs: deuil, chômage, violence, maladie, solitude)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


# ─── Constantes ──────────────────────────────────────────────────────────────

BACKBONE_MODEL = "xlm-roberta-base"

CATEGORIES = [
    "depression",
    "anxiety",
    "suicidal_ideation",
    "isolation",
    "burnout",
]

CONTEXT_FACTORS = [
    "grief",
    "unemployment",
    "violence",
    "illness",
    "loneliness",
    "family_conflict",
    "financial_stress",
]

# Poids de la loss combinée (Signal:Categ:Context = 50:30:20)
LOSS_WEIGHTS = {"task1_signal": 0.50, "task2_category": 0.30, "task3_context": 0.20}


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    """Sortie complète du modèle multi-tâches."""

    signal_logits: torch.Tensor          # (batch, 2)
    category_logits: torch.Tensor        # (batch, len(CATEGORIES))
    context_logits: torch.Tensor         # (batch, len(CONTEXT_FACTORS))
    hidden_states: Optional[torch.Tensor] = None  # (batch, seq_len, hidden)


@dataclass
class TrainingConfig:
    """Configuration d'entraînement."""

    backbone: str = BACKBONE_MODEL
    max_length: int = 256
    dropout: float = 0.1
    hidden_dim: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 30
    warmup_steps: int = 200
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    freeze_backbone_epochs: int = 3  # Phase 1: geler backbone


# ─── Architecture ────────────────────────────────────────────────────────────

class MultiTaskDistressDetector(nn.Module):
    """
    Modèle NLP multi-tâches pour détecter la détresse psychologique.

    Architecture:
        XLM-RoBERTa (backbone partagé)
             │
        SharedLayers (dense + dropout)
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
    Task1  Task2  Task3
    (bin) (multi) (multi)

    Args:
        config: TrainingConfig avec hyperparamètres
        num_categories: Nombre de catégories de détresse
        num_context_factors: Nombre de facteurs contextuels
    """

    def __init__(
        self,
        config: TrainingConfig = TrainingConfig(),
        num_categories: int = len(CATEGORIES),
        num_context_factors: int = len(CONTEXT_FACTORS),
    ):
        super().__init__()
        self.config = config

        # Backbone XLM-RoBERTa
        logger.info(f"Chargement backbone: {config.backbone}")
        self.backbone = AutoModel.from_pretrained(config.backbone)
        hidden_size = self.backbone.config.hidden_size  # 768 pour base

        # Couches partagées
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Têtes de prédiction (task-specific)
        self.task1_head = nn.Linear(config.hidden_dim, 2)          # Signal binaire
        self.task2_head = nn.Linear(config.hidden_dim, num_categories)   # Catégories
        self.task3_head = nn.Linear(config.hidden_dim, num_context_factors)  # Contexte

        # Initialisation Xavier
        self._init_weights()

    def _init_weights(self):
        """Initialisation des poids des couches custom (pas backbone)."""
        for module in [self.shared_layers, self.task1_head,
                       self.task2_head, self.task3_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def freeze_backbone(self):
        """Geler le backbone (Phase 1 entraînement)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone gelé (Phase 1)")

    def unfreeze_backbone(self, layers_from_top: int = 4):
        """Dégeler progressivement les dernières couches (Phase 2)."""
        # D'abord tout geler
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Puis dégeler les N dernières couches
        encoder_layers = self.backbone.encoder.layer
        for layer in encoder_layers[-layers_from_top:]:
            for param in layer.parameters():
                param.requires_grad = True

        logger.info(f"Backbone partiellement dégelé: {layers_from_top} dernières couches")

    def unfreeze_all(self):
        """Dégeler tout le modèle (Phase 3)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Modèle complet dégelé (Phase 3)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            input_ids: Tokens (batch, seq_len)
            attention_mask: Masques (batch, seq_len)
            token_type_ids: Optionnel, types de tokens

        Returns:
            ModelOutput avec logits des 3 tâches
        """
        # Backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Représentation [CLS] (pooler_output) — résumé de la séquence
        cls_output = outputs.pooler_output  # (batch, 768)

        # Couches partagées
        shared = self.shared_layers(cls_output)  # (batch, 256)

        # Têtes task-specific
        signal_logits = self.task1_head(shared)      # (batch, 2)
        category_logits = self.task2_head(shared)    # (batch, 5)
        context_logits = self.task3_head(shared)     # (batch, 7)

        return ModelOutput(
            signal_logits=signal_logits,
            category_logits=category_logits,
            context_logits=context_logits,
            hidden_states=outputs.last_hidden_state,
        )


# ─── Loss Function ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Loss combinée pondérée pour l'entraînement multi-tâches.

    Loss = 0.5 * L_signal + 0.3 * L_category + 0.2 * L_context

    Justification des poids:
    - Signal (50%): Tâche principale, la plus critique
    - Catégorie (30%): Important pour le routage
    - Contexte (20%): Enrichit l'interprétation
    """

    def __init__(
        self,
        weights: Dict[str, float] = LOSS_WEIGHTS,
        class_weight_signal: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weights = weights
        # BCE pour signal (binaire)
        self.loss_signal = nn.CrossEntropyLoss(weight=class_weight_signal)
        # BCE multi-label pour catégorie et contexte
        self.loss_category = nn.BCEWithLogitsLoss()
        self.loss_context = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: ModelOutput,
        labels_signal: torch.Tensor,
        labels_category: torch.Tensor,
        labels_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcul de la loss totale.

        Returns:
            total_loss: Loss scalaire pour backprop
            individual_losses: Dict avec losses par tâche (pour monitoring)
        """
        l1 = self.loss_signal(outputs.signal_logits, labels_signal)
        l2 = self.loss_category(
            outputs.category_logits, labels_category.float()
        )
        l3 = self.loss_context(
            outputs.context_logits, labels_context.float()
        )

        total = (
            self.weights["task1_signal"] * l1
            + self.weights["task2_category"] * l2
            + self.weights["task3_context"] * l3
        )

        return total, {"loss_signal": l1, "loss_category": l2, "loss_context": l3}


# ─── Exemple d'utilisation ───────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = TrainingConfig(backbone=BACKBONE_MODEL, hidden_dim=256)
    model = MultiTaskDistressDetector(config)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL)
    texts = [
        "Je ne vois plus d'avenir, tout s'effondre",
        "La vie est belle aujourd'hui!",
        "M bi kɛ dɔ, n ma se ka fɛ",  # Moore (example)
    ]

    encoded = tokenizer(
        texts, max_length=256, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**encoded)

    print("Signal logits shape:", output.signal_logits.shape)
    print("Category logits shape:", output.category_logits.shape)
    print("Context logits shape:", output.context_logits.shape)

    # Prédictions
    signal_probs = torch.softmax(output.signal_logits, dim=-1)
    print("\nSignal probabilities (distress):", signal_probs[:, 1].tolist())
