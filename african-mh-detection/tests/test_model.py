"""
test_model.py
=============
Tests unitaires pour le modèle multi-tâches.
"""

import pytest
import torch

from src.models.multitask_model import (
    CATEGORIES,
    CONTEXT_FACTORS,
    MultiTaskDistressDetector,
    MultiTaskLoss,
    TrainingConfig,
)


@pytest.fixture
def model():
    """Fixture modèle avec config minimale."""
    config = TrainingConfig(backbone="xlm-roberta-base", hidden_dim=64)
    return MultiTaskDistressDetector(config)


@pytest.fixture
def dummy_batch():
    """Batch fictif pour les tests."""
    batch_size, seq_len = 2, 32
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


class TestMultiTaskDistressDetector:
    def test_output_shapes(self, model, dummy_batch):
        """Vérifie les dimensions de sortie."""
        with torch.no_grad():
            output = model(**dummy_batch)

        batch_size = dummy_batch["input_ids"].shape[0]
        assert output.signal_logits.shape == (batch_size, 2)
        assert output.category_logits.shape == (batch_size, len(CATEGORIES))
        assert output.context_logits.shape == (batch_size, len(CONTEXT_FACTORS))

    def test_freeze_backbone(self, model):
        """Vérifie le gel du backbone."""
        model.freeze_backbone()
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Param {name} should be frozen"

        # Les têtes restent entraînables
        for param in model.task1_head.parameters():
            assert param.requires_grad

    def test_unfreeze_all(self, model):
        """Vérifie le dégel complet."""
        model.freeze_backbone()
        model.unfreeze_all()
        for param in model.parameters():
            assert param.requires_grad

    def test_forward_no_error(self, model, dummy_batch):
        """Le forward pass ne doit pas lever d'erreur."""
        with torch.no_grad():
            output = model(**dummy_batch)
        assert output is not None


class TestMultiTaskLoss:
    def test_loss_is_scalar(self, model, dummy_batch):
        """La loss doit être un scalaire."""
        loss_fn = MultiTaskLoss()
        with torch.no_grad():
            output = model(**dummy_batch)

        batch_size = dummy_batch["input_ids"].shape[0]
        labels_signal = torch.randint(0, 2, (batch_size,))
        labels_category = torch.randint(0, 2, (batch_size, len(CATEGORIES))).float()
        labels_context = torch.randint(0, 2, (batch_size, len(CONTEXT_FACTORS))).float()

        total_loss, individual = loss_fn(output, labels_signal, labels_category, labels_context)

        assert total_loss.dim() == 0  # scalaire
        assert "loss_signal" in individual
        assert "loss_category" in individual
        assert "loss_context" in individual

    def test_loss_positive(self, model, dummy_batch):
        """La loss doit être positive."""
        loss_fn = MultiTaskLoss()
        with torch.no_grad():
            output = model(**dummy_batch)

        batch_size = dummy_batch["input_ids"].shape[0]
        total_loss, _ = loss_fn(
            output,
            torch.randint(0, 2, (batch_size,)),
            torch.randint(0, 2, (batch_size, len(CATEGORIES))).float(),
            torch.randint(0, 2, (batch_size, len(CONTEXT_FACTORS))).float(),
        )
        assert total_loss.item() > 0
