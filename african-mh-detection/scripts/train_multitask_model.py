"""
train_multitask_model.py
========================
Pipeline d'entraînement multi-tâches en 3 phases:

  Phase 1: Geler backbone, entraîner les têtes (3 epochs)
  Phase 2: Dégeler progressivement (layers_from_top=4)
  Phase 3: Fine-tuning complet du modèle entier

Usage:
    python scripts/train_multitask_model.py \
        --corpus data/corpus/annotated_texts.csv \
        --output data/models/ \
        --epochs 30
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_linear_schedule_with_warmup

from src.models.multitask_model import MultiTaskDistressDetector, MultiTaskLoss, TrainingConfig
from src.utils.data_loader import load_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    """Une epoch d'entraînement."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_signal = batch["label_signal"].to(device)
        label_category = batch["label_category"].to(device)
        label_context = batch["label_context"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss, individual = loss_fn(outputs, label_signal, label_category, label_context)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            logger.info(f"  Batch {i+1}/{n_batches} | Loss: {loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluation sur un dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_signal = batch["label_signal"].to(device)
        label_category = batch["label_category"].to(device)
        label_context = batch["label_context"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, _ = loss_fn(outputs, label_signal, label_category, label_context)
        total_loss += loss.item()

        preds = outputs.signal_logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(label_signal.cpu().tolist())

    # F2-score sur signal (Task 1)
    from sklearn.metrics import fbeta_score
    f2 = fbeta_score(all_labels, all_preds, beta=2, zero_division=0)

    return total_loss / len(loader), f2


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Data
    logger.info("Chargement corpus...")
    train_loader, val_loader, test_loader = load_corpus(
        args.corpus,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # Modèle et loss
    model = MultiTaskDistressDetector(config).to(device)
    loss_fn = MultiTaskLoss()

    os.makedirs(args.output, exist_ok=True)
    best_f2 = 0.0

    # ─── Phase 1: Backbone gelé (3 epochs) ──────────────────────────
    logger.info("=== Phase 1: Têtes uniquement (backbone gelé) ===")
    model.freeze_backbone()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(config.freeze_backbone_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_f2 = evaluate(model, val_loader, loss_fn, device)
        logger.info(f"Phase1 Epoch {epoch+1} | train_loss={train_loss:.4f} | val_f2={val_f2:.4f}")

    # ─── Phase 2: Dégel progressif (6 epochs) ────────────────────────
    logger.info("=== Phase 2: Dégel progressif (4 couches) ===")
    model.unfreeze_backbone(layers_from_top=4)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(6):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_f2 = evaluate(model, val_loader, loss_fn, device)
        logger.info(f"Phase2 Epoch {epoch+1} | train_loss={train_loss:.4f} | val_f2={val_f2:.4f}")

        if val_f2 > best_f2:
            best_f2 = val_f2
            torch.save(model.state_dict(), Path(args.output) / "best_model.pt")

    # ─── Phase 3: Fine-tuning complet ────────────────────────────────
    logger.info(f"=== Phase 3: Fine-tuning complet ({args.epochs} epochs) ===")
    model.unfreeze_all()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    remaining = args.epochs - config.freeze_backbone_epochs - 6
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=remaining * len(train_loader),
    )

    for epoch in range(remaining):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        val_loss, val_f2 = evaluate(model, val_loader, loss_fn, device)
        logger.info(f"Phase3 Epoch {epoch+1} | train_loss={train_loss:.4f} | val_f2={val_f2:.4f}")

        if val_f2 > best_f2:
            best_f2 = val_f2
            torch.save(model.state_dict(), Path(args.output) / "best_model.pt")
            logger.info(f"  ✅ Nouveau meilleur modèle: F2={best_f2:.4f}")

    # Evaluation finale sur test set
    model.load_state_dict(torch.load(Path(args.output) / "best_model.pt"))
    test_loss, test_f2 = evaluate(model, test_loader, loss_fn, device)
    logger.info(f"\n=== RÉSULTAT FINAL ===")
    logger.info(f"Test F2-Score: {test_f2:.4f}")
    logger.info(f"Modèle sauvegardé: {args.output}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner le modèle multi-tâches")
    parser.add_argument("--corpus", type=str, required=True, help="Chemin vers le CSV annoté")
    parser.add_argument("--output", type=str, default="data/models/", help="Dossier de sortie")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    main(args)
