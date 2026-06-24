"""
data_loader.py
==============
Chargement et préparation du corpus pour l'entraînement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

CATEGORIES = ["depression", "anxiety", "suicidal_ideation", "isolation", "burnout"]
CONTEXT_FACTORS = ["grief", "unemployment", "violence", "illness", "loneliness",
                   "family_conflict", "financial_stress"]


class DistressDataset(Dataset):
    """
    Dataset PyTorch pour le corpus de détresse psychologique.

    Format CSV attendu:
        text, task1_signal, task2_category, task3_context, dialect
        "Je souffre...", 1, depression|isolation, loneliness, french
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Tokenisation
        encoded = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels Task 1 (binaire)
        label_signal = torch.tensor(int(row["task1_signal"]), dtype=torch.long)

        # Labels Task 2 (multi-label)
        cats = str(row.get("task2_category", "")).split("|")
        label_category = torch.zeros(len(CATEGORIES))
        for c in cats:
            if c.strip() in CATEGORIES:
                label_category[CATEGORIES.index(c.strip())] = 1.0

        # Labels Task 3 (multi-label)
        ctxs = str(row.get("task3_context", "")).split("|")
        label_context = torch.zeros(len(CONTEXT_FACTORS))
        for c in ctxs:
            if c.strip() in CONTEXT_FACTORS:
                label_context[CONTEXT_FACTORS.index(c.strip())] = 1.0

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_signal": label_signal,
            "label_category": label_category,
            "label_context": label_context,
        }


def load_corpus(
    csv_path: str,
    tokenizer_name: str = "xlm-roberta-base",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 16,
    max_length: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Charge le corpus et retourne train/val/test DataLoaders.

    Args:
        csv_path: Chemin vers le CSV annoté
        tokenizer_name: Nom du tokenizer HuggingFace
        test_size: Fraction pour le test set
        val_size: Fraction pour le validation set
        batch_size: Taille des batches
        max_length: Longueur max des séquences
        seed: Graine aléatoire

    Returns:
        (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Corpus chargé: {len(df)} exemples depuis {csv_path}")

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_df = df[:n_test]
    val_df = df[n_test:n_test + n_val]
    train_df = df[n_test + n_val:]

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = DistressDataset(train_df, tokenizer, max_length)
    val_ds = DistressDataset(val_df, tokenizer, max_length)
    test_ds = DistressDataset(test_df, tokenizer, max_length)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )
