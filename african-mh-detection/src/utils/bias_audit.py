"""
bias_audit.py
=============
Audit de fairness mensuel par dialecte.
Vérifie que le modèle est équitable entre French, Moore, Dioula.

Alerte si écart de performance > 5% entre dialectes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

DIALECTS = ["french", "moore", "dioula", "mix"]
MAX_BIAS_GAP = 0.05  # Alerte si écart > 5%


@dataclass
class AuditReport:
    """Rapport d'audit de fairness."""

    date: str
    metrics_by_dialect: Dict[str, Dict[str, float]]
    max_gap: float
    passes_fairness: bool
    alerts: List[str]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcul des métriques de performance."""
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2),
    }


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2.0) -> float:
    """F-beta score (F2 par défaut — rappel prioritaire)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def run_bias_audit(
    df: pd.DataFrame,
    predictions: np.ndarray,
    dialect_col: str = "dialect",
    label_col: str = "task1_signal",
) -> AuditReport:
    """
    Audit complet de biais par dialecte.

    Args:
        df: DataFrame avec colonnes dialect et label
        predictions: Prédictions du modèle (0/1)
        dialect_col: Nom colonne dialecte
        label_col: Nom colonne labels vrais

    Returns:
        AuditReport avec métriques et alertes
    """
    from datetime import datetime

    metrics_by_dialect: Dict[str, Dict[str, float]] = {}
    alerts: List[str] = []

    for dialect in DIALECTS:
        mask = df[dialect_col] == dialect
        if mask.sum() < 5:
            logger.warning(f"Trop peu d'exemples pour {dialect}: {mask.sum()}")
            continue

        y_true = df.loc[mask, label_col].values
        y_pred = predictions[mask]
        metrics_by_dialect[dialect] = compute_metrics(y_true, y_pred)

    # Calcul des écarts
    f2_scores = [m["f2"] for m in metrics_by_dialect.values()]
    max_gap = max(f2_scores) - min(f2_scores) if len(f2_scores) > 1 else 0.0

    passes = max_gap <= MAX_BIAS_GAP

    if not passes:
        worst_dialect = min(metrics_by_dialect, key=lambda d: metrics_by_dialect[d]["f2"])
        alerts.append(
            f"⚠️ Écart de biais trop élevé: {max_gap:.1%} > {MAX_BIAS_GAP:.0%}. "
            f"Dialecte le moins performant: {worst_dialect} "
            f"(F2={metrics_by_dialect[worst_dialect]['f2']:.3f})"
        )

    report = AuditReport(
        date=datetime.now().strftime("%Y-%m-%d"),
        metrics_by_dialect=metrics_by_dialect,
        max_gap=round(max_gap, 4),
        passes_fairness=passes,
        alerts=alerts,
    )

    # Log résumé
    logger.info(f"Audit biais — Écart max: {max_gap:.1%} | Passe: {passes}")
    for d, m in metrics_by_dialect.items():
        logger.info(f"  {d}: F2={m['f2']:.3f} | Recall={m['recall']:.3f}")

    if alerts:
        for alert in alerts:
            logger.warning(alert)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Exemple avec données fictives
    df = pd.DataFrame({
        "dialect": ["french"] * 20 + ["moore"] * 15 + ["dioula"] * 10 + ["mix"] * 5,
        "task1_signal": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
                         1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                         0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                         1, 0, 1, 0, 1],
    })

    # Prédictions simulées (légèrement moins bonnes en Moore)
    np.random.seed(42)
    preds = df["task1_signal"].values.copy()
    # Simuler biais sur Moore (quelques erreurs)
    moore_indices = df[df["dialect"] == "moore"].index[:3]
    preds[moore_indices] = 1 - preds[moore_indices]

    report = run_bias_audit(df, preds)
    print(f"\nRésultat audit: {'✅ PASSE' if report.passes_fairness else '❌ ÉCHEC'}")
    print(f"Écart max: {report.max_gap:.1%}")
