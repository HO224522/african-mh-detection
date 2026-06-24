"""
evaluate_model.py
=================
Évaluation complète: F2, biais, latence, taille modèle.

Usage:
    python scripts/evaluate_model.py \
        --model data/models/best_model.pt \
        --corpus data/corpus/annotated_texts.csv
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from src.inference import DistressDetector
from src.utils.bias_audit import run_bias_audit, fbeta_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_latency(detector, texts, n_runs=50):
    """Mesure la latence moyenne d'inférence."""
    times = []
    for text in texts[:n_runs]:
        start = time.time()
        detector.predict(text)
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def main(args):
    detector = DistressDetector(model_path=args.model)
    df = pd.read_csv(args.corpus)

    logger.info(f"Évaluation sur {len(df)} exemples...")

    # Prédictions
    y_true, y_pred, y_conf = [], [], []
    for _, row in df.iterrows():
        result = detector.predict(str(row["text"]))
        y_true.append(int(row["task1_signal"]))
        y_pred.append(1 if result.signal else 0)
        y_conf.append(result.confidence)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Métriques Task 1
    f2 = fbeta_score(y_true, y_pred, beta=2)
    print("\n" + "="*50)
    print("  ÉVALUATION COMPLÈTE")
    print("="*50)
    print(f"\nTask 1 — Détection Signal:")
    print(f"  F2-Score:    {f2:.4f}  (cible: ≥0.85)")
    print(classification_report(y_true, y_pred, target_names=["Neutre", "Détresse"]))

    # Audit biais
    print("\nAudit Biais par Dialecte:")
    if "dialect" in df.columns:
        report = run_bias_audit(df, y_pred)
        print(f"  Écart max:   {report.max_gap:.1%}  (cible: <5%)")
        print(f"  Fairness:    {'✅ OK' if report.passes_fairness else '❌ ÉCHEC'}")
        for dialect, m in report.metrics_by_dialect.items():
            print(f"  {dialect:10s}: F2={m['f2']:.3f} | Recall={m['recall']:.3f}")

    # Latence
    texts = df["text"].tolist()
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    mean_lat, std_lat = evaluate_latency(detector, texts)
    print(f"\nLatence ({device_type}): {mean_lat*1000:.0f}ms ± {std_lat*1000:.0f}ms")
    print(f"  (cible: <2000ms CPU, <500ms GPU)")

    # Taille modèle
    if args.model:
        import os
        size_mb = os.path.getsize(args.model) / 1e6
        print(f"\nTaille modèle: {size_mb:.0f}MB (cible: <100MB quantisé)")

    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    main(args)
