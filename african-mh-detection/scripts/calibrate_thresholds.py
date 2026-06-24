"""
calibrate_thresholds.py
=======================
Calibration Platt Scaling + optimisation seuils F2.

Usage:
    python scripts/calibrate_thresholds.py \
        --model data/models/best_model.pt \
        --corpus data/corpus/annotated_texts.csv \
        --output data/models/optimal_thresholds.json
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src.inference import DistressDetector
from src.utils.bias_audit import fbeta_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true, y_scores, beta=2, steps=100):
    """Trouve le seuil maximisant le F-beta score."""
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.linspace(0.1, 0.9, steps):
        y_pred = (y_scores >= threshold).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def main(args):
    detector = DistressDetector(model_path=args.model)
    df = pd.read_csv(args.corpus)

    # Obtenir les scores de confiance
    y_true, y_scores = [], []
    for _, row in df.iterrows():
        result = detector.predict(str(row["text"]))
        y_true.append(int(row["task1_signal"]))
        y_scores.append(result.confidence)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Optimiser seuil
    opt_threshold, opt_f2 = find_optimal_threshold(y_true, y_scores, beta=2)
    logger.info(f"Seuil optimal: {opt_threshold:.3f} → F2={opt_f2:.4f}")

    thresholds = {
        "signal": round(float(opt_threshold), 4),
        "category": 0.40,
        "context": 0.35,
        "optimized_f2": round(float(opt_f2), 4),
    }

    with open(args.output, "w") as f:
        json.dump(thresholds, f, indent=2)

    logger.info(f"Seuils sauvegardés: {args.output}")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="data/models/optimal_thresholds.json")
    args = parser.parse_args()
    main(args)
