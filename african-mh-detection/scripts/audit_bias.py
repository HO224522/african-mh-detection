"""
audit_bias.py
=============
Script d'audit mensuel des biais par dialecte.

Usage:
    python scripts/audit_bias.py \
        --corpus data/corpus/annotated_texts.csv \
        --model data/models/best_model.pt \
        --output reports/bias_audit_$(date +%Y%m).json
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.inference import DistressDetector
from src.utils.bias_audit import run_bias_audit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    logger.info(f"Audit biais — {datetime.now().strftime('%Y-%m-%d')}")

    # Charger données
    df = pd.read_csv(args.corpus)
    logger.info(f"Corpus: {len(df)} exemples")

    # Distribution par dialecte
    logger.info("Distribution par dialecte:")
    for dialect, count in df["dialect"].value_counts().items():
        logger.info(f"  {dialect}: {count} exemples ({count/len(df):.1%})")

    # Prédictions
    detector = DistressDetector(model_path=args.model)
    predictions = []
    for text in df["text"].tolist():
        result = detector.predict(str(text))
        predictions.append(1 if result.signal else 0)

    predictions = np.array(predictions)

    # Audit
    report = run_bias_audit(df, predictions)

    # Rapport JSON
    report_dict = {
        "date": report.date,
        "passes_fairness": report.passes_fairness,
        "max_gap": report.max_gap,
        "threshold": 0.05,
        "metrics_by_dialect": report.metrics_by_dialect,
        "alerts": report.alerts,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Rapport sauvegardé: {args.output}")

    if not report.passes_fairness:
        logger.error("❌ AUDIT ÉCHOUÉ — Biais détecté!")
        for alert in report.alerts:
            logger.error(alert)
        exit(1)
    else:
        logger.info("✅ AUDIT RÉUSSI — Modèle équitable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", default="data/models/best_model.pt")
    parser.add_argument("--output", default=f"reports/bias_audit_{datetime.now().strftime('%Y%m')}.json")
    args = parser.parse_args()
    main(args)
