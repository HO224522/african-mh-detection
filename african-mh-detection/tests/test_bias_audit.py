"""
test_bias_audit.py
==================
Tests pour le système d'audit de biais.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.bias_audit import run_bias_audit, fbeta_score


class TestFbetaScore:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1])
        assert fbeta_score(y, y, beta=2) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        assert fbeta_score(y_true, y_pred, beta=2) == pytest.approx(0.0)

    def test_beta_2_weights_recall(self):
        # Recall-heavy prediction (few FN, many FP)
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1])  # All positive
        f2 = fbeta_score(y_true, y_pred, beta=2)
        assert f2 > 0.5  # F2 favors recall


class TestBiasAudit:
    @pytest.fixture
    def balanced_df(self):
        """DataFrame équilibré entre dialectes."""
        data = []
        for dialect in ["french", "moore", "dioula"]:
            for i in range(20):
                data.append({"dialect": dialect, "task1_signal": i % 2})
        return pd.DataFrame(data)

    def test_passes_with_equal_performance(self, balanced_df):
        """Doit passer si toutes les prédictions sont parfaites."""
        predictions = balanced_df["task1_signal"].values.copy()
        report = run_bias_audit(balanced_df, predictions)
        assert report.passes_fairness
        assert report.max_gap == pytest.approx(0.0)

    def test_fails_with_large_gap(self, balanced_df):
        """Doit échouer si un dialecte est très mal prédit."""
        predictions = balanced_df["task1_signal"].values.copy()
        # Inverser toutes les prédictions pour Moore
        moore_mask = balanced_df["dialect"] == "moore"
        predictions[moore_mask.values] = 1 - predictions[moore_mask.values]

        report = run_bias_audit(balanced_df, predictions)
        assert not report.passes_fairness
        assert len(report.alerts) > 0

    def test_report_has_required_fields(self, balanced_df):
        """Le rapport doit avoir tous les champs requis."""
        predictions = balanced_df["task1_signal"].values
        report = run_bias_audit(balanced_df, predictions)

        assert hasattr(report, "date")
        assert hasattr(report, "passes_fairness")
        assert hasattr(report, "max_gap")
        assert hasattr(report, "metrics_by_dialect")
        assert hasattr(report, "alerts")
