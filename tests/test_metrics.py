"""Discrimination, accuracy and calibration metrics."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import all_metrics, confusion, gini, ks_statistic, ranking_metrics, sub_grade_rank


def test_gini_is_two_auc_minus_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    assert gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1)


def test_perfect_and_uninformative_models():
    y = np.array([0, 0, 1, 1])
    assert ranking_metrics(y, [0.1, 0.2, 0.8, 0.9]) == {"roc_auc": 1.0, "gini": 1.0, "ks": 1.0}
    flat = ranking_metrics(y, [0.5, 0.5, 0.5, 0.5])
    assert flat["roc_auc"] == 0.5 and flat["ks"] == 0.0


def test_ks_hand_example():
    # cumulative bad% - good% peaks at 2/3 when the threshold is 0.5
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.8])
    assert ks_statistic(y, p) == pytest.approx(2 / 3)


def test_threshold_metrics_and_confusion():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.2, 0.7, 0.3])
    m = all_metrics(y, p, threshold=0.5)
    # flagged: 0.6 (good) and 0.7 (bad) -> TP 1, FP 1, FN 1, TN 2
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    cm = confusion(y, p, 0.5)
    assert cm.loc["Actual bad", "Predicted bad"] == 1
    assert cm.loc["Actual good", "Predicted bad"] == 1
    assert cm.to_numpy().sum() == 5


def test_sub_grade_rank():
    assert sub_grade_rank(pd.Series(["A1", "A5", "B1", "G5"])).tolist() == [1, 5, 6, 35]
