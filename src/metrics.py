"""
Model performance metrics used in credit risk.

ROC-AUC  probability that a random defaulter gets a higher PD than a random
         non-defaulter (0.5 = random, 1 = perfect).
Gini     = 2 * AUC - 1, the industry's usual way of quoting discrimination.
KS       max over thresholds of (cumulative % of bads - cumulative % of goods),
         i.e. max(TPR - FPR); the point where the two groups are most separated.
Brier    mean((PD - outcome)^2): accuracy of the probabilities themselves
         (lower is better); sensitive to calibration, not only ranking.
Accuracy / Precision / Recall / F1 need a threshold that turns a PD into a
"predicted default" flag; see threshold choice in the scripts.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def gini(y, score) -> float:
    return 2 * roc_auc_score(y, score) - 1


def ks_statistic(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def ranking_metrics(y, pd_pred) -> dict:
    return {
        "roc_auc": roc_auc_score(y, pd_pred),
        "gini": gini(y, pd_pred),
        "ks": ks_statistic(y, pd_pred),
    }


def all_metrics(y, pd_pred, threshold: float) -> dict:
    flag = (np.asarray(pd_pred) >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, flag),
        "precision": precision_score(y, flag, zero_division=0),
        "recall": recall_score(y, flag),
        "f1": f1_score(y, flag),
        **ranking_metrics(y, pd_pred),
        "brier": brier_score_loss(y, pd_pred),
        "mean_pd": float(np.mean(pd_pred)),
        "observed_default_rate": float(np.mean(y)),
    }
    return out


def confusion(y, pd_pred, threshold: float) -> pd.DataFrame:
    flag = (np.asarray(pd_pred) >= threshold).astype(int)
    cm = confusion_matrix(y, flag, labels=[0, 1])
    return pd.DataFrame(cm, index=["Actual good", "Actual bad"],
                        columns=["Predicted good", "Predicted bad"])


def sub_grade_rank(sub_grade: pd.Series) -> pd.Series:
    """A1 -> 1 ... G5 -> 35 (higher = riskier): LendingClub's grade as a benchmark score."""
    letter = sub_grade.str[0].map({g: i for i, g in enumerate("ABCDEFG")})
    return letter * 5 + sub_grade.str[1].astype(int)
