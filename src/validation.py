"""
Phase 10 - Model validation helpers.

Population Stability Index (PSI) / Characteristic Stability Index (CSI)
    PSI = sum over buckets of (actual% - expected%) * ln(actual% / expected%)
    expected = development (train) distribution, actual = new (test) distribution
    < 0.10 stable | 0.10-0.25 moderate shift, monitor | > 0.25 significant shift

Hosmer-Lemeshow test (calibration)
    Split loans into g = 10 PD deciles; compare observed (O) vs expected (E) defaults:
    HL = sum (O - E)^2 / (E * (1 - E / n))
    Degrees of freedom: g - 2 on the development sample (two parameters were
    estimated on it), g on an independent validation sample (nothing was
    estimated on it). A p-value below 0.05 means the PDs do not match the
    observed default rates.

Calibration-in-the-large (binomial test)
    z = (observed defaults - sum of PDs) / sqrt(sum PD * (1 - PD))
    Tests whether the overall PD level matches the observed default rate.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.config import PSI_SHIFT, PSI_STABLE
from sklearn.metrics import roc_auc_score


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI with buckets set on the expected (train) distribution's deciles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.Series(pd.cut(expected, edges)).value_counts(normalize=True, sort=False).sort_index()
    a = pd.Series(pd.cut(actual, edges)).value_counts(normalize=True, sort=False).sort_index()
    t = pd.DataFrame({"bucket": e.index.astype(str), "expected_pct": e.values * 100,
                      "actual_pct": a.values * 100})
    ep, ap = np.clip(e.values, 1e-6, None), np.clip(a.values, 1e-6, None)
    t["psi_contribution"] = (ap - ep) * np.log(ap / ep)
    return float(t["psi_contribution"].sum()), t


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    ep = np.clip(e.reindex(idx).fillna(0).values, 1e-6, None)
    ap = np.clip(a.reindex(idx).fillna(0).values, 1e-6, None)
    return float(((ap - ep) * np.log(ap / ep)).sum())


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "Stable"
    if value < PSI_SHIFT:
        return "Monitor"
    return "Significant shift"


def hosmer_lemeshow(y, pd_pred, n_groups: int = 10,
                    out_of_sample: bool = True) -> tuple[float, float, pd.DataFrame]:
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_pred)})
    df["decile"] = pd.qcut(df["pd"], n_groups, labels=False, duplicates="drop") + 1
    t = df.groupby("decile").agg(loans=("y", "size"), observed=("y", "sum"), expected=("pd", "sum"),
                                 avg_pd=("pd", "mean"), observed_rate=("y", "mean"))
    hl = (((t["observed"] - t["expected"]) ** 2) / (t["expected"] * (1 - t["expected"] / t["loans"]))).sum()
    dof = len(t) if out_of_sample else len(t) - 2
    p_value = 1 - chi2.cdf(hl, dof)
    t["avg_pd_pct"] = t["avg_pd"] * 100
    t["observed_rate_pct"] = t["observed_rate"] * 100
    return float(hl), float(p_value), t.drop(columns=["avg_pd", "observed_rate"]).reset_index()


def auc_by_period(df: pd.DataFrame, period_col: str, y_col: str, pd_cols: list[str]) -> pd.DataFrame:
    rows = []
    for period, g in df.groupby(period_col):
        if g[y_col].nunique() < 2:
            continue
        row = {"period": period, "loans": len(g), "default_rate_pct": g[y_col].mean() * 100}
        for c in pd_cols:
            row[f"auc_{c}"] = roc_auc_score(g[y_col], g[c])
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sensitivity(y, pd_pred, thresholds) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(pd_pred)
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = (flag & (y == 1)).sum()
        precision = tp / flag.sum() if flag.sum() else np.nan
        recall = tp / (y == 1).sum()
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
        rows.append({"threshold": t, "flagged_pct": flag.mean() * 100, "precision": precision,
                     "recall": recall, "f1": f1, "accuracy": (flag == (y == 1)).mean()})
    return pd.DataFrame(rows)


def calibration_in_the_large(y, pd_pred) -> dict:
    """Two-sided binomial (normal approximation) test of the overall PD level."""
    y, p = np.asarray(y, float), np.asarray(pd_pred, float)
    expected, observed = p.sum(), y.sum()
    z = (observed - expected) / np.sqrt((p * (1 - p)).sum())
    return {"mean_pd": p.mean(), "observed_rate": y.mean(), "expected_defaults": expected,
            "observed_defaults": observed, "z": z, "p_value": 2 * (1 - norm.cdf(abs(z)))}
