"""
Phase 5 - Risk classification and approval cut-off analysis.

Risk bands are defined on the TRAINING score distribution:
    High   = worst 20% of scores
    Medium = next 40%
    Low    = best 40%
The cut-off scores are then applied unchanged to any new borrower.

Cut-off analysis: if we approve everyone with score >= c,
    approval rate          = approved / all applicants
    approved bad rate      = defaults among approved / approved
    bads rejected (%)      = share of all defaulters we turn away
    goods rejected (%)     = share of all good borrowers we turn away (lost business)
The KS cut-off (max of bads rejected - goods rejected) is the statistically
best separating point; the business can pick a stricter or looser cut-off.
"""
import numpy as np
import pandas as pd

from src.config import RISK_BAND_SHARES


def fit_band_cutoffs(train_scores: pd.Series) -> dict:
    """Score limits between bands, from training percentiles."""
    high_share = RISK_BAND_SHARES["High"]
    medium_share = RISK_BAND_SHARES["Medium"]
    return {
        "high_below": float(np.quantile(train_scores, high_share)),
        "low_from": float(np.quantile(train_scores, high_share + medium_share)),
    }


def assign_band(scores, cutoffs: dict) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < cutoffs["high_below"], "High",
                    np.where(scores < cutoffs["low_from"], "Medium", "Low"))


def band_summary(scores, y, pd_pred, cutoffs: dict) -> pd.DataFrame:
    df = pd.DataFrame({"band": assign_band(scores, cutoffs), "score": scores, "y": y, "pd": pd_pred})
    t = df.groupby("band").agg(
        loans=("y", "size"), defaults=("y", "sum"), avg_score=("score", "mean"),
        min_score=("score", "min"), max_score=("score", "max"), avg_pd_pct=("pd", "mean"),
    )
    t = t.reindex(["Low", "Medium", "High"])
    t["share_pct"] = t["loans"] / t["loans"].sum() * 100
    t["default_rate_pct"] = t["defaults"] / t["loans"] * 100
    t["avg_pd_pct"] *= 100
    return t.round(2).reset_index()


def cutoff_table(scores, y, cutoffs) -> pd.DataFrame:
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    total_bad, total_good = y.sum(), (1 - y).sum()
    rows = []
    for c in cutoffs:
        approved = scores >= c
        n_app = approved.sum()
        rows.append(
            {
                "cutoff_score": c,
                "approval_rate_pct": n_app / len(y) * 100,
                "approved_bad_rate_pct": y[approved].mean() * 100 if n_app else np.nan,
                "rejected_bad_rate_pct": y[~approved].mean() * 100 if (~approved).any() else np.nan,
                "bads_rejected_pct": y[~approved].sum() / total_bad * 100,
                "goods_rejected_pct": (1 - y[~approved]).sum() / total_good * 100,
            }
        )
    t = pd.DataFrame(rows)
    t["ks_gap"] = t["bads_rejected_pct"] - t["goods_rejected_pct"]
    return t.round(2)
