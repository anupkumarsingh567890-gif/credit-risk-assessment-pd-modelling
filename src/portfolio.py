"""
Phase 9 helpers - portfolio views shared by the pipeline and the dashboard.
"""
import pandas as pd

from src.config import TARGET_COL

BANDS = ["Low", "Medium", "High"]


def band_summary(pf: pd.DataFrame) -> pd.DataFrame:
    """Loans, exposure, predicted PD and observed default rate per risk band."""
    t = (
        pf.groupby("risk_category")
        .agg(loans=("id", "size"), exposure_musd=("loan_amnt", lambda s: s.sum() / 1e6),
             defaults=(TARGET_COL, "sum"), avg_pd_pct=("probability_of_default", "mean"),
             default_rate_pct=(TARGET_COL, "mean"), avg_score=("credit_score", "mean"))
        .reindex(BANDS)
        .dropna(how="all")
        .reset_index()
    )
    t["avg_pd_pct"] *= 100
    t["default_rate_pct"] *= 100
    t["loan_share_pct"] = t["loans"] / t["loans"].sum() * 100
    t["exposure_share_pct"] = t["exposure_musd"] / t["exposure_musd"].sum() * 100
    return t.round(2)


def concentration(pf: pd.DataFrame, col: str, top: int | None = 10) -> pd.DataFrame:
    """Share of the portfolio vs share of High-risk loans for each segment.

    concentration_ratio = high-risk share / portfolio share; above 1 means the
    segment holds more than its fair share of High-risk loans.
    """
    high = pf["risk_category"] == "High"
    size = pf.groupby(col).size()
    t = pd.DataFrame({
        "loans": size,
        "portfolio_share_pct": size / len(pf) * 100,
        "high_risk_share_pct": pf[high].groupby(col).size() / max(int(high.sum()), 1) * 100,
        "high_risk_rate_pct": pf.groupby(col)["risk_category"].apply(lambda s: (s == "High").mean() * 100),
        "exposure_musd": pf.groupby(col)["loan_amnt"].sum() / 1e6,
    }).fillna(0)
    t["concentration_ratio"] = t["high_risk_share_pct"] / t["portfolio_share_pct"]
    t = t.sort_values("high_risk_share_pct", ascending=False).round(2).reset_index()
    return t.head(top) if top else t
