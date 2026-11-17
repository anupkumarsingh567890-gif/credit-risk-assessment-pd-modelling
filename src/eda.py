"""
Phase 3 - Exploratory data analysis helpers.

The core tool is `default_rate_table`: group loans by a variable (or by bands
of a numeric variable) and compare the default rate in each group. A variable
that separates high and low default rates is a useful credit-risk driver.
"""
import numpy as np
import pandas as pd

from src.config import TARGET_COL

# Fixed, business-readable bands for numeric variables
BANDS = {
    "annual_inc": ([0, 30_000, 45_000, 60_000, 80_000, 100_000, np.inf],
                   ["<30k", "30-45k", "45-60k", "60-80k", "80-100k", "100k+"]),
    "loan_amnt": ([0, 5_000, 10_000, 15_000, 20_000, 25_000, np.inf],
                  ["<5k", "5-10k", "10-15k", "15-20k", "20-25k", "25k+"]),
    "int_rate": ([0, 8, 10, 12, 14, 16, 18, np.inf],
                 ["<8%", "8-10%", "10-12%", "12-14%", "14-16%", "16-18%", "18%+"]),
    "dti": ([-0.1, 5, 10, 15, 20, 25, np.inf],
            ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]),
    "revol_util": ([-0.1, 20, 40, 60, 80, np.inf],
                   ["0-20%", "20-40%", "40-60%", "60-80%", "80%+"]),
    "credit_history_years": ([0, 5, 10, 15, 20, 30, np.inf],
                             ["<5y", "5-10y", "10-15y", "15-20y", "20-30y", "30y+"]),
    "loan_to_income": ([0, 0.1, 0.2, 0.3, np.inf], ["<10%", "10-20%", "20-30%", "30%+"]),
}


def add_band(df: pd.DataFrame, col: str) -> pd.Series:
    edges, labels = BANDS[col]
    return pd.cut(df[col], bins=edges, labels=labels)


def default_rate_table(df: pd.DataFrame, col: str, target: str = TARGET_COL,
                       sort_by_rate: bool = False, pd_col: str | None = None) -> pd.DataFrame:
    """Loans, defaults, default rate and portfolio share per group.

    If pd_col is given, the average predicted PD per group is added too.
    """
    group = add_band(df, col) if col in BANDS else df[col]
    t = (
        df.groupby(group, observed=True)[target]
        .agg(loans="size", defaults="sum")
        .reset_index()
        .rename(columns={col: "group"})
    )
    t.columns = ["group", "loans", "defaults"]
    if pd.api.types.is_float_dtype(t["group"]) and (t["group"] % 1 == 0).all():
        t["group"] = t["group"].astype(int)  # 36.0 -> 36
    t["default_rate_pct"] = (t["defaults"] / t["loans"] * 100).round(2)
    t["share_pct"] = (t["loans"] / t["loans"].sum() * 100).round(2)
    if pd_col is not None:
        t["avg_pd_pct"] = (df.groupby(group, observed=True)[pd_col].mean().to_numpy() * 100).round(2)
    if sort_by_rate:
        t = t.sort_values("default_rate_pct", ascending=False)
    return t.reset_index(drop=True)


def two_way_rate(df: pd.DataFrame, row: str, col: str, target: str = TARGET_COL) -> pd.DataFrame:
    """Default rate (%) for every combination of two variables (multivariate view)."""
    r = add_band(df, row) if row in BANDS else df[row]
    c = add_band(df, col) if col in BANDS else df[col]
    return (df.groupby([r, c], observed=True)[target].mean().unstack() * 100).round(1)


def vintage_curves(df: pd.DataFrame, max_mob: int = 60) -> pd.DataFrame:
    """Cumulative charge-off rate by months on book, one curve per issue year.

    Default month = estimated onset of the delinquency that led to charge-off
    (`default_onset_month`, see preprocessing.default_onset_month). Uses every
    charge-off, not only those inside the 36-month window, so the full life of
    each cohort is visible.
    """
    rows = []
    for year, g in df.assign(default_mob=df["default_onset_month"]).groupby("issue_year"):
        n = len(g)
        for mob in range(0, max_mob + 1, 3):
            rows.append(
                {"issue_year": year, "months_on_book": mob,
                 "cum_default_rate_pct": (g["default_mob"] <= mob).sum() / n * 100}
            )
    return pd.DataFrame(rows)


def insight_line(table: pd.DataFrame, name: str) -> str:
    """One factual sentence: lowest vs highest default-rate group.

    Groups holding less than 1% of loans are ignored so a handful of loans
    cannot drive the headline.
    """
    table = table[table["share_pct"] >= 1].reset_index(drop=True)
    lo = table.loc[table["default_rate_pct"].idxmin()].to_dict()
    hi = table.loc[table["default_rate_pct"].idxmax()].to_dict()
    lo["group"] = table.at[table["default_rate_pct"].idxmin(), "group"]
    hi["group"] = table.at[table["default_rate_pct"].idxmax(), "group"]
    ratio = hi["default_rate_pct"] / lo["default_rate_pct"] if lo["default_rate_pct"] > 0 else np.nan
    return (
        f"- **{name}:** default rate ranges from {lo['default_rate_pct']:.1f}% "
        f"({lo['group']}) to {hi['default_rate_pct']:.1f}% ({hi['group']}), "
        f"{ratio:.1f}x higher."
    )
