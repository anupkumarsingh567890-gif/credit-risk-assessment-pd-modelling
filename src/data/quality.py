"""
Data quality diagnostics for Phase 1.

Every function here only *measures* the data and returns a DataFrame or dict.
Nothing is modified, so the same checks can be re-run after Phase 2 cleaning to
prove each issue was resolved.
"""
import numpy as np
import pandas as pd

from src.config import SNAPSHOT_DATE, STATUS_COL, TARGET_MAP
from src.data.load import months_between, parse_lc_date

TEXT_PERCENT_COLS = ["int_rate", "revol_util"]

# Columns produced by repayment after issuance (see dictionary.py)
LEAKAGE_EVIDENCE_COLS = [
    "total_rec_prncp",
    "total_pymnt",
    "last_pymnt_amnt",
    "recoveries",
    "collection_recovery_fee",
    "total_rec_late_fee",
    "out_prncp",
]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def dataset_overview(df: pd.DataFrame) -> dict:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric_columns": len(numeric),
        "non_numeric_columns": df.shape[1] - len(numeric),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
    }


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count / % per column with a severity band."""
    n_missing = df.isna().sum()
    pct = (n_missing / len(df) * 100).round(2)
    bands = pd.cut(
        pct,
        bins=[-0.01, 0, 5, 50, 99.99, 100],
        labels=["0%", "0-5%", "5-50%", "50-<100%", "100%"],
    )
    out = pd.DataFrame({"n_missing": n_missing, "missing_pct": pct, "band": bands})
    return out.sort_values("missing_pct", ascending=False).rename_axis("column").reset_index()


def duplicate_report(df: pd.DataFrame) -> dict:
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_id": int(df["id"].duplicated().sum()),
        "duplicate_member_id": int(df["member_id"].duplicated().sum()),
        # same borrower+amount+date would indicate a re-listed loan
        "duplicate_business_key": int(
            df.duplicated(subset=["member_id", "loan_amnt", "issue_d"]).sum()
        ),
    }


def constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Populated columns with a single distinct value (zero information)."""
    rows = []
    for c in df.columns:
        s = df[c]
        if s.notna().any() and s.nunique(dropna=True) <= 1:
            rows.append(
                {
                    "column": c,
                    "only_value": s.dropna().iloc[0],
                    "missing_pct": round(s.isna().mean() * 100, 2),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Format and value checks
# --------------------------------------------------------------------------- #
def format_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Fields whose storage format will need conversion in Phase 2."""
    rows = []
    for c in TEXT_PERCENT_COLS:
        rows.append(
            {
                "column": c,
                "issue": "Numeric value stored as text with '%'",
                "example": df[c].dropna().iloc[0],
            }
        )
    rows.append(
        {
            "column": "term",
            "issue": "Leading whitespace; numeric months stored as text",
            "example": repr(df["term"].iloc[0]),
        }
    )
    rows.append(
        {
            "column": "emp_length",
            "issue": "Ordinal years stored as text ('< 1 year', '10+ years')",
            "example": df["emp_length"].dropna().iloc[0],
        }
    )
    rows.append(
        {
            "column": "issue_d / earliest_cr_line / *_d",
            "issue": "Dates stored as 'Mon-YY' text with two-digit years",
            "example": df["earliest_cr_line"].iloc[0],
        }
    )
    return pd.DataFrame(rows)


def date_checks(df: pd.DataFrame) -> dict:
    issue = parse_lc_date(df["issue_d"])
    ecl_raw = parse_lc_date(df["earliest_cr_line"])
    ecl_fixed = parse_lc_date(df["earliest_cr_line"], reference=issue)
    last_pay = parse_lc_date(df["last_pymnt_d"])
    pull = parse_lc_date(df["last_credit_pull_d"])
    return {
        "issue_d_min": issue.min().strftime("%Y-%m"),
        "issue_d_max": issue.max().strftime("%Y-%m"),
        "earliest_cr_line_parsed_max": ecl_raw.max().strftime("%Y-%m"),
        "earliest_cr_line_after_issue_d": int((ecl_raw > issue).sum()),
        "earliest_cr_line_min_after_fix": ecl_fixed.min().strftime("%Y-%m"),
        "last_pymnt_d_max": last_pay.max().strftime("%Y-%m"),
        "last_credit_pull_d_max": pull.max().strftime("%Y-%m"),
        "inferred_snapshot": max(last_pay.max(), pull.max()).strftime("%Y-%m"),
    }


def numeric_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Distribution summary plus an IQR-rule outlier count (flag only)."""
    rows = []
    for c in cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s.astype("string").str.rstrip("%").str.strip(), errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        lower = q1 - 1.5 * (q3 - q1)
        rows.append(
            {
                "column": c,
                "min": s.min(),
                "p01": s.quantile(0.01),
                "median": s.median(),
                "mean": s.mean(),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "iqr_outliers": int(((s > upper) | (s < lower)).sum()),
                "skew": round(s.skew(), 2),
            }
        )
    return pd.DataFrame(rows).round(2)


def category_profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Level counts for low-cardinality categoricals, incl. rare levels."""
    rows = []
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        rare = vc[vc / len(df) < 0.005]
        rows.append(
            {
                "column": c,
                "n_levels": int(df[c].nunique(dropna=True)),
                "missing": int(df[c].isna().sum()),
                "top_level": vc.index[0],
                "top_share_pct": round(vc.iloc[0] / len(df) * 100, 1),
                "rare_levels_<0.5%": ", ".join(f"{k} ({v})" for k, v in rare.items()) or "-",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Target, maturity and leakage
# --------------------------------------------------------------------------- #
def maturity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Status mix by issue year and term.

    Shows which cohorts still contain unresolved ('Current') loans and how
    much excluding them changes the observed default rate.
    """
    issue = parse_lc_date(df["issue_d"])
    tmp = pd.DataFrame(
        {
            "issue_year": issue.dt.year,
            "term": df["term"].astype("string").str.strip(),
            "status": df[STATUS_COL],
        }
    )
    tab = pd.crosstab([tmp["issue_year"], tmp["term"]], tmp["status"])
    for s in ["Charged Off", "Fully Paid", "Current"]:
        if s not in tab:
            tab[s] = 0
    tab["total"] = tab[["Charged Off", "Fully Paid", "Current"]].sum(axis=1)
    tab["current_pct"] = (tab["Current"] / tab["total"] * 100).round(1)
    resolved = tab["Charged Off"] + tab["Fully Paid"]
    tab["default_rate_excl_current_pct"] = (tab["Charged Off"] / resolved * 100).round(2)
    tab["default_rate_current_as_good_pct"] = (tab["Charged Off"] / tab["total"] * 100).round(2)
    return tab.reset_index()


def current_loans_age(df: pd.DataFrame) -> dict:
    cur = df[df[STATUS_COL] == "Current"]
    issue = parse_lc_date(cur["issue_d"])
    mob = (SNAPSHOT_DATE.year - issue.dt.year) * 12 + (SNAPSHOT_DATE.month - issue.dt.month)
    return {
        "n_current": len(cur),
        "terms": cur["term"].astype("string").str.strip().value_counts().to_dict(),
        "issue_from": issue.min().strftime("%Y-%m"),
        "issue_to": issue.max().strftime("%Y-%m"),
        "months_on_book_min": int(mob.min()),
        "months_on_book_max": int(mob.max()),
    }


def default_timing(df: pd.DataFrame) -> pd.DataFrame:
    """For charged-off loans: months from issue to last payment (default proxy)."""
    co = df[df[STATUS_COL] == "Charged Off"]
    mob = months_between(parse_lc_date(co["last_pymnt_d"]), parse_lc_date(co["issue_d"]))
    term = co["term"].astype("string").str.strip()
    rows = []
    for t in sorted(term.unique()):
        m = mob[term == t]
        rows.append(
            {
                "term": t,
                "charged_off": len(m),
                "no_payment_ever": int(m.isna().sum()),
                "median_mob": m.median(),
                "p75_mob": m.quantile(0.75),
                "p90_mob": m.quantile(0.90),
                "p99_mob": m.quantile(0.99),
                "share_defaulting_after_mob_53_pct": round((m >= 53).mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def leakage_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of repayment fields by outcome: large gaps = the field encodes the label."""
    resolved = df[df[STATUS_COL].isin(TARGET_MAP)]
    return (
        resolved.groupby(STATUS_COL)[LEAKAGE_EVIDENCE_COLS]
        .mean()
        .round(1)
        .T.rename_axis("column")
        .reset_index()
    )


def missingness_vs_default(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Default rate when a field is missing vs present (is missingness informative?)."""
    resolved = df[df[STATUS_COL].isin(TARGET_MAP)]
    y = resolved[STATUS_COL].map(TARGET_MAP)
    rows = []
    for c in cols:
        miss = resolved[c].isna()
        rows.append(
            {
                "column": c,
                "n_missing": int(miss.sum()),
                "default_rate_missing_pct": round(y[miss].mean() * 100, 2) if miss.any() else np.nan,
                "default_rate_present_pct": round(y[~miss].mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)
