"""
Phase 2 - Data preprocessing.

The functions run in this order (see scripts/run_phase2.py):

    1. drop_unusable_columns   empty and constant columns
    2. clean_formats           text -> numbers, dates, tidy categories
    3. add_targets             main 36-month target + original target
    4. engineer_features       ratios, credit-history age, missing flags
    5. split_by_date           chronological train / test split
    6. fit_caps / apply_caps   outlier capping learned on train only

Leakage rule: only columns known when the loan is approved are used as
features. Repayment columns are kept in the clean file for reference and
portfolio reporting, but they are never listed in config.ALL_FEATURES.
"""
import numpy as np
import pandas as pd

from src.config import (
    CAP_COLUMNS,
    LC_PRICING_COLS,
    PERFORMANCE_WINDOW_MONTHS,
    SNAPSHOT_DATE,
    STATUS_COL,
    TARGET_COL,
    TARGET_MAP,
    TARGET_SPEC_COL,
    TEST_START_DATE,
)
from src.data.load import months_between, parse_lc_date

# Columns produced by repayment after the loan is issued (target leakage)
POST_ORIGINATION_COLS = [
    "pymnt_plan", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d",
]
# Identifiers and free-text fields: no modelling value
ID_AND_TEXT_COLS = ["member_id", "url", "desc", "title", "emp_title", "zip_code"]
# Columns that describe the outcome; they must never be used as features
LABEL_COLS = [STATUS_COL, TARGET_COL, TARGET_SPEC_COL, "default_onset_month",
              "months_to_last_payment", "months_observed"]


# --------------------------------------------------------------------------- #
# 1. Remove columns with no information
# --------------------------------------------------------------------------- #
def drop_unusable_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    empty = [c for c in df.columns if df[c].isna().all()]
    constant = [c for c in df.columns if c not in empty and df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=empty + constant)
    return df, {"empty": empty, "constant": constant}


# --------------------------------------------------------------------------- #
# 2. Clean formats
# --------------------------------------------------------------------------- #
def _percent_to_float(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.astype("string").str.replace("%", "").str.strip(), errors="coerce")
    return out.astype(float)  # plain float64 (NaN), not pandas' nullable Float64 (pd.NA)


def _emp_length_to_years(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10, missing stays NaN."""
    s = s.astype("string").str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10"})
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce").astype(float)


def clean_formats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["term_months"] = pd.to_numeric(df["term"].astype("string").str.extract(r"(\d+)")[0]).astype(int)
    df["int_rate"] = _percent_to_float(df["int_rate"])
    df["revol_util"] = _percent_to_float(df["revol_util"])
    df["emp_length_years"] = _emp_length_to_years(df["emp_length"])

    # Dates. earliest_cr_line uses issue_d to fix the two-digit-year problem.
    df["issue_d"] = parse_lc_date(df["issue_d"])
    df["earliest_cr_line"] = parse_lc_date(df["earliest_cr_line"], reference=df["issue_d"])
    df["last_pymnt_d"] = parse_lc_date(df["last_pymnt_d"])

    # Rare categories merged into a sensible neighbour
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER"})
    df["purpose"] = df["purpose"].replace({"renewable_energy": "other"})

    df["issue_year"] = df["issue_d"].dt.year
    df["issue_quarter"] = df["issue_d"].dt.to_period("Q").astype(str)
    return df


# --------------------------------------------------------------------------- #
# 3. Targets
# --------------------------------------------------------------------------- #
def instalments_covered(principal_repaid: pd.Series, funded: pd.Series,
                        instalment: pd.Series, int_rate_pct: pd.Series) -> pd.Series:
    """Number of scheduled instalments whose principal has been repaid.

    For an annuity loan with principal P, monthly rate r and instalment A, the
    balance left after k on-time payments is
        B_k = A/r - (A/r - P) * (1 + r)^k
    Solving for k with B_k = P - principal_repaid gives how far along the
    repayment schedule the borrower really got, whatever the payment dates were.
    """
    r = int_rate_pct / 1200
    annuity_ratio = instalment / r
    balance = (funded - principal_repaid).clip(lower=0)
    valid = (annuity_ratio - funded) > 0  # always true for a correctly priced annuity
    k = np.log((annuity_ratio - balance) / (annuity_ratio - funded).where(valid)) / np.log1p(r)
    return k


def default_onset_month(df: pd.DataFrame) -> pd.Series:
    """Estimated month on book of the first missed instalment (charged-off loans).

    LendingClub gives no payment history, so the onset of the delinquency that
    led to charge-off is taken as the EARLIEST of three upper bounds:
      a) the first instalment not covered by principal repaid (arrears; borrowers
         in arrears often keep making partial or late payments, so the last
         payment date alone dates default about 2 months too late),
      b) the month after the last payment received (borrower stopped paying),
      c) the contractual term (any balance left at maturity is a missed payment).
    Loans that never paid get onset = 1.
    """
    covered = instalments_covered(df["total_rec_prncp"], df["funded_amnt"], df["installment"], df["int_rate"])
    first_unpaid = np.floor(covered + 1e-6) + 1
    after_last_payment = months_between(df["last_pymnt_d"], df["issue_d"]).fillna(0) + 1
    onset = np.fmin(np.fmin(first_unpaid, after_last_payment), df["term_months"])
    return pd.Series(onset, index=df.index)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add both target definitions.

    default (main)  1 = charged off AND the delinquency that led to charge-off
                    started within the first 36 months on book
                    (default_onset_month <= PERFORMANCE_WINDOW_MONTHS).
                    0 = everything else: Fully Paid, Current (all repaying for
                    53+ months) and charge-offs that started after month 36.
                    The same rule applies to 36- and 60-month loans.
    default_spec    1 = Charged Off, 0 = Fully Paid, NaN = Current
                    (the original definition, kept for sensitivity analysis).

    Repayment fields are used ONLY to build the label and are dropped before
    modelling (see finalize_columns).
    """
    df = df.copy()
    charged_off = df[STATUS_COL].eq("Charged Off")

    # Outcome maturity: every loan must be observed for the whole window,
    # otherwise recent loans would look artificially good.
    months_observed = months_between(pd.Series(SNAPSHOT_DATE, index=df.index), df["issue_d"])
    too_young = int((months_observed < PERFORMANCE_WINDOW_MONTHS).sum())
    if too_young:
        raise ValueError(
            f"{too_young} loans have fewer than {PERFORMANCE_WINDOW_MONTHS} months between issue "
            f"date and the snapshot ({SNAPSHOT_DATE:%Y-%m}); their outcome is not fully observed."
        )
    df["months_observed"] = months_observed

    df["months_to_last_payment"] = months_between(df["last_pymnt_d"], df["issue_d"])
    df["default_onset_month"] = default_onset_month(df).where(charged_off)

    df[TARGET_COL] = (charged_off & (df["default_onset_month"] <= PERFORMANCE_WINDOW_MONTHS)).astype("int8")
    df[TARGET_SPEC_COL] = df[STATUS_COL].map(TARGET_MAP)  # Current -> NaN
    return df


# --------------------------------------------------------------------------- #
# 4. Feature engineering (origination-time information only)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Length of credit history at the time of the loan
    df["credit_history_years"] = months_between(df["issue_d"], df["earliest_cr_line"]) / 12

    # Affordability ratios
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"]

    # Missing-value flags: Phase 1 showed missingness itself predicts default
    df["emp_length_missing"] = df["emp_length_years"].isna().astype("int8")
    df["emp_title_missing"] = df["emp_title"].isna().astype("int8")

    # A blank 'months since last delinquency' means no delinquency on file
    df["has_delinq_history"] = df["mths_since_last_delinq"].notna().astype("int8")
    df["has_public_record"] = (
        df["mths_since_last_record"].notna() | (df["pub_rec"] > 0)
    ).astype("int8")
    return df


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leakage, identifiers and free text; keep id + dates for tracking."""
    drop = [c for c in POST_ORIGINATION_COLS + ID_AND_TEXT_COLS if c in df.columns]
    drop += ["term", "emp_length", "mths_since_last_record", "funded_amnt", "funded_amnt_inv"]
    return df.drop(columns=[c for c in drop if c in df.columns])


def check_no_leakage(features: list[str]) -> None:
    """Fail loudly if a feature list contains outcome, repayment or pricing fields."""
    forbidden = set(POST_ORIGINATION_COLS) | set(LABEL_COLS) | set(LC_PRICING_COLS) | {"issue_d", "id"}
    leaked = sorted(set(features) & forbidden)
    if leaked:
        raise ValueError(f"Leakage: these columns must not be model features: {leaked}")


# --------------------------------------------------------------------------- #
# 5. Chronological split
# --------------------------------------------------------------------------- #
def split_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Older loans train the model, newer loans test it (out-of-time)."""
    df = df.copy()
    df["sample"] = np.where(df["issue_d"] < TEST_START_DATE, "train", "test")
    return df.sort_values(["issue_d", "id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Outlier capping (winsorising) - learned on train only
# --------------------------------------------------------------------------- #
def fit_caps(train: pd.DataFrame) -> dict:
    caps = {}
    for col, (low_q, high_q) in CAP_COLUMNS.items():
        caps[col] = (float(train[col].quantile(low_q)), float(train[col].quantile(high_q)))
    return caps


def apply_caps(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (low, high) in caps.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, dropped = drop_unusable_columns(raw)
    df = clean_formats(df)
    df = add_targets(df)
    df = engineer_features(df)
    df = finalize_columns(df)
    df = split_by_date(df)

    caps = fit_caps(df[df["sample"] == "train"])
    df = apply_caps(df, caps)

    log = {"dropped": dropped, "caps": caps}
    return df, log
