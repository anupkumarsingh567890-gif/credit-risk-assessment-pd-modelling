"""
Data loading utilities.

Phase 1 rule: we only *read* the data here. No cleaning, no imputation and no
column removal happens in this module; that is Phase 2's job. The one helper
that converts values (parse_lc_date) is used for diagnostics only.
"""
from pathlib import Path

import pandas as pd

from src.config import DATE_FORMAT, RAW_FILE, RAW_PARQUET


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the raw LendingClub CSV exactly as delivered.

    low_memory=False makes pandas infer each column's dtype from the whole
    file instead of chunks, which avoids mixed-type columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place loan.csv in data/raw/ before running."
        )
    return pd.read_csv(path, low_memory=False)


EXPECTED_SHAPE = (39_717, 111)
REQUIRED_COLUMNS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d",
    "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "total_rec_prncp", "last_pymnt_d", "pub_rec_bankruptcies",
]


def validate_raw(df: pd.DataFrame) -> list[str]:
    """Check that the file is the LendingClub 2007-2011 extract this project expects.

    Raises if a required column is missing; returns warnings for other differences.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"loan.csv is missing required columns: {missing}. "
                         "Use the LendingClub 2007-2011 file described in the README.")
    warnings = []
    if df.shape != EXPECTED_SHAPE:
        warnings.append(f"Expected {EXPECTED_SHAPE[0]:,} rows x {EXPECTED_SHAPE[1]} columns, found "
                        f"{df.shape[0]:,} x {df.shape[1]}; results will differ from the documented ones.")
    unexpected = set(df["loan_status"].dropna().unique()) - {"Fully Paid", "Charged Off", "Current"}
    if unexpected:
        warnings.append(f"Unexpected loan_status values: {sorted(unexpected)}")
    return warnings


def save_raw_parquet(df: pd.DataFrame, path: Path = RAW_PARQUET) -> Path:
    """Store an unmodified Parquet copy for faster reloads in later phases."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def parse_lc_date(series: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """Parse LendingClub 'Mon-YY' strings into Timestamps.

    Two-digit years are ambiguous: Python maps '68' to 2068, so a credit line
    opened in Dec-1968 is read as Dec-2068. If `reference` (e.g. issue_d) is
    given, any parsed date later than the reference is shifted back 100 years.
    """
    parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
    if reference is not None:
        future = parsed > reference
        parsed = parsed.where(~future, parsed - pd.DateOffset(years=100))
    return parsed


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Whole calendar months from `earlier` to `later` (NaN if either is missing)."""
    later, earlier = pd.to_datetime(later), pd.to_datetime(earlier)
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)
