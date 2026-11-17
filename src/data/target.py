"""
Phase 1 view of the target: the ORIGINAL status-based definition.

    default_spec = 1  -> loan_status == 'Charged Off'
    default_spec = 0  -> loan_status == 'Fully Paid'
    excluded          -> loan_status == 'Current' (outcome not yet observed)

The modelling target used from Phase 2 onwards (default within 36 months) is
built in src/preprocessing.py; this module only describes the raw statuses.
"""
import pandas as pd

from src.config import EXCLUDED_STATUSES, STATUS_COL, TARGET_MAP, TARGET_SPEC_COL


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return the resolved loans only, with a binary `default_spec` column.

    Raises if an unexpected loan_status appears, so that a new status in a
    refreshed file can never be silently dropped or mislabelled.
    """
    known = set(TARGET_MAP) | set(EXCLUDED_STATUSES)
    unexpected = set(df[STATUS_COL].dropna().unique()) - known
    if unexpected:
        raise ValueError(f"Unmapped loan_status values: {sorted(unexpected)}")

    resolved = df[df[STATUS_COL].isin(TARGET_MAP)].copy()
    resolved[TARGET_SPEC_COL] = resolved[STATUS_COL].map(TARGET_MAP).astype("int8")
    return resolved


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares for every loan_status, with its target treatment."""
    counts = df[STATUS_COL].value_counts(dropna=False)
    out = pd.DataFrame({"count": counts, "share_pct": (counts / len(df) * 100).round(2)})
    out["target_value"] = [
        str(TARGET_MAP[s]) if s in TARGET_MAP else "excluded" for s in out.index
    ]
    out.index.name = STATUS_COL
    return out.reset_index()
