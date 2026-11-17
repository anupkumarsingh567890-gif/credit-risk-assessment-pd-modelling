"""Small helpers shared by every phase: loading the clean data, saving tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLEAN_PARQUET, TABLES_DIR


def require(paths, hint: str) -> None:
    """Stop with a clear message if an input produced by an earlier phase is missing."""
    missing = [str(Path(p)) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}. {hint}")


def load_clean() -> pd.DataFrame:
    require([CLEAN_PARQUET], "Run Phase 2 first: python -m scripts.run_phase2")
    return pd.read_parquet(CLEAN_PARQUET)


def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["sample"] == "train"].copy(), df[df["sample"] == "test"].copy()


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Save a result table as CSV under reports/tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%Y-%m-%d")
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def md_table(df: pd.DataFrame, decimals: int = 2) -> str:
    """Render a DataFrame as a markdown table (no extra libraries needed)."""

    def fmt(v, col):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:,.{decimals}f}"
        if isinstance(v, (int, np.integer)):
            return str(v) if ("year" in col or col == "id") else f"{v:,}"
        return str(v)

    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * df.shape[1]) + "|"
    rows = [
        "| " + " | ".join(fmt(v, c) for v, c in zip(r, cols)) + " |"
        for r in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])
