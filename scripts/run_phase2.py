"""
Phase 2 - Data preprocessing.

Run from the project root:   python -m scripts.run_phase2

Outputs:
    data/processed/loans_clean.parquet (+ .csv copy)
    reports/tables/phase2_*.csv
    reports/phase2_preprocessing_report.md
"""
import pandas as pd

from src.config import (
    ALL_FEATURES, CLEAN_CSV, CLEAN_PARQUET, MODELS_DIR, PROCESSED_DIR, RAW_PARQUET, REPORTS_DIR,
    STATUS_COL, TARGET_COL, TARGET_SPEC_COL,
)
from src.data.load import load_raw
from src.preprocessing import preprocess
from src.utils import md_table, save_json, save_table


def main() -> None:
    raw = pd.read_parquet(RAW_PARQUET) if RAW_PARQUET.exists() else load_raw()
    df, log = preprocess(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    df.to_csv(CLEAN_CSV, index=False)
    # caps are needed again when scoring a new borrower (Phase 8 / app)
    save_json({"caps": log["caps"]}, MODELS_DIR / "preprocessing.json")

    # ---- Summary tables ----------------------------------------------------
    target_compare = (
        df.groupby(STATUS_COL)
        .agg(loans=("id", "size"), main_target_bad=(TARGET_COL, "sum"))
        .reset_index()
    )

    split = (
        df.groupby("sample")
        .agg(
            loans=("id", "size"),
            first_issue=("issue_d", "min"),
            last_issue=("issue_d", "max"),
            default_rate_pct=(TARGET_COL, lambda s: round(s.mean() * 100, 2)),
            spec_default_rate_pct=(TARGET_SPEC_COL, lambda s: round(s.mean() * 100, 2)),
        )
        .reindex(["train", "test"])
        .reset_index()
    )
    split["first_issue"] = split["first_issue"].dt.strftime("%Y-%m")
    split["last_issue"] = split["last_issue"].dt.strftime("%Y-%m")

    cohort = (
        df.groupby(["issue_year", "term_months"])
        .agg(
            loans=("id", "size"),
            main_default_rate_pct=(TARGET_COL, lambda s: round(s.mean() * 100, 2)),
            spec_default_rate_pct=(TARGET_SPEC_COL, lambda s: round(s.mean() * 100, 2)),
        )
        .reset_index()
    )

    missing = df[ALL_FEATURES].isna().sum()
    missing = missing[missing > 0].rename("n_missing").rename_axis("feature").reset_index()
    missing["treatment"] = "Median in ML models / own WoE bin in scorecard"

    caps = pd.DataFrame(
        [(c, lo, hi) for c, (lo, hi) in log["caps"].items()],
        columns=["feature", "lower_cap", "upper_cap"],
    )

    save_table(split, "phase2_split_summary")
    save_table(cohort, "phase2_target_by_cohort")
    save_table(caps, "phase2_outlier_caps")

    report = [
        "# Phase 2 - Preprocessing report\n",
        f"Clean dataset: **{len(df):,} loans x {df.shape[1]} columns**; "
        f"{len(ALL_FEATURES)} model features.\n",
        f"Dropped {len(log['dropped']['empty'])} empty and {len(log['dropped']['constant'])} "
        f"constant columns, plus repayment (leakage), identifier and free-text columns.\n",
        "## Target: default within 36 months\n", md_table(target_compare),
        "\n## Default rate by cohort under both definitions\n", md_table(cohort),
        "\n## Chronological split\n", md_table(split),
        "\n## Remaining missing values in model features\n", md_table(missing),
        "\n## Outlier caps (learned on training sample)\n", md_table(caps),
    ]
    (REPORTS_DIR / "phase2_preprocessing_report.md").write_text("\n".join(report), encoding="utf-8")

    print(md_table(split))
    print(md_table(target_compare))


if __name__ == "__main__":
    main()
