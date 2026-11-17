"""
Phase 5 - Risk classification and approval cut-off analysis.

Run from the project root:   python -m scripts.run_phase5

Outputs: band cut-offs added to models/scorecard.joblib,
         reports/tables/risk_*.csv, reports/figures/risk/*.png,
         reports/phase5_risk_classification_report.md
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR, TARGET_COL
from src.plots import BLUE, RISK_COLORS, TEXT_MUTED, save, set_style
from src.risk_bands import assign_band, band_summary, cutoff_table, fit_band_cutoffs
from src.scorecard import score_from_points, score_to_pd
from src.utils import load_clean, md_table, require, save_table, train_test


def plot_bands(test_bands) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [RISK_COLORS[b] for b in test_bands["band"]]
    ax.bar(test_bands["band"], test_bands["default_rate_pct"], color=colors, width=0.6)
    for x, (r, s) in enumerate(zip(test_bands["default_rate_pct"], test_bands["share_pct"])):
        ax.text(x, r + 0.4, f"{r:.1f}% default\n{s:.0f}% of loans", ha="center", fontsize=9,
                color=TEXT_MUTED)
    ax.set_ylabel("Observed default rate (%)")
    ax.set_ylim(0, test_bands["default_rate_pct"].max() * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title("Default rate by risk band (test sample)")
    save(fig, "01_default_rate_by_band", "risk")


def plot_tradeoff(table, ks_row) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(table["approval_rate_pct"], table["approved_bad_rate_pct"], color=BLUE, marker="o", ms=4)
    ax.scatter([ks_row["approval_rate_pct"]], [ks_row["approved_bad_rate_pct"]], s=80,
               color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"KS cut-off: score {ks_row['cutoff_score']:.0f}",
                (ks_row["approval_rate_pct"], ks_row["approved_bad_rate_pct"]),
                xytext=(10, -18), textcoords="offset points", fontsize=9, color=TEXT_MUTED)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Bad rate among approved loans (%)")
    ax.set_title("Approval rate vs bad rate trade-off (test sample)")
    save(fig, "02_approval_vs_bad_rate", "risk")


def main() -> None:
    df = load_clean()
    train, test = train_test(df)
    require([MODELS_DIR / "scorecard.joblib"], "Run Phase 4 first: python -m scripts.run_phase4")
    card = joblib.load(MODELS_DIR / "scorecard.joblib")

    s_train = score_from_points(train, card)
    s_test = score_from_points(test, card)
    cutoffs = fit_band_cutoffs(s_train)

    band_train = band_summary(s_train, train[TARGET_COL], score_to_pd(s_train), cutoffs)
    band_test = band_summary(s_test, test[TARGET_COL], score_to_pd(s_test), cutoffs)
    band_train.insert(0, "sample", "train")
    band_test.insert(0, "sample", "test")

    # Candidate cut-offs: every 5th percentile of training scores
    grid = np.unique(np.round(np.quantile(s_train, np.arange(0, 0.95, 0.05))))
    cut_test = cutoff_table(s_test, test[TARGET_COL], grid)
    cut_train = cutoff_table(s_train, train[TARGET_COL], grid)
    ks_cut = cut_train.loc[cut_train["ks_gap"].idxmax(), "cutoff_score"]  # chosen on train
    ks_row = cut_test.loc[cut_test["cutoff_score"] == ks_cut].iloc[0]

    card["band_cutoffs"] = cutoffs
    card["approval_cutoff"] = float(ks_cut)
    joblib.dump(card, MODELS_DIR / "scorecard.joblib")

    save_table(pd.concat([band_train, band_test], ignore_index=True), "risk_band_summary")
    save_table(cut_test, "risk_cutoff_analysis_test")
    plot_bands(band_test)
    plot_tradeoff(cut_test, ks_row)

    # Band x LendingClub grade on test: does our view agree with LendingClub's?
    test = test.assign(band=assign_band(s_test, cutoffs))
    cross = (test.groupby(["band", "grade"])[TARGET_COL].agg(["size", "mean"])
             .rename(columns={"size": "loans", "mean": "default_rate"}).reset_index())
    cross["default_rate"] = (cross["default_rate"] * 100).round(2)
    save_table(cross, "risk_band_by_lc_grade_test")

    report = [
        "# Phase 5 - Risk classification report\n",
        f"Band cut-offs (from training scores): High < {cutoffs['high_below']:.0f} <= Medium < "
        f"{cutoffs['low_from']:.0f} <= Low.\n",
        "## Risk bands\n", md_table(band_train), "", md_table(band_test),
        f"\n## Approval cut-off analysis (test sample)\nKS-optimal cut-off chosen on the training "
        f"sample: **score {ks_cut:.0f}**.\n",
        md_table(cut_test),
        f"\nAt the KS cut-off on test: approve {ks_row['approval_rate_pct']:.1f}% of applicants, "
        f"approved bad rate {ks_row['approved_bad_rate_pct']:.2f}% vs "
        f"{test[TARGET_COL].mean() * 100:.2f}% if everyone is approved; "
        f"{ks_row['bads_rejected_pct']:.1f}% of defaulters rejected at the cost of "
        f"{ks_row['goods_rejected_pct']:.1f}% of good borrowers.",
    ]
    (REPORTS_DIR / "phase5_risk_classification_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
