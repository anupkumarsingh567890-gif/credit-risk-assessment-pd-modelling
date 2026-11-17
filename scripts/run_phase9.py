"""
Phase 9 - Portfolio risk analysis.

Run from the project root:   python -m scripts.run_phase9

Scores the whole loan book (all 39,717 loans) with the scorecard and the
champion PD model, then looks at the portfolio the way a risk manager would:
overall default rate, risk distribution (by count and by exposure), default
rates by key segments, predicted vs actual defaults, and where the high-risk
loans are concentrated.

Note: training-period loans were used to fit the models, so their PDs are
in-sample; the portfolio view is descriptive, not a validation.

Outputs: data/processed/portfolio_scored.parquet, reports/tables/portfolio_*.csv,
         reports/figures/portfolio/*.png, reports/phase9_portfolio_report.md
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.assessment import assess_features, load_artifacts
from src.config import ALL_FEATURES, REPORTS_DIR, SCORED_PORTFOLIO, TARGET_COL, TARGET_SPEC_COL
from src.eda import default_rate_table
from src.portfolio import band_summary, concentration
from src.plots import BLUE, ORANGE, RISK_COLORS, TEXT_MUTED, save, set_style
from src.utils import load_clean, md_table, save_table


def plot_distribution(band_t: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = [RISK_COLORS[b] for b in band_t["risk_category"]]
    for ax, col, title in [(axes[0], "loan_share_pct", "Share of loans (%)"),
                           (axes[1], "exposure_share_pct", "Share of exposure - loan amount (%)")]:
        ax.bar(band_t["risk_category"], band_t[col], color=colors, width=0.6)
        for x, v in enumerate(band_t[col]):
            ax.text(x, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9, color=TEXT_MUTED)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, band_t[col].max() * 1.2)
    fig.suptitle("Portfolio risk distribution", x=0.01, ha="left", fontweight="bold")
    save(fig, "01_risk_distribution", "portfolio")


def plot_expected_vs_actual(band_t: pd.DataFrame) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(band_t))
    ax.bar(x - 0.18, band_t["avg_pd_pct"], width=0.34, color=BLUE, label="Average predicted PD")
    ax.bar(x + 0.18, band_t["default_rate_pct"], width=0.34, color=ORANGE, label="Observed default rate")
    ax.set_xticks(x, band_t["risk_category"])
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Predicted vs observed default rate by risk band (all loans)")
    save(fig, "02_predicted_vs_observed_by_band", "portfolio")


def main() -> None:
    df = load_clean()
    art = load_artifacts()
    scored = assess_features(df[ALL_FEATURES], art)
    pf = df.join(scored)
    pf.to_parquet(SCORED_PORTFOLIO, index=False)

    overall = {
        "loans": len(pf),
        "exposure_musd": pf["loan_amnt"].sum() / 1e6,
        "default_rate_36m_pct": pf[TARGET_COL].mean() * 100,
        "charged_off_rate_resolved_pct": pf[TARGET_SPEC_COL].mean() * 100,
        "average_pd_pct": pf["probability_of_default"].mean() * 100,
        "expected_defaults": pf["probability_of_default"].sum(),
        "actual_defaults_36m": int(pf[TARGET_COL].sum()),
        "high_risk_loans_pct": (pf["risk_category"] == "High").mean() * 100,
    }
    overall_t = pd.DataFrame({
        "metric": list(overall),
        "value": [f"{v:,.2f}" if isinstance(v, (float, np.floating)) else f"{v:,}" for v in overall.values()],
    })

    band_t = band_summary(pf)

    segments = {}
    for col in ["grade", "annual_inc", "purpose", "int_rate", "term_months"]:
        segments[col] = default_rate_table(pf, col, pd_col="probability_of_default")
        save_table(segments[col], f"portfolio_default_by_{col}")

    conc_purpose = concentration(pf, "purpose")
    conc_state = concentration(pf, "addr_state")
    save_table(overall_t, "portfolio_overview")
    save_table(band_t, "portfolio_by_risk_band")
    save_table(conc_purpose, "portfolio_concentration_purpose")
    save_table(conc_state, "portfolio_concentration_state")

    plot_distribution(band_t)
    plot_expected_vs_actual(band_t)

    report = [
        "# Phase 9 - Portfolio risk analysis (all 39,717 loans)\n",
        "## Overview\n", md_table(overall_t),
        "\n## Risk distribution\n", md_table(band_t),
        *[f"\n## Default rate by {c}\n{md_table(t)}" for c, t in segments.items()],
        "\n## High-risk concentration by purpose (top 10)\n", md_table(conc_purpose),
        "\n## High-risk concentration by state (top 10)\n", md_table(conc_state),
        "\nconcentration_ratio > 1: the segment holds more than its fair share of High-risk loans.",
    ]
    (REPORTS_DIR / "phase9_portfolio_report.md").write_text("\n".join(report), encoding="utf-8")
    print(md_table(overall_t))
    print(md_table(band_t))
    print(md_table(conc_purpose))


if __name__ == "__main__":
    main()
