"""
Phase 3 - Exploratory data analysis.

Run from the project root:   python -m scripts.run_phase3

EDA uses the TRAINING sample only, so nothing learned here (bands, feature
choices) is influenced by the test period. The vintage analysis is the one
exception: it is a time view of every cohort by design.

Outputs: reports/figures/eda/*.png, reports/tables/eda_*.csv,
         reports/phase3_eda_insights.md
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import NUMERIC_FEATURES, REPORTS_DIR, TARGET_COL
from src.eda import default_rate_table, insight_line, two_way_rate, vintage_curves
from src.plots import BLUE, DIVERGING, SEQUENTIAL, SERIES, TEXT_MUTED, bar_rate, save, set_style
from src.utils import load_clean, save_table, train_test

# variable -> (chart title, horizontal bars?, sort by rate?)
DRIVERS = {
    "grade": ("Default rate by LendingClub grade", False, False),
    "sub_grade": ("Default rate by LendingClub sub-grade", False, False),
    "int_rate": ("Default rate by interest rate", False, False),
    "annual_inc": ("Default rate by annual income", False, False),
    "loan_amnt": ("Default rate by loan amount", False, False),
    "dti": ("Default rate by debt-to-income ratio", False, False),
    "emp_length_years": ("Default rate by employment length (years)", False, False),
    "home_ownership": ("Default rate by home ownership", False, False),
    "purpose": ("Default rate by loan purpose", True, True),
    "credit_history_years": ("Default rate by length of credit history", False, False),
    "revol_util": ("Default rate by revolving utilisation", False, False),
    "inq_last_6mths": ("Default rate by credit inquiries (last 6 months)", False, False),
    "term_months": ("Default rate by loan term (months)", False, False),
    "verification_status": ("Default rate by income verification", False, False),
    "loan_to_income": ("Default rate by loan-to-income ratio", False, False),
}


def univariate(train) -> None:
    set_style()
    cols = ["annual_inc", "loan_amnt", "dti", "revol_util", "credit_history_years", "int_rate"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(train[c].dropna(), bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
        ax.set_title(c, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distributions of key numeric variables (training sample)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "01_distributions", "eda")


def target_view(df) -> None:
    set_style()
    t = df.groupby("sample")[TARGET_COL].agg(["size", "mean"]).reindex(["train", "test"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(t.index, t["mean"] * 100, color=BLUE, width=0.5)
    for x, (n, r) in enumerate(zip(t["size"], t["mean"])):
        ax.text(x, r * 100 + 0.3, f"{r * 100:.1f}%\n({n:,} loans)", ha="center", fontsize=9,
                color=TEXT_MUTED)
    ax.set_ylabel("Default rate (%)")
    ax.set_ylim(0, t["mean"].max() * 130)
    ax.grid(axis="x", visible=False)
    ax.set_title("36-month default rate by sample")
    save(fig, "00_target_by_sample", "eda")


def correlation(train) -> None:
    set_style()
    cols = NUMERIC_FEATURES + ["int_rate", TARGET_COL]
    corr = train[cols].corr()
    save_table(corr.round(3), "eda_correlation_matrix", index=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation of numeric features (training sample)")
    save(fig, "20_correlation_heatmap", "eda")


def multivariate(train) -> None:
    set_style()
    for row, col, name in [("grade", "term_months", "21_grade_x_term"),
                           ("annual_inc", "dti", "22_income_x_dti")]:
        table = two_way_rate(train, row, col)
        save_table(table, f"eda_{name}", index=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(table, cmap=SEQUENTIAL, annot=True, fmt=".1f", linewidths=2,
                    linecolor="white", ax=ax, cbar_kws={"label": "Default rate (%)"})
        ax.set_title(f"Default rate (%) by {row} and {col}")
        save(fig, name, "eda")


def vintage(df) -> None:
    set_style()
    curves = vintage_curves(df)
    save_table(curves, "eda_vintage_curves")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (year, g) in enumerate(curves.groupby("issue_year")):
        ax.plot(g["months_on_book"], g["cum_default_rate_pct"], color=SERIES[i], label=str(year))
    ax.axvline(36, color=TEXT_MUTED, ls="--", lw=1)
    ax.text(36.5, 1, "36-month window", fontsize=8, color=TEXT_MUTED)
    ax.set_xlabel("Months on book")
    ax.set_ylabel("Cumulative charge-off rate (%)")
    ax.set_title("Vintage analysis: cumulative charge-offs by issue year (all loans)")
    ax.legend(title="Issue year", loc="upper left")
    save(fig, "23_vintage_curves", "eda")

    q = df.groupby(["issue_quarter", "term_months"])[TARGET_COL].mean().unstack() * 100
    save_table(q.round(2), "eda_default_by_issue_quarter", index=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, term in enumerate(q.columns):
        ax.plot(q.index, q[term], marker="o", ms=4, color=SERIES[i], label=f"{term} months")
    ax.set_ylabel("36-month default rate (%)")
    ax.set_title("Default rate by issue quarter and term (all loans)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "24_default_by_issue_quarter", "eda")


def main() -> None:
    df = load_clean()
    train, _ = train_test(df)
    overall = train[TARGET_COL].mean() * 100

    target_view(df)
    univariate(train)

    insights = []
    for i, (col, (title, horizontal, sort_rate)) in enumerate(DRIVERS.items(), start=2):
        table = default_rate_table(train, col, sort_by_rate=sort_rate)
        save_table(table, f"eda_default_rate_{col}")
        fig, _ = bar_rate(table, "group", title, overall=overall, horizontal=horizontal)
        save(fig, f"{i:02d}_default_by_{col}", "eda")
        insights.append(insight_line(table, col))

    correlation(train)
    multivariate(train)
    vintage(df)

    corr = train[NUMERIC_FEATURES + ["int_rate", TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    top_corr = corr.abs().sort_values(ascending=False).head(6)
    pairs = train[NUMERIC_FEATURES].corr().abs().where(
        np.triu(np.ones((len(NUMERIC_FEATURES),) * 2), k=1).astype(bool)).stack()
    high_pairs = pairs[pairs > 0.6].sort_values(ascending=False)

    lines = [
        "# Phase 3 - EDA insights (training sample, 2007-06 to 2011-06)\n",
        f"Training default rate (36-month window): **{overall:.2f}%** on {len(train):,} loans.\n",
        "## Default-rate drivers (lowest vs highest group)\n",
        *insights,
        "\n## Strongest linear relationships with default\n",
        *[f"- {c}: r = {corr[c]:+.3f}" for c in top_corr.index],
        "\n## Highly correlated feature pairs (|r| > 0.6) - redundancy to manage\n",
        *([f"- {a} & {b}: {v:.2f}" for (a, b), v in high_pairs.items()] or ["- none"]),
        "\nCharts: `reports/figures/eda/`. Tables: `reports/tables/eda_*.csv`.",
    ]
    (REPORTS_DIR / "phase3_eda_insights.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
