"""
Phase 4 - Scorecard development (WoE, IV, logistic regression, PDO scaling).

Run from the project root:   python -m scripts.run_phase4

Everything is fitted on the TRAINING sample; the test sample is only scored.
Outputs: models/scorecard.joblib, reports/tables/scorecard_*.csv,
         reports/figures/scorecard/*.png, reports/phase4_scorecard_report.md
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import woe
from src.config import (
    BASE_ODDS, BASE_SCORE, CATEGORICAL_FEATURES, FLAG_FEATURES, MODELS_DIR, NUMERIC_FEATURES,
    PDO, REPORTS_DIR, TARGET_COL,
)
from src.metrics import ranking_metrics, sub_grade_rank
from src.plots import BLUE, ORANGE, TEXT_MUTED, save, set_style
from src.scorecard import build_scorecard, score_from_points, score_to_pd
from src.utils import load_clean, md_table, save_table, train_test


def plot_iv(iv_table: pd.DataFrame, final: list[str]) -> None:
    set_style()
    t = iv_table.sort_values("iv")
    colors = [BLUE if f in final else "#c3c2b7" for f in t["feature"]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.5))
    ax.barh(t["feature"], t["iv"], color=colors, height=0.7)
    ax.axvline(0.02, color=TEXT_MUTED, ls="--", lw=1)
    ax.text(0.021, -0.9, "IV = 0.02", fontsize=8, color=TEXT_MUTED)
    ax.set_xlabel("Information Value")
    ax.grid(axis="y", visible=False)
    ax.set_title("Information Value by feature\n(blue = in final scorecard, grey = not selected)",
                 fontsize=11)
    save(fig, "01_information_value", "scorecard")


def plot_woe(card: dict) -> None:
    set_style()
    feats = card["features"]
    cols = 3
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
    for ax, f in zip(axes.ravel(), feats):
        t = card["specs"][f]["table"]
        ax.bar(range(len(t)), t["woe"], color=BLUE, width=0.7)
        ax.axhline(0, color=TEXT_MUTED, lw=0.8)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels(t["bin"], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{f}  (IV {card['specs'][f]['iv']:.3f})", fontsize=9)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("Weight of Evidence by bin (WoE > 0 = safer than average)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "02_woe_by_feature", "scorecard")


def plot_score_distribution(scores: pd.Series, y: pd.Series) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.6, color=BLUE, label="Good (no default)")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.6, color=ORANGE, label="Bad (default)")
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution of goods and bads (test sample)")
    ax.legend()
    save(fig, "03_score_distribution", "scorecard")


def main() -> None:
    df = load_clean()
    train, test = train_test(df)
    y_train, y_test = train[TARGET_COL], test[TARGET_COL]

    card = build_scorecard(train, y_train, NUMERIC_FEATURES,
                           FLAG_FEATURES + CATEGORICAL_FEATURES)
    model = card["model"]

    # ---- Coefficients with p-values and VIF ---------------------------------
    woe_train = woe.transform(train, card["specs"], card["features"])
    X = woe_train.assign(const=1.0)
    vif = {f: variance_inflation_factor(X.values, i) for i, f in enumerate(X.columns) if f != "const"}
    coefs = pd.DataFrame({
        "coefficient": model.params,
        "std_error": model.bse,
        "p_value": model.pvalues,
    }).rename_axis("term").reset_index()
    coefs["vif"] = coefs["term"].map(vif)
    coefs["iv"] = coefs["term"].map({f: card["specs"][f]["iv"] for f in card["features"]})

    # ---- Score both samples -------------------------------------------------
    perf = []
    for name, part, y in [("train", train, y_train), ("test", test, y_test)]:
        score = score_from_points(part, card)
        pd_hat = score_to_pd(score)
        perf.append({"model": "Scorecard (WoE logistic)", "sample": name, **ranking_metrics(y, pd_hat),
                     "mean_pd": pd_hat.mean(), "observed_rate": y.mean()})
        rank = sub_grade_rank(part["sub_grade"])
        perf.append({"model": "Benchmark: LendingClub sub-grade", "sample": name,
                     **ranking_metrics(y, rank)})
        perf.append({"model": "Benchmark: interest rate", "sample": name,
                     **ranking_metrics(y, part["int_rate"])})
        if name == "test":
            test_score = score
    perf = pd.DataFrame(perf)

    # ---- Save ---------------------------------------------------------------
    save_table(card["iv_table"], "scorecard_iv_table")
    save_table(woe.bins_long_table(card["specs"], list(card["specs"])), "scorecard_woe_bins_all")
    save_table(coefs, "scorecard_coefficients")
    save_table(card["points"].drop(columns="key"), "scorecard_points")
    save_table(perf, "scorecard_performance")
    (REPORTS_DIR / "phase4_logit_summary.txt").write_text(str(model.summary()), encoding="utf-8")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    to_save = {k: v for k, v in card.items() if k != "model"}  # statsmodels object not needed
    joblib.dump(to_save, MODELS_DIR / "scorecard.joblib")

    plot_iv(card["iv_table"], card["features"])
    plot_woe(card)
    plot_score_distribution(test_score, y_test)

    selection = pd.DataFrame(
        [{"feature": f, "step": "IV < 0.02 (or > 0.5)", "result": "removed"}
         for f in card["iv_table"]["feature"] if f not in card["iv_selected"]]
        + [{"feature": f, "step": "WoE correlation > 0.6", "result": "removed"} for f in card["corr_dropped"]]
        + [{"feature": e["feature"], "step": e["reason"], "result": "removed"} for e in card["elimination_log"]]
        + [{"feature": f, "step": "final model", "result": "kept"} for f in card["features"]]
    )
    save_table(selection, "scorecard_feature_selection")

    pts = card["points"]
    score_range = pts.groupby("feature")["points"].agg(["min", "max"]).sum()
    report = [
        "# Phase 4 - Scorecard report\n",
        f"Scaling: {BASE_SCORE} points = odds {BASE_ODDS}:1 (good:bad); "
        f"PDO = {PDO}; factor = {card['factor']:.4f}; offset = {card['offset']:.2f}.\n",
        f"Possible score range: {int(score_range['min'])} to {int(score_range['max'])}.\n",
        "## Information Value\n", md_table(card["iv_table"], 4),
        "\n## Feature selection\n", md_table(selection),
        "\n## Final logistic regression (WoE inputs)\n", md_table(coefs, 4),
        "\n## Scorecard points\n", md_table(pts.drop(columns="key"), 3),
        "\n## Performance vs LendingClub benchmarks\n", md_table(perf, 4),
    ]
    (REPORTS_DIR / "phase4_scorecard_report.md").write_text("\n".join(report), encoding="utf-8")

    print(md_table(card["iv_table"], 4))
    print(md_table(selection))
    print(md_table(coefs, 4))
    print(md_table(perf, 4))


if __name__ == "__main__":
    main()
