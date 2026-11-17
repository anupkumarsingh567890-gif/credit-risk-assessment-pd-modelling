"""
Phase 7 - Model evaluation and champion selection.

Run from the project root:   python -m scripts.run_phase7

Decision threshold: a loan is flagged "predicted default" when its PD is at
least the training default rate (12.5%). With only ~13% defaults, the usual
0.5 threshold would flag almost nobody.

Champion: chosen on CROSS-VALIDATED AUC from Phase 6 (never on the test set),
with the interpretability rule in config.py. The test set is used only to
report final out-of-time performance.

Outputs: models/champion.json, reports/tables/eval_*.csv,
         reports/figures/evaluation/*.png, reports/phase7_model_evaluation_report.md
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src.config import (
    INTERPRETABILITY_MARGIN_AUC, MODEL_FILES, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, TARGET_COL,
    TEST_PREDICTIONS, TABLES_DIR,
)
from src.metrics import all_metrics, confusion, sub_grade_rank
from src.plots import SERIES, TEXT_MUTED, save, set_style
from src.utils import md_table, require, save_json, save_table

MODELS = MODEL_FILES


def choose_champion(cv: pd.DataFrame) -> tuple[str, str]:
    best = cv.loc[cv["cv_auc_mean"].idxmax()]
    lr = cv.loc[cv["model"] == "Logistic Regression"].iloc[0]
    if best["model"] != "Logistic Regression" and best["cv_auc_mean"] - lr["cv_auc_mean"] < INTERPRETABILITY_MARGIN_AUC:
        reason = (f"{best['model']} has the highest CV AUC ({best['cv_auc_mean']:.4f}) but Logistic "
                  f"Regression ({lr['cv_auc_mean']:.4f}) is within {INTERPRETABILITY_MARGIN_AUC} AUC, "
                  "so the interpretable scorecard model is preferred.")
        return "Logistic Regression", reason
    return best["model"], f"Highest cross-validated AUC ({best['cv_auc_mean']:.4f})."


def plot_curves(test: pd.DataFrame, y) -> None:
    set_style()
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (name, col) in enumerate(MODELS.items()):
        fpr, tpr, _ = roc_curve(y, test[f"pd_{col}"])
        ax.plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(y, sub_grade_rank(test["sub_grade"]))
    ax.plot(fpr, tpr, color=TEXT_MUTED, ls=":", lw=1.5, label=f"LC sub-grade (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (out-of-time test)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "01_roc_curves", "evaluation")

    # Precision-Recall
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (name, col) in enumerate(MODELS.items()):
        prec, rec, _ = precision_recall_curve(y, test[f"pd_{col}"])
        ap = average_precision_score(y, test[f"pd_{col}"])
        ax.plot(rec, prec, color=SERIES[i], label=f"{name} (AP {ap:.3f})")
    ax.axhline(y.mean(), color="#c3c2b7", lw=1, ls="--")
    ax.text(0.98, y.mean() + 0.01, f"default rate {y.mean():.1%}", ha="right", fontsize=8,
            color=TEXT_MUTED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (out-of-time test)")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "02_precision_recall_curves", "evaluation")

    # Calibration
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, (name, col) in enumerate(MODELS.items()):
        obs, pred = calibration_curve(y, test[f"pd_{col}"], n_bins=10, strategy="quantile")
        ax.plot(pred * 100, obs * 100, marker="o", ms=4, color=SERIES[i], label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="#c3c2b7", lw=1, ls="--")
    ax.set_xlabel("Average predicted PD (%) per decile")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Calibration curves (out-of-time test)")
    ax.legend(fontsize=8)
    save(fig, "03_calibration_curves", "evaluation")


def plot_confusions(test: pd.DataFrame, y, threshold: float) -> None:
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (name, col) in zip(axes, MODELS.items()):
        cm = confusion(y, test[f"pd_{col}"], threshold)
        ax.imshow(cm.values, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm.values):
            ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=10,
                    color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred. good", "Pred. bad"])
        ax.set_yticks([0, 1], ["Actual good", "Actual bad"])
        ax.grid(False)
        ax.set_title(name, fontsize=10)
    fig.suptitle(f"Confusion matrices (test, threshold PD >= {threshold:.1%})", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "04_confusion_matrices", "evaluation")


def main() -> None:
    require([TEST_PREDICTIONS, TABLES_DIR / "models_cv_summary.csv"],
            "Run Phase 6 first: python -m scripts.run_phase6")
    test = pd.read_parquet(TEST_PREDICTIONS)
    train = pd.read_parquet(PROCESSED_DIR / "train_predictions.parquet")
    cv = pd.read_csv(TABLES_DIR / "models_cv_summary.csv")
    y_test, y_train = test[TARGET_COL], train[TARGET_COL]
    threshold = float(y_train.mean())

    rows = []
    for name, col in MODELS.items():
        rows.append({"model": name, "sample": "train", **all_metrics(y_train, train[f"pd_{col}"], threshold)})
        rows.append({"model": name, "sample": "test", **all_metrics(y_test, test[f"pd_{col}"], threshold)})
    metrics = pd.DataFrame(rows)
    save_table(metrics, "eval_metrics")

    test_only = metrics[metrics["sample"] == "test"].drop(columns="sample")
    cm_rows = []
    for name, col in MODELS.items():
        cm = confusion(y_test, test[f"pd_{col}"], threshold)
        cm_rows.append({"model": name, "TN": cm.iloc[0, 0], "FP": cm.iloc[0, 1],
                        "FN": cm.iloc[1, 0], "TP": cm.iloc[1, 1]})
    save_table(pd.DataFrame(cm_rows), "eval_confusion_matrices")

    plot_curves(test, y_test)
    plot_confusions(test, y_test, threshold)

    champion, reason = choose_champion(cv)
    save_json({"model": champion, "file": f"{MODELS[champion]}.joblib", "threshold": threshold,
               "reason": reason}, MODELS_DIR / "champion.json")

    report = [
        "# Phase 7 - Model evaluation report\n",
        f"Decision threshold: PD >= {threshold:.4f} (training default rate).\n",
        "## Out-of-time test performance\n", md_table(test_only, 4),
        "\n## Train vs test (overfitting check)\n", md_table(metrics, 4),
        "\n## Confusion matrices (test)\n", md_table(pd.DataFrame(cm_rows)),
        "\n## Cross-validated AUC (selection basis)\n",
        md_table(cv[["model", "cv_auc_mean", "cv_auc_std", "calibrated"]], 4),
        f"\n## Champion model: **{champion}**\n{reason}\n",
    ]
    (REPORTS_DIR / "phase7_model_evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
