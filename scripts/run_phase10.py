"""
Phase 10 - Model validation.

Run from the project root:   python -m scripts.run_phase10

Checks (all on the champion unless stated):
  1. Out-of-time validation   train vs CV vs test AUC for every model
  2. Cross-validation         fold-by-fold stability (from Phase 6)
  3. Overfitting              train-test AUC gap
  4. Population Stability     PSI of score and PD, CSI of scorecard features
  5. Feature importance       coefficients, tree importances, permutation importance
  6. Calibration              decile table + Hosmer-Lemeshow, calibration-in-the-large
  7. Threshold sensitivity    precision / recall / F1 across PD thresholds
  8. Stability over time      AUC by issue quarter
  9. Target-definition check  AUC against the original Charged-Off target

Outputs: reports/tables/validation_*.csv, reports/figures/validation/*.png,
         reports/phase10_validation_report.md
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src import woe
from src.config import (
    ALL_FEATURES, CALIBRATION_ALPHA, MAX_CV_AUC_STD, MAX_OVERFIT_GAP_AUC, MIN_TEST_AUC, MODEL_FILES,
    MODELS_DIR, PROCESSED_DIR, PSI_STABLE, RANDOM_SEED, REPORTS_DIR, TABLES_DIR, TARGET_COL,
    TARGET_SPEC_COL, TEST_PREDICTIONS,
)
from src.plots import BLUE, SERIES, TEXT_MUTED, save, set_style
from src.scorecard import score_from_points, score_to_pd
from src.utils import load_clean, md_table, require, save_table, train_test
from src.validation import (
    auc_by_period, calibration_in_the_large, hosmer_lemeshow, psi, psi_categorical, psi_status,
    threshold_sensitivity,
)

FILES = MODEL_FILES


def importance_tables(train, test, card) -> pd.DataFrame:
    rows = []
    # Logistic regression: |coefficient| x std of WoE feature = contribution to the logit
    woe_train = woe.transform(train, card["specs"], card["features"])
    for f, b in card["coefficients"].items():
        rows.append({"model": "Logistic Regression (scorecard)", "feature": f,
                     "importance": abs(b) * woe_train[f].std()})
    # Tree models: built-in importances, mapped back from one-hot columns
    for name in ["Random Forest", "XGBoost"]:
        pipe = joblib.load(MODELS_DIR / f"{FILES[name]}.joblib")["pipeline"]
        cols = pipe.named_steps["prep"].get_feature_names_out()
        imp = pipe.named_steps["model"].feature_importances_
        s = pd.Series(imp, index=[c.split("__", 1)[1] for c in cols])
        for f in ALL_FEATURES:
            val = s[[c for c in s.index if c == f or c.startswith(f + "_")]].sum()
            rows.append({"model": name, "feature": f, "importance": val})
    t = pd.DataFrame(rows)
    t["importance_pct"] = t.groupby("model")["importance"].transform(lambda s: s / s.sum() * 100)
    return t.sort_values(["model", "importance_pct"], ascending=[True, False])


def main() -> None:
    require([TEST_PREDICTIONS, MODELS_DIR / "champion.json", TABLES_DIR / "models_cv_summary.csv"],
            "Run Phases 6 and 7 first: python run_all.py --from 6")
    df = load_clean()
    train, test = train_test(df)
    preds = pd.read_parquet(TEST_PREDICTIONS)
    train_preds = pd.read_parquet(PROCESSED_DIR / "train_predictions.parquet")
    champion = json.loads((MODELS_DIR / "champion.json").read_text())
    champ_col = f"pd_{FILES[champion['model']]}"
    card = joblib.load(MODELS_DIR / "scorecard.joblib")
    cv = pd.read_csv(TABLES_DIR / "models_cv_summary.csv")
    folds = pd.read_csv(TABLES_DIR / "models_cv_folds.csv")
    set_style()

    # 1-3. Out-of-time, CV, overfitting ---------------------------------------
    oot = []
    for name, f in FILES.items():
        tr = roc_auc_score(train_preds[TARGET_COL], train_preds[f"pd_{f}"])
        te = roc_auc_score(preds[TARGET_COL], preds[f"pd_{f}"])
        cvm = cv.loc[cv.model == name, "cv_auc_mean"].iloc[0]
        oot.append({"model": name, "train_auc": tr, "cv_auc": cvm, "test_auc": te,
                    "train_test_gap": tr - te,
                    "overfitting": "Yes" if tr - te > MAX_OVERFIT_GAP_AUC else "No"})
    oot = pd.DataFrame(oot)
    save_table(oot, "validation_oot_overfitting")
    fold_view = folds.pivot(index="fold", columns="model", values="valid_auc").reset_index()

    # 4. PSI / CSI --------------------------------------------------------------
    s_train = score_from_points(train, card)
    s_test = score_from_points(test, card)
    psi_score, psi_score_t = psi(s_train, s_test)
    psi_pd, _ = psi(train_preds[champ_col], preds[champ_col])
    csi = []
    for f in card["features"]:
        spec = card["specs"][f]
        k_tr = woe.keys_for(train[f], spec)
        k_te = woe.keys_for(test[f], spec)
        val = psi_categorical(k_tr, k_te)
        csi.append({"feature": f, "csi": val, "status": psi_status(val)})
    csi = pd.DataFrame(csi).sort_values("csi", ascending=False)
    stability = pd.DataFrame([
        {"measure": "Credit score PSI", "value": psi_score, "status": psi_status(psi_score)},
        {"measure": f"PD PSI ({champion['model']})", "value": psi_pd, "status": psi_status(psi_pd)},
    ])
    save_table(stability, "validation_psi")
    save_table(psi_score_t, "validation_psi_score_buckets")
    save_table(csi, "validation_csi_features")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(psi_score_t))
    ax.bar(x - 0.18, psi_score_t["expected_pct"], width=0.34, color=BLUE, label="Train (expected)")
    ax.bar(x + 0.18, psi_score_t["actual_pct"], width=0.34, color=SERIES[1], label="Test (actual)")
    ax.set_xticks(x, [f"D{i + 1}" for i in x])
    ax.set_xlabel("Training score decile")
    ax.set_ylabel("% of loans")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title(f"Score stability: PSI = {psi_score:.3f} ({psi_status(psi_score)})")
    save(fig, "01_score_psi", "validation")

    # 5. Feature importance ------------------------------------------------------
    imp = importance_tables(train, test, card)
    save_table(imp, "validation_feature_importance")
    bundle = joblib.load(MODELS_DIR / champion["file"])
    perm = permutation_importance(bundle["pipeline"], test[ALL_FEATURES], test[TARGET_COL],
                                  scoring="roc_auc", n_repeats=5, random_state=RANDOM_SEED)
    perm_t = pd.DataFrame({"feature": ALL_FEATURES, "auc_drop_mean": perm.importances_mean,
                           "auc_drop_std": perm.importances_std})
    perm_t = perm_t[perm_t["auc_drop_mean"].abs() > 0].sort_values("auc_drop_mean", ascending=False)
    save_table(perm_t, "validation_permutation_importance")

    xgb = imp[imp.model == "XGBoost"].head(12).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].barh(perm_t["feature"].iloc[::-1], perm_t["auc_drop_mean"].iloc[::-1], color=BLUE)
    axes[0].set_xlabel("Drop in test AUC when the feature is shuffled")
    axes[0].set_title(f"Permutation importance - {champion['model']} (champion)", fontsize=10)
    axes[1].barh(xgb["feature"], xgb["importance_pct"], color=BLUE)
    axes[1].set_xlabel("Share of total importance (%)")
    axes[1].set_title("XGBoost - top 12 features", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    save(fig, "02_feature_importance", "validation")

    # 6. Calibration -------------------------------------------------------------
    # test sample is independent of model fitting -> HL with g degrees of freedom
    hl, hl_p, deciles = hosmer_lemeshow(preds[TARGET_COL], preds[champ_col], out_of_sample=True)
    save_table(deciles, "validation_calibration_deciles")
    citl = calibration_in_the_large(preds[TARGET_COL], preds[champ_col])
    save_table(pd.DataFrame([citl]), "validation_calibration_in_the_large")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(deciles))
    ax.bar(x - 0.18, deciles["avg_pd_pct"], width=0.34, color=BLUE, label="Predicted PD")
    ax.bar(x + 0.18, deciles["observed_rate_pct"], width=0.34, color=SERIES[1], label="Observed default rate")
    ax.set_xticks(x, [f"D{d}" for d in deciles["decile"]])
    ax.set_xlabel("PD decile (D1 = safest)")
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title(f"Calibration by decile - {champion['model']} (test)")
    save(fig, "03_calibration_deciles", "validation")

    # 7. Threshold sensitivity --------------------------------------------------
    thr = threshold_sensitivity(preds[TARGET_COL], preds[champ_col], np.round(np.arange(0.05, 0.41, 0.025), 3))
    save_table(thr, "validation_threshold_sensitivity")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(["precision", "recall", "f1"]):
        ax.plot(thr["threshold"], thr[m], color=SERIES[i], label=m.capitalize())
    ax.axvline(champion["threshold"], color=TEXT_MUTED, ls="--", lw=1)
    ax.text(champion["threshold"] + 0.005, 0.95, "chosen threshold", fontsize=8, color=TEXT_MUTED)
    ax.set_xlabel("PD threshold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Threshold sensitivity - {champion['model']} (test)")
    save(fig, "04_threshold_sensitivity", "validation")

    # 8. Stability over time ----------------------------------------------------
    all_preds = pd.concat([
        train_preds.merge(train[["id", "issue_quarter"]], on="id").assign(sample="train"),
        preds.assign(sample="test"),
    ])
    challenger = "XGBoost" if champion["model"] != "XGBoost" else "Logistic Regression"
    chall_col = f"pd_{FILES[challenger]}"
    by_q = auc_by_period(all_preds, "issue_quarter", TARGET_COL, [champ_col, chall_col])
    by_q = by_q[by_q["loans"] >= 300]
    save_table(by_q, "validation_auc_by_quarter")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_q["period"], by_q[f"auc_{champ_col}"], marker="o", ms=4, color=SERIES[0],
            label=f"{champion['model']} (champion)")
    ax.plot(by_q["period"], by_q[f"auc_{chall_col}"], marker="o", ms=4, color=SERIES[1], label=challenger)
    first_test = by_q.index[by_q["period"] >= "2011Q3"]
    if len(first_test):
        ax.axvspan(by_q.loc[first_test[0], "period"], by_q["period"].iloc[-1], color="#f0efec")
        ax.text(by_q.loc[first_test[0], "period"], 0.52, " out-of-time test", fontsize=8, color=TEXT_MUTED)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 0.8)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Discrimination by issue quarter (train quarters are in-sample; 300+ loans per quarter)")
    save(fig, "05_auc_by_quarter", "validation")

    # 9. Target-definition sensitivity -------------------------------------------
    resolved = preds[preds[TARGET_SPEC_COL].notna()]
    target_check = pd.DataFrame([
        {"target": "Default within 36 months (main)", "loans": len(preds),
         "default_rate_pct": preds[TARGET_COL].mean() * 100,
         "test_auc": roc_auc_score(preds[TARGET_COL], preds[champ_col])},
        {"target": "Charged Off vs Fully Paid, Current excluded", "loans": len(resolved),
         "default_rate_pct": resolved[TARGET_SPEC_COL].mean() * 100,
         "test_auc": roc_auc_score(resolved[TARGET_SPEC_COL], resolved[champ_col])},
    ])
    save_table(target_check, "validation_target_definition")

    # Consistency of the two PD views shown to users (scorecard vs champion) ---
    scorecard_pd = score_to_pd(s_test)
    rank_agreement = pd.Series(scorecard_pd).corr(pd.Series(preds[champ_col].to_numpy()), method="spearman")

    # Summary ---------------------------------------------------------------------
    def flag(ok: bool, bad: str = "Monitor") -> str:
        return "Pass" if ok else bad

    champ_oot = oot[oot.model == champion["model"]].iloc[0]
    cv_std = cv.loc[cv.model == champion["model"], "cv_auc_std"].iloc[0]
    summary = pd.DataFrame([
        {"check": "Out-of-time discrimination",
         "result": f"test AUC {champ_oot.test_auc:.3f} (Gini {2 * champ_oot.test_auc - 1:.3f})",
         "rule": f"AUC >= {MIN_TEST_AUC}", "status": flag(champ_oot.test_auc >= MIN_TEST_AUC, "Review")},
        {"check": "Overfitting (train - test AUC)", "result": f"{champ_oot.train_test_gap:+.3f}",
         "rule": f"gap <= {MAX_OVERFIT_GAP_AUC}",
         "status": flag(champ_oot.train_test_gap <= MAX_OVERFIT_GAP_AUC, "Review")},
        {"check": "CV stability (AUC std across folds)", "result": f"{cv_std:.3f}",
         "rule": f"std <= {MAX_CV_AUC_STD}", "status": flag(cv_std <= MAX_CV_AUC_STD)},
        {"check": "Score PSI (train -> test)", "result": f"{psi_score:.3f}",
         "rule": f"PSI < {PSI_STABLE}", "status": flag(psi_score < PSI_STABLE)},
        {"check": "Max characteristic CSI", "result": f"{csi['csi'].max():.3f} ({csi.iloc[0]['feature']})",
         "rule": f"CSI < {PSI_STABLE}", "status": flag(csi["csi"].max() < PSI_STABLE)},
        {"check": "Calibration-in-the-large (binomial test)",
         "result": f"mean PD {citl['mean_pd']:.2%} vs observed {citl['observed_rate']:.2%}; "
                   f"z = {citl['z']:+.2f}, p = {citl['p_value']:.3f}",
         "rule": f"p >= {CALIBRATION_ALPHA}", "status": flag(citl["p_value"] >= CALIBRATION_ALPHA)},
        {"check": "Hosmer-Lemeshow (test, 10 groups, 10 d.f.)", "result": f"HL = {hl:.1f}, p = {hl_p:.3f}",
         "rule": f"p >= {CALIBRATION_ALPHA}", "status": flag(hl_p >= CALIBRATION_ALPHA)},
        {"check": "Scorecard PD vs champion PD (rank correlation)", "result": f"{rank_agreement:.3f}",
         "rule": "Spearman >= 0.9", "status": flag(rank_agreement >= 0.9)},
    ])
    save_table(summary, "validation_summary")

    report = [
        "# Phase 10 - Model validation report\n",
        f"Champion: **{champion['model']}**.\n",
        "## Validation summary\n", md_table(summary),
        "\n## Out-of-time and overfitting\n", md_table(oot, 4),
        "\n## Cross-validation AUC by fold (time-series folds)\n", md_table(fold_view, 4),
        "\n## Population stability\n", md_table(stability, 4),
        "\n### Characteristic stability (scorecard features)\n", md_table(csi, 4),
        "\n## Permutation importance (champion, test)\n", md_table(perm_t, 4),
        "\n## Calibration by decile (test)\n", md_table(deciles, 2),
        "\n## Threshold sensitivity (test)\n", md_table(thr, 3),
        "\n## AUC by issue quarter\n", md_table(by_q, 4),
        "\n## Target-definition sensitivity\n", md_table(target_check, 4),
    ]
    (REPORTS_DIR / "phase10_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:4]))
    print(md_table(oot, 4))
    print(md_table(csi, 4))
    print(md_table(target_check, 4))
    print(md_table(deciles, 2))


if __name__ == "__main__":
    main()
