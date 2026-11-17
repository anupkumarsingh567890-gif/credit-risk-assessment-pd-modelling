"""
Phase 6 - Model development: train and tune the four PD models.

Run from the project root:   python -m scripts.run_phase6
(takes a few minutes: Random Forest and XGBoost are tuned with time-series CV)

Outputs: models/<model>.joblib, data/processed/test_predictions.parquet,
         reports/tables/models_*.csv, reports/phase6_model_development_report.md
"""
import time

import joblib
import pandas as pd

from src.config import (
    ALL_FEATURES, MODEL_FILES, MODELS_DIR, REPORTS_DIR, TARGET_COL, TARGET_SPEC_COL, TEST_PREDICTIONS,
)
from src.models import decide_calibration, model_specs, out_of_fold, predict_pd, tune
from src.preprocessing import check_no_leakage
from src.scorecard import score_from_points
from src.utils import load_clean, md_table, require, save_table, train_test


def main() -> None:
    df = load_clean()
    train, test = train_test(df)  # already sorted by issue date
    X_train, y_train = train[ALL_FEATURES], train[TARGET_COL]
    X_test = test[ALL_FEATURES]

    check_no_leakage(ALL_FEATURES)
    require([MODELS_DIR / "scorecard.joblib"], "Run Phases 4-5 first: python run_all.py --from 4")
    card = joblib.load(MODELS_DIR / "scorecard.joblib")
    specs = model_specs()

    summary, folds_all, params_all = [], [], []
    preds = test[["id", "issue_d", "issue_quarter", TARGET_COL, TARGET_SPEC_COL, "grade",
                  "sub_grade", "int_rate", "term_months", "loan_amnt"]].copy()
    preds["score"] = score_from_points(test, card).to_numpy()
    train_preds = train[["id", TARGET_COL]].copy()

    for name, spec in specs.items():
        start = time.time()
        print(f"Tuning {name} ...", flush=True)
        search = tune(spec, X_train, y_train)
        best = search.best_estimator_

        oof, folds = out_of_fold(search.estimator.set_params(**search.best_params_), X_train, y_train)
        calibrator, cal_info = decide_calibration(oof, y_train)

        bundle = {"name": name, "pipeline": best, "calibrator": calibrator,
                  "features": ALL_FEATURES, "best_params": search.best_params_, **cal_info}
        joblib.dump(bundle, MODELS_DIR / f"{MODEL_FILES[name]}.joblib")

        if name == "Logistic Regression":
            lr_features = best.named_steps["woe"].features_
        col = MODEL_FILES[name]
        preds[f"pd_{col}"] = predict_pd(bundle, X_test)
        train_preds[f"pd_{col}"] = predict_pd(bundle, X_train)
        train_preds[f"oof_{col}"] = oof.to_numpy()

        folds.insert(0, "model", name)
        folds_all.append(folds)
        params_all += [{"model": name, "parameter": k.replace("model__", ""), "value": v}
                       for k, v in search.best_params_.items()]
        summary.append({
            "model": name,
            "cv_auc_mean": search.best_score_,
            "cv_auc_std": search.cv_results_["std_test_score"][search.best_index_],
            "configs_tried": len(search.cv_results_["params"]),
            **cal_info,
            "minutes": (time.time() - start) / 60,
        })
        print(f"  CV AUC {search.best_score_:.4f}  calibrated={cal_info['calibrated']}", flush=True)

    summary = pd.DataFrame(summary)
    folds_all = pd.concat(folds_all, ignore_index=True)
    params_all = pd.DataFrame(params_all)

    preds.to_parquet(TEST_PREDICTIONS, index=False)
    train_preds.to_parquet(TEST_PREDICTIONS.with_name("train_predictions.parquet"), index=False)
    save_table(summary, "models_cv_summary")
    save_table(folds_all, "models_cv_folds")
    save_table(params_all, "models_best_params")

    report = [
        "# Phase 6 - Model development report\n",
        "Cross-validation: TimeSeriesSplit with 4 folds on the training sample "
        "(2007-06 to 2011-06), ROC-AUC as the tuning metric.\n",
        "## Cross-validated performance of the best configuration\n", md_table(summary, 4),
        "\n## Fold-by-fold results (best configuration)\n", md_table(folds_all, 4),
        "\n## Selected hyperparameters\n", md_table(params_all),
        "\n## Logistic Regression features\n",
        "WoE binning and feature selection are re-run inside every CV fold. On the full training "
        f"sample the pipeline selects: {', '.join(lr_features)}.",
        "Identical to the Phase 4 scorecard." if sorted(lr_features) == sorted(card["features"])
        else f"Phase 4 scorecard selected: {', '.join(card['features'])}.",
    ]
    (REPORTS_DIR / "phase6_model_development_report.md").write_text("\n".join(report), encoding="utf-8")
    print(md_table(summary, 4))


if __name__ == "__main__":
    main()
