# Phase 6 - Model development report

Cross-validation: TimeSeriesSplit with 4 folds on the training sample (2007-06 to 2011-06), ROC-AUC as the tuning metric.

## Cross-validated performance of the best configuration

| model | cv_auc_mean | cv_auc_std | configs_tried | oof_brier_raw | oof_brier_calibrated | calibrated | minutes |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6508 | 0.0268 | 4 | 0.1040 | 0.1041 | 0 | 0.1634 |
| Decision Tree | 0.6109 | 0.0168 | 20 | 0.1072 | 0.1060 | 1 | 0.1125 |
| Random Forest | 0.6585 | 0.0264 | 12 | 0.1042 | 0.1045 | 0 | 0.5962 |
| XGBoost | 0.6653 | 0.0215 | 15 | 0.1040 | 0.1039 | 0 | 0.3918 |

## Fold-by-fold results (best configuration)

| model | fold | train_loans | valid_loans | valid_auc | valid_brier |
|---|---|---|---|---|---|
| Logistic Regression | 1 | 5,455 | 5,451 | 0.6174 | 0.1008 |
| Logistic Regression | 2 | 10,906 | 5,451 | 0.6318 | 0.1042 |
| Logistic Regression | 3 | 16,357 | 5,451 | 0.6752 | 0.1001 |
| Logistic Regression | 4 | 21,808 | 5,451 | 0.6790 | 0.1110 |
| Decision Tree | 1 | 5,455 | 5,451 | 0.5925 | 0.1055 |
| Decision Tree | 2 | 10,906 | 5,451 | 0.5997 | 0.1066 |
| Decision Tree | 3 | 16,357 | 5,451 | 0.6149 | 0.1028 |
| Decision Tree | 4 | 21,808 | 5,451 | 0.6364 | 0.1139 |
| Random Forest | 1 | 5,455 | 5,451 | 0.6207 | 0.1013 |
| Random Forest | 2 | 10,906 | 5,451 | 0.6480 | 0.1038 |
| Random Forest | 3 | 16,357 | 5,451 | 0.6889 | 0.1001 |
| Random Forest | 4 | 21,808 | 5,451 | 0.6764 | 0.1118 |
| XGBoost | 1 | 5,455 | 5,451 | 0.6345 | 0.1011 |
| XGBoost | 2 | 10,906 | 5,451 | 0.6567 | 0.1042 |
| XGBoost | 3 | 16,357 | 5,451 | 0.6897 | 0.0994 |
| XGBoost | 4 | 21,808 | 5,451 | 0.6804 | 0.1112 |

## Selected hyperparameters

| model | parameter | value |
|---|---|---|
| Logistic Regression | C | 1.00 |
| Decision Tree | max_depth | 6 |
| Decision Tree | min_samples_leaf | 50 |
| Random Forest | n_estimators | 200 |
| Random Forest | min_samples_leaf | 20 |
| Random Forest | max_features | sqrt |
| Random Forest | max_depth | 10 |
| XGBoost | subsample | 0.70 |
| XGBoost | reg_lambda | 10 |
| XGBoost | n_estimators | 600 |
| XGBoost | min_child_weight | 10 |
| XGBoost | max_depth | 3 |
| XGBoost | learning_rate | 0.02 |
| XGBoost | colsample_bytree | 0.80 |

## Logistic Regression features

WoE binning and feature selection are re-run inside every CV fold. On the full training sample the pipeline selects: revol_util, purpose, loan_to_income, term_months, inq_last_6mths, annual_inc, has_public_record.
Identical to the Phase 4 scorecard.