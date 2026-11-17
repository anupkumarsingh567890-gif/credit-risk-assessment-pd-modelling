# Phase 10 - Model validation report

Champion: **XGBoost**.

## Validation summary

| check | result | rule | status |
|---|---|---|---|
| Out-of-time discrimination | test AUC 0.681 (Gini 0.362) | AUC >= 0.65 | Pass |
| Overfitting (train - test AUC) | +0.042 | gap <= 0.05 | Pass |
| CV stability (AUC std across folds) | 0.021 | std <= 0.03 | Pass |
| Score PSI (train -> test) | 0.014 | PSI < 0.1 | Pass |
| Max characteristic CSI | 0.044 (term_months) | CSI < 0.1 | Pass |
| Calibration-in-the-large (binomial test) | mean PD 13.33% vs observed 14.14%; z = +2.74, p = 0.006 | p >= 0.05 | Monitor |
| Hosmer-Lemeshow (test, 10 groups, 10 d.f.) | HL = 29.6, p = 0.001 | p >= 0.05 | Monitor |
| Scorecard PD vs champion PD (rank correlation) | 0.902 | Spearman >= 0.9 | Pass |

## Out-of-time and overfitting

| model | train_auc | cv_auc | test_auc | train_test_gap | overfitting |
|---|---|---|---|---|---|
| Logistic Regression | 0.6682 | 0.6508 | 0.6745 | -0.0063 | No |
| Decision Tree | 0.6660 | 0.6109 | 0.6456 | 0.0204 | No |
| Random Forest | 0.7939 | 0.6585 | 0.6721 | 0.1217 | Yes |
| XGBoost | 0.7227 | 0.6653 | 0.6811 | 0.0416 | No |

## Cross-validation AUC by fold (time-series folds)

| fold | Decision Tree | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| 1 | 0.5925 | 0.6174 | 0.6207 | 0.6345 |
| 2 | 0.5997 | 0.6318 | 0.6480 | 0.6567 |
| 3 | 0.6149 | 0.6752 | 0.6889 | 0.6897 |
| 4 | 0.6364 | 0.6790 | 0.6764 | 0.6804 |

## Population stability

| measure | value | status |
|---|---|---|
| Credit score PSI | 0.0141 | Stable |
| PD PSI (XGBoost) | 0.0130 | Stable |

### Characteristic stability (scorecard features)

| feature | csi | status |
|---|---|---|
| term_months | 0.0439 | Stable |
| loan_to_income | 0.0324 | Stable |
| purpose | 0.0265 | Stable |
| revol_util | 0.0224 | Stable |
| has_public_record | 0.0165 | Stable |
| annual_inc | 0.0117 | Stable |
| inq_last_6mths | 0.0023 | Stable |

## Permutation importance (champion, test)

| feature | auc_drop_mean | auc_drop_std |
|---|---|---|
| revol_util | 0.0451 | 0.0019 |
| annual_inc | 0.0263 | 0.0025 |
| term_months | 0.0258 | 0.0014 |
| purpose | 0.0182 | 0.0011 |
| inq_last_6mths | 0.0181 | 0.0011 |
| loan_to_income | 0.0101 | 0.0019 |
| credit_history_years | 0.0053 | 0.0006 |
| revol_bal | 0.0028 | 0.0005 |
| emp_title_missing | 0.0028 | 0.0012 |
| open_acc | 0.0021 | 0.0008 |
| total_acc | 0.0017 | 0.0004 |
| mths_since_last_delinq | 0.0016 | 0.0006 |
| dti | 0.0013 | 0.0005 |
| has_delinq_history | 0.0013 | 0.0004 |
| revol_bal_to_income | 0.0011 | 0.0010 |
| delinq_2yrs | 0.0008 | 0.0004 |
| has_public_record | 0.0007 | 0.0005 |
| home_ownership | 0.0004 | 0.0001 |
| verification_status | 0.0002 | 0.0002 |
| emp_length_missing | 0.0001 | 0.0001 |
| pub_rec_bankruptcies | 0.0001 | 0.0001 |
| loan_amnt | 0.0000 | 0.0009 |
| pub_rec | -0.0002 | 0.0004 |
| emp_length_years | -0.0008 | 0.0004 |

## Calibration by decile (test)

| decile | loans | observed | expected | avg_pd_pct | observed_rate_pct |
|---|---|---|---|---|---|
| 1 | 1,246 | 41 | 54.93 | 4.41 | 3.29 |
| 2 | 1,246 | 77 | 81.49 | 6.54 | 6.18 |
| 3 | 1,246 | 122 | 100.38 | 8.06 | 9.79 |
| 4 | 1,245 | 97 | 118.20 | 9.49 | 7.79 |
| 5 | 1,246 | 147 | 136.56 | 10.96 | 11.80 |
| 6 | 1,246 | 173 | 157.28 | 12.62 | 13.88 |
| 7 | 1,245 | 209 | 180.46 | 14.49 | 16.79 |
| 8 | 1,246 | 236 | 211.95 | 17.01 | 18.94 |
| 9 | 1,246 | 287 | 255.83 | 20.53 | 23.03 |
| 10 | 1,246 | 373 | 363.42 | 29.17 | 29.94 |

## Threshold sensitivity (test)

| threshold | flagged_pct | precision | recall | f1 | accuracy |
|---|---|---|---|---|---|
| 0.050 | 93.073 | 0.150 | 0.989 | 0.261 | 0.207 |
| 0.075 | 78.921 | 0.166 | 0.929 | 0.282 | 0.332 |
| 0.100 | 61.599 | 0.188 | 0.819 | 0.306 | 0.474 |
| 0.125 | 45.698 | 0.213 | 0.687 | 0.325 | 0.596 |
| 0.150 | 32.533 | 0.233 | 0.537 | 0.325 | 0.685 |
| 0.175 | 23.214 | 0.259 | 0.425 | 0.321 | 0.747 |
| 0.200 | 15.998 | 0.274 | 0.310 | 0.291 | 0.786 |
| 0.225 | 10.860 | 0.290 | 0.223 | 0.252 | 0.813 |
| 0.250 | 7.152 | 0.314 | 0.159 | 0.211 | 0.832 |
| 0.275 | 4.921 | 0.318 | 0.111 | 0.164 | 0.841 |
| 0.300 | 3.267 | 0.349 | 0.081 | 0.131 | 0.849 |
| 0.325 | 2.256 | 0.359 | 0.057 | 0.099 | 0.852 |
| 0.350 | 1.397 | 0.385 | 0.038 | 0.069 | 0.855 |
| 0.375 | 0.963 | 0.433 | 0.030 | 0.055 | 0.857 |
| 0.400 | 0.618 | 0.468 | 0.020 | 0.039 | 0.858 |

## AUC by issue quarter

| period | loans | default_rate_pct | auc_pd_xgboost | auc_pd_logistic_regression |
|---|---|---|---|---|
| 2008Q1 | 581 | 16.5232 | 0.6913 | 0.5895 |
| 2008Q4 | 503 | 15.7058 | 0.7084 | 0.6641 |
| 2009Q1 | 775 | 12.0000 | 0.6223 | 0.5978 |
| 2009Q2 | 965 | 12.9534 | 0.6886 | 0.6313 |
| 2009Q3 | 1,231 | 11.6978 | 0.7062 | 0.6471 |
| 2009Q4 | 1,745 | 13.2951 | 0.6902 | 0.6270 |
| 2010Q1 | 1,953 | 10.6503 | 0.7231 | 0.6621 |
| 2010Q2 | 2,776 | 12.8602 | 0.7571 | 0.6965 |
| 2010Q3 | 3,283 | 12.0926 | 0.7136 | 0.6571 |
| 2010Q4 | 3,520 | 11.1932 | 0.7443 | 0.6882 |
| 2011Q1 | 4,120 | 12.0631 | 0.7402 | 0.6828 |
| 2011Q2 | 5,078 | 13.6865 | 0.7309 | 0.6873 |
| 2011Q3 | 5,861 | 13.3766 | 0.6881 | 0.6831 |
| 2011Q4 | 6,597 | 14.8249 | 0.6738 | 0.6659 |

## Target-definition sensitivity

| target | loans | default_rate_pct | test_auc |
|---|---|---|---|
| Default within 36 months (main) | 12,458 | 14.1435 | 0.6811 |
| Charged Off vs Fully Paid, Current excluded | 11,501 | 16.6768 | 0.7145 |