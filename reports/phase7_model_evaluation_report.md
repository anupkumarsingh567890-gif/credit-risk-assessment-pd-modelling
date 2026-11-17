# Phase 7 - Model evaluation report

Decision threshold: PD >= 0.1260 (training default rate).

## Out-of-time test performance

| model | accuracy | precision | recall | f1 | roc_auc | gini | ks | brier | mean_pd | observed_default_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.5938 | 0.2068 | 0.6600 | 0.3149 | 0.6745 | 0.3490 | 0.2543 | 0.1156 | 0.1333 | 0.1414 |
| Decision Tree | 0.6633 | 0.2141 | 0.5170 | 0.3028 | 0.6456 | 0.2912 | 0.2182 | 0.1183 | 0.1236 | 0.1414 |
| Random Forest | 0.5704 | 0.2031 | 0.6969 | 0.3145 | 0.6721 | 0.3443 | 0.2514 | 0.1165 | 0.1328 | 0.1414 |
| XGBoost | 0.5995 | 0.2132 | 0.6805 | 0.3246 | 0.6811 | 0.3621 | 0.2715 | 0.1154 | 0.1333 | 0.1414 |

## Train vs test (overfitting check)

| model | sample | accuracy | precision | recall | f1 | roc_auc | gini | ks | brier | mean_pd | observed_default_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | train | 0.6242 | 0.1907 | 0.6115 | 0.2908 | 0.6682 | 0.3365 | 0.2448 | 0.1056 | 0.1259 | 0.1260 |
| Logistic Regression | test | 0.5938 | 0.2068 | 0.6600 | 0.3149 | 0.6745 | 0.3490 | 0.2543 | 0.1156 | 0.1333 | 0.1414 |
| Decision Tree | train | 0.6894 | 0.2079 | 0.5215 | 0.2973 | 0.6660 | 0.3319 | 0.2391 | 0.1059 | 0.1196 | 0.1260 |
| Decision Tree | test | 0.6633 | 0.2141 | 0.5170 | 0.3028 | 0.6456 | 0.2912 | 0.2182 | 0.1183 | 0.1236 | 0.1414 |
| Random Forest | train | 0.6475 | 0.2360 | 0.8037 | 0.3649 | 0.7939 | 0.5877 | 0.4358 | 0.0993 | 0.1263 | 0.1260 |
| Random Forest | test | 0.5704 | 0.2031 | 0.6969 | 0.3145 | 0.6721 | 0.3443 | 0.2514 | 0.1165 | 0.1328 | 0.1414 |
| XGBoost | train | 0.6431 | 0.2136 | 0.6838 | 0.3255 | 0.7227 | 0.4454 | 0.3268 | 0.1017 | 0.1259 | 0.1260 |
| XGBoost | test | 0.5995 | 0.2132 | 0.6805 | 0.3246 | 0.6811 | 0.3621 | 0.2715 | 0.1154 | 0.1333 | 0.1414 |

## Confusion matrices (test)

| model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 6,234 | 4,462 | 599 | 1,163 |
| Decision Tree | 7,352 | 3,344 | 851 | 911 |
| Random Forest | 5,878 | 4,818 | 534 | 1,228 |
| XGBoost | 6,270 | 4,426 | 563 | 1,199 |

## Cross-validated AUC (selection basis)

| model | cv_auc_mean | cv_auc_std | calibrated |
|---|---|---|---|
| Logistic Regression | 0.6508 | 0.0268 | 0 |
| Decision Tree | 0.6109 | 0.0168 | 1 |
| Random Forest | 0.6585 | 0.0264 | 0 |
| XGBoost | 0.6653 | 0.0215 | 0 |

## Champion model: **XGBoost**
Highest cross-validated AUC (0.6653).
