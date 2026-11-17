# Phase 4 - Scorecard report

Scaling: 600 points = odds 50:1 (good:bad); PDO = 20; factor = 28.8539; offset = 487.12.

Possible score range: 449 to 595.

## Information Value

| feature | iv | n_bins | strength |
|---|---|---|---|
| revol_util | 0.0803 | 6 | Weak |
| purpose | 0.0742 | 9 | Weak |
| loan_to_income | 0.0584 | 8 | Weak |
| term_months | 0.0566 | 2 | Weak |
| inq_last_6mths | 0.0513 | 4 | Weak |
| annual_inc | 0.0461 | 9 | Weak |
| has_public_record | 0.0279 | 2 | Weak |
| pub_rec | 0.0256 | 2 | Weak |
| revol_bal_to_income | 0.0221 | 9 | Weak |
| emp_title_missing | 0.0155 | 2 | Not useful |
| dti | 0.0139 | 6 | Not useful |
| loan_amnt | 0.0137 | 3 | Not useful |
| total_acc | 0.0104 | 6 | Not useful |
| open_acc | 0.0096 | 4 | Not useful |
| credit_history_years | 0.0096 | 5 | Not useful |
| emp_length_missing | 0.0084 | 2 | Not useful |
| mths_since_last_delinq | 0.0065 | 7 | Not useful |
| verification_status | 0.0063 | 3 | Not useful |
| emp_length_years | 0.0044 | 4 | Not useful |
| has_delinq_history | 0.0043 | 2 | Not useful |
| delinq_2yrs | 0.0031 | 2 | Not useful |
| home_ownership | 0.0029 | 4 | Not useful |
| revol_bal | 0.0026 | 5 | Not useful |
| pub_rec_bankruptcies | 0.0000 | 1 | Not useful |

## Feature selection

| feature | step | result |
|---|---|---|
| emp_title_missing | IV < 0.02 (or > 0.5) | removed |
| dti | IV < 0.02 (or > 0.5) | removed |
| loan_amnt | IV < 0.02 (or > 0.5) | removed |
| total_acc | IV < 0.02 (or > 0.5) | removed |
| open_acc | IV < 0.02 (or > 0.5) | removed |
| credit_history_years | IV < 0.02 (or > 0.5) | removed |
| emp_length_missing | IV < 0.02 (or > 0.5) | removed |
| mths_since_last_delinq | IV < 0.02 (or > 0.5) | removed |
| verification_status | IV < 0.02 (or > 0.5) | removed |
| emp_length_years | IV < 0.02 (or > 0.5) | removed |
| has_delinq_history | IV < 0.02 (or > 0.5) | removed |
| delinq_2yrs | IV < 0.02 (or > 0.5) | removed |
| home_ownership | IV < 0.02 (or > 0.5) | removed |
| revol_bal | IV < 0.02 (or > 0.5) | removed |
| pub_rec_bankruptcies | IV < 0.02 (or > 0.5) | removed |
| pub_rec | WoE correlation > 0.6 | removed |
| revol_bal_to_income | p-value 0.053 > 0.05 | removed |
| revol_util | final model | kept |
| purpose | final model | kept |
| loan_to_income | final model | kept |
| term_months | final model | kept |
| inq_last_6mths | final model | kept |
| annual_inc | final model | kept |
| has_public_record | final model | kept |

## Final logistic regression (WoE inputs)

| term | coefficient | std_error | p_value | vif | iv |
|---|---|---|---|---|---|
| const | -1.9392 | 0.0191 | 0.0000 |  |  |
| revol_util | -1.0474 | 0.0672 | 0.0000 | 1.0154 | 0.0803 |
| purpose | -0.9710 | 0.0678 | 0.0000 | 1.0098 | 0.0742 |
| loan_to_income | -0.4920 | 0.0817 | 0.0000 | 1.1983 | 0.0584 |
| term_months | -0.9211 | 0.0784 | 0.0000 | 1.0696 | 0.0566 |
| inq_last_6mths | -1.0766 | 0.0814 | 0.0000 | 1.0101 | 0.0513 |
| annual_inc | -1.0319 | 0.0934 | 0.0000 | 1.1463 | 0.0461 |
| has_public_record | -0.9138 | 0.1048 | 0.0000 | 1.0059 | 0.0279 |

## Scorecard points

| feature | bin | woe | bad_rate_pct | share_pct | coefficient | points |
|---|---|---|---|---|---|---|
| revol_util | (-inf, 23.8] | 0.411 | 8.726 | 25.056 | -1.047 | 90 |
| revol_util | (23.8, 38.2] | 0.220 | 10.369 | 14.894 | -1.047 | 84 |
| revol_util | (38.2, 56.5] | -0.012 | 12.734 | 19.964 | -1.047 | 77 |
| revol_util | (56.5, 71.3] | -0.144 | 14.268 | 15.041 | -1.047 | 73 |
| revol_util | (71.3, 87.4] | -0.232 | 15.379 | 14.909 | -1.047 | 71 |
| revol_util | (87.4, inf] or Missing | -0.461 | 18.603 | 10.136 | -1.047 | 64 |
| purpose | RARE (educational, house, medical, moving, vacation) | -0.122 | 13.997 | 6.343 | -0.971 | 74 |
| purpose | car | 0.529 | 7.799 | 4.233 | -0.971 | 92 |
| purpose | credit_card | 0.350 | 9.215 | 12.381 | -0.971 | 87 |
| purpose | debt_consolidation | -0.039 | 13.048 | 45.236 | -0.971 | 76 |
| purpose | home_improvement | 0.163 | 10.907 | 7.770 | -0.971 | 82 |
| purpose | major_purchase | 0.269 | 9.911 | 6.181 | -0.971 | 85 |
| purpose | other | -0.132 | 14.120 | 10.730 | -0.971 | 74 |
| purpose | small_business | -0.797 | 24.233 | 4.663 | -0.971 | 55 |
| purpose | wedding | 0.324 | 9.389 | 2.462 | -0.971 | 87 |
| loan_to_income | (-inf, 0.05747] | 0.259 | 10.007 | 10.008 | -0.492 | 81 |
| loan_to_income | (0.05747, 0.07111] | 0.245 | 10.117 | 5.004 | -0.492 | 81 |
| loan_to_income | (0.07111, 0.1455] | 0.200 | 10.557 | 30.023 | -0.492 | 80 |
| loan_to_income | (0.1455, 0.16] | 0.132 | 11.200 | 5.503 | -0.492 | 79 |
| loan_to_income | (0.16, 0.2034] | 0.034 | 12.227 | 14.461 | -0.492 | 78 |
| loan_to_income | (0.2034, 0.2442] | -0.051 | 13.169 | 10.000 | -0.492 | 77 |
| loan_to_income | (0.2442, 0.3] | -0.193 | 14.883 | 10.008 | -0.492 | 75 |
| loan_to_income | (0.3, inf] | -0.434 | 18.204 | 14.993 | -0.492 | 71 |
| term_months | (-inf, 36] | 0.147 | 11.066 | 76.213 | -0.921 | 81 |
| term_months | (36, inf] | -0.387 | 17.505 | 23.787 | -0.921 | 67 |
| inq_last_6mths | (-inf, 0] | 0.219 | 10.374 | 48.303 | -1.077 | 84 |
| inq_last_6mths | (0, 1] | -0.088 | 13.599 | 27.407 | -1.077 | 75 |
| inq_last_6mths | (1, 2] | -0.123 | 14.015 | 14.736 | -1.077 | 74 |
| inq_last_6mths | (2, inf] | -0.473 | 18.779 | 9.553 | -1.077 | 63 |
| annual_inc | (-inf, 2.39e+04] | -0.439 | 18.269 | 5.000 | -1.032 | 65 |
| annual_inc | (2.39e+04, 3.64e+04] | -0.248 | 15.597 | 15.030 | -1.032 | 70 |
| annual_inc | (3.64e+04, 4e+04] | -0.165 | 14.525 | 5.910 | -1.032 | 73 |
| annual_inc | (4e+04, 6e+04] | -0.019 | 12.820 | 29.418 | -1.032 | 77 |
| annual_inc | (6e+04, 6.5e+04] | 0.037 | 12.183 | 5.209 | -1.032 | 79 |
| annual_inc | (6.5e+04, 7e+04] | 0.049 | 12.053 | 5.022 | -1.032 | 79 |
| annual_inc | (7e+04, 8.1e+04] | 0.115 | 11.383 | 9.443 | -1.032 | 81 |
| annual_inc | (8.1e+04, 1e+05] | 0.250 | 10.090 | 11.017 | -1.032 | 85 |
| annual_inc | (1e+05, inf] | 0.352 | 9.203 | 13.951 | -1.032 | 88 |
| has_public_record | 0 | 0.055 | 12.010 | 92.006 | -0.914 | 79 |
| has_public_record | 1 | -0.511 | 19.367 | 7.994 | -0.914 | 64 |

## Performance vs LendingClub benchmarks

| model | sample | roc_auc | gini | ks | mean_pd | observed_rate |
|---|---|---|---|---|---|---|
| Scorecard (WoE logistic) | train | 0.6679 | 0.3358 | 0.2420 | 0.1298 | 0.1260 |
| Benchmark: LendingClub sub-grade | train | 0.6429 | 0.2858 | 0.2097 |  |  |
| Benchmark: interest rate | train | 0.6398 | 0.2797 | 0.2056 |  |  |
| Scorecard (WoE logistic) | test | 0.6743 | 0.3486 | 0.2521 | 0.1374 | 0.1414 |
| Benchmark: LendingClub sub-grade | test | 0.6557 | 0.3113 | 0.2311 |  |  |
| Benchmark: interest rate | test | 0.6562 | 0.3124 | 0.2340 |  |  |