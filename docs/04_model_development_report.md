# 4. Model development report

All figures come from the reference run: `python run_all.py`, Python 3.12, the versions pinned in `requirements.txt`, random seed 42. The auto-generated reports in `reports/` (`phase4_scorecard_report.md`, `phase6_model_development_report.md`, …) are the source of truth after any code change.

## 4.1 Development data
| Sample | Issue dates | Loans | 36-month default rate |
|---|---|---|---|
| Train | Jun 2007 – Jun 2011 | 27,259 | 12.60% |
| Test (out-of-time) | Jul – Dec 2011 | 12,458 | 14.14% |

The test period is riskier because it has more 60-month loans (33% of loans vs 24% in train). In training, 60-month loans default at 17.5% against 11.1% for 36-month loans.

## 4.2 Default drivers (EDA, training sample)
| Driver | Lowest group | Highest group |
|---|---|---|
| LendingClub grade | A: 5.3% | F: 23.9% |
| Interest rate | < 8%: 4.6% | 18%+: 24.5% |
| Loan purpose | car: 7.8% | small business: 24.2% |
| Revolving utilisation | 0–20%: 8.9% | 80%+: 17.2% |
| Inquiries (6 months) | 0: 10.4% | 3: 18.8% |
| Annual income | 100k+: 9.2% | < 30k: 16.7% |
| Loan-to-income | < 10%: 10.4% | 30%+: 18.1% |
| Term | 36 months: 11.1% | 60 months: 17.5% |
| Credit history | 30y+: 11.1% | < 5y: 16.1% |

Groups with less than 1% of loans are ignored. Charts are in `reports/figures/eda/`.

## 4.3 Credit scorecard

| Characteristic | IV | Bins | Result |
|---|---|---|---|
| revol_util | 0.080 | 6 | Kept |
| purpose | 0.074 | 9 (5 rare purposes pooled) | Kept |
| loan_to_income | 0.058 | 8 | Kept |
| term_months | 0.057 | 2 | Kept |
| inq_last_6mths | 0.051 | 4 | Kept |
| annual_inc | 0.046 | 9 | Kept |
| has_public_record | 0.028 | 2 | Kept |
| pub_rec | 0.026 | 2 | Dropped: WoE correlation > 0.6 with has_public_record |
| revol_bal_to_income | 0.022 | 9 | Dropped: p = 0.053 |
| 15 others | < 0.02 | – | Dropped: IV too low |

No characteristic reaches medium strength (IV ≥ 0.1), which is expected without a bureau score.

**Final logistic regression** (statsmodels, WoE inputs, training sample):

| Term | Coefficient | Std. error | p-value | VIF |
|---|---|---|---|---|
| Intercept | −1.939 | 0.019 | < 0.001 | – |
| revol_util | −1.047 | 0.067 | < 0.001 | 1.02 |
| purpose | −0.971 | 0.068 | < 0.001 | 1.01 |
| loan_to_income | −0.492 | 0.082 | < 0.001 | 1.20 |
| term_months | −0.921 | 0.078 | < 0.001 | 1.07 |
| inq_last_6mths | −1.077 | 0.081 | < 0.001 | 1.01 |
| annual_inc | −1.032 | 0.093 | < 0.001 | 1.15 |
| has_public_record | −0.914 | 0.105 | < 0.001 | 1.01 |

**Scaling:** 600 = 50:1, PDO 20, factor 28.85, offset 487.12. Score range 449–595. Points per bin are in `reports/tables/scorecard_points.csv`. The best and worst bins are summarised in the README.

**Scorecard vs LendingClub's own grading**

| | Train AUC | Train Gini | Test AUC | Test Gini | Test KS |
|---|---|---|---|---|---|
| Our scorecard | 0.668 | 0.336 | 0.674 | 0.349 | 0.252 |
| LendingClub sub-grade | 0.643 | 0.286 | 0.656 | 0.311 | 0.231 |
| LendingClub interest rate | 0.640 | 0.280 | 0.656 | 0.312 | 0.234 |

## 4.4 Risk classification
Band cut-offs from training scores: **High < 531 ≤ Medium < 551 ≤ Low**.

| Band | Train share | Train default rate | Test share | Test default rate | Test average scorecard PD |
|---|---|---|---|---|---|
| Low | 40.3% | 6.5% | 35.9% | 6.8% | 7.2% |
| Medium | 40.7% | 13.8% | 41.8% | 14.2% | 13.6% |
| High | 18.9% | 23.1% | 22.4% | 25.8% | 24.5% |

**Approval cut-off (test sample)**

| Cut-off score | Approval rate | Bad rate, approved | Defaulters rejected | Good borrowers rejected |
|---|---|---|---|---|
| none | 100% | 14.1% | 0% | 0% |
| 523 | 88.8% | 12.2% | 23.8% | 9.2% |
| 531 (decline High band) | 77.6% | 10.8% | 40.8% | 19.3% |
| **542 (KS-optimal, chosen on train)** | **56.0%** | **8.8%** | **65.4%** | **40.5%** |

The KS cut-off is the statistically best separating point, but it would reject 44% of applicants that LendingClub had in fact funded. A volume-focused lender would more likely decline only the High band.

## 4.5 PD models (time-series cross-validation on train)
| Model | CV AUC | Hyperparameters | Calibration |
|---|---|---|---|
| Logistic Regression (WoE) | 0.651 ± 0.027 | C = 1; bins and selection re-learned in each fold | not applied |
| Decision Tree | 0.611 ± 0.017 | max_depth 6, min_samples_leaf 50 | Platt scaling (Brier 0.1072 → 0.1060) |
| Random Forest | 0.659 ± 0.026 | 200 trees, max_depth 10, min_samples_leaf 20, max_features sqrt | not applied |
| XGBoost | 0.665 ± 0.022 | 600 trees, learning_rate 0.02, max_depth 3, min_child_weight 10, subsample 0.7, colsample 0.8, lambda 10 | not applied |

Fold AUCs rise from 0.59–0.64 in the earliest fold to 0.64–0.68 in the latest, because later folds train on more loans. Details: `reports/tables/models_cv_folds.csv`.

**Champion: XGBoost.** Its CV AUC lead over logistic regression (0.015) exceeds the 0.01 interpretability margin set in `config.py` before testing.

- The **scorecard** stays the decision tool: points, reason codes, cut-off.
- **XGBoost** supplies the PD estimate.

Fitted on the full training sample, the logistic-regression pipeline selects the same 7 characteristics as the Phase 4 scorecard.
