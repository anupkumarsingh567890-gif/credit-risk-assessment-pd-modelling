# 5. Model evaluation and validation report

Evaluation sample: the **out-of-time test**, 12,458 loans issued July–December 2011 with a 14.14% default rate. These loans were not used for any fitting, binning, feature selection, tuning, calibration or threshold choice.

## 5.1 Discrimination and classification
The threshold for the classification metrics is PD ≥ 12.60%, the training default rate.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Gini | KS | Brier |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.594 | 0.207 | 0.660 | 0.315 | 0.675 | 0.349 | 0.254 | 0.116 |
| Decision Tree | 0.663 | 0.214 | 0.517 | 0.303 | 0.646 | 0.291 | 0.218 | 0.118 |
| Random Forest | 0.570 | 0.203 | 0.697 | 0.315 | 0.672 | 0.344 | 0.251 | 0.117 |
| **XGBoost (champion)** | 0.600 | 0.213 | 0.677 | 0.324 | **0.681** | **0.361** | **0.273** | **0.115** |

How to read this:
- **Accuracy is misleading** with a 14% default rate. Flagging nobody would score 86%.
- **Precision of about 0.21** means one in five flagged loans defaults, about 1.5 times the base rate. That is typical for application scoring without a bureau score.
- **The ranking metrics and the Brier score are what matter for a PD model.** XGBoost leads on all of them, but only by 0.006–0.009 AUC over logistic regression and Random Forest.

Charts: `reports/figures/evaluation/`.

## 5.2 Validation summary (champion: XGBoost)
| Check | Result | Rule | Status |
|---|---|---|---|
| Out-of-time discrimination | AUC 0.681, Gini 0.361 | AUC ≥ 0.65 | Pass |
| Overfitting (train − test AUC) | +0.042 | ≤ 0.05 | Pass |
| CV stability (AUC std across folds) | 0.022 | ≤ 0.03 | Pass |
| Score PSI (train → test) | 0.014 | < 0.10 | Pass |
| Maximum characteristic CSI | 0.044 (term) | < 0.10 | Pass |
| Calibration-in-the-large (binomial) | mean PD 13.34% vs observed 14.14%; z = 2.71, p = 0.007 | p ≥ 0.05 | **Monitor** |
| Hosmer-Lemeshow (10 groups, 10 d.f.) | HL = 27.4, p = 0.002 | p ≥ 0.05 | **Monitor** |
| Scorecard PD vs champion PD (Spearman) | 0.902 | ≥ 0.90 | Pass (borderline) |

## 5.3 Overfitting
| Model | Train AUC | CV AUC | Test AUC | Train − test |
|---|---|---|---|---|
| Logistic Regression | 0.668 | 0.651 | 0.675 | −0.006 |
| Decision Tree | 0.666 | 0.611 | 0.646 | 0.020 |
| Random Forest | 0.794 | 0.659 | 0.672 | **0.122: overfits** |
| XGBoost | 0.723 | 0.665 | 0.681 | 0.042 |

Random Forest memorises the training data and generalises no better than logistic regression. XGBoost's AUC on the 2011 Q1–Q2 training quarters (0.73–0.74, in-sample) falls to 0.69 and 0.67 in the two test quarters. Its out-of-time edge is therefore smaller than its in-sample fit suggests.

## 5.4 Stability
- **Score PSI 0.014:** the score distribution is essentially unchanged between train and test.
- **CSI:** every scorecard characteristic is below 0.05. The largest is term (0.044), reflecting the higher share of 60-month loans in late 2011. Purpose CSI is 0.027 now that rare purposes are pooled. Before pooling it was 0.136, driven by small categories.
- **AUC by issue quarter** (`reports/figures/validation/05_auc_by_quarter.png`):
  - 2011 Q3: champion 0.688, scorecard 0.683;
  - 2011 Q4: champion 0.673, scorecard 0.666.

## 5.5 Calibration (champion, test)
| PD decile | Average predicted PD | Observed default rate |
|---|---|---|
| 1 (safest) | 4.4% | 3.2% |
| 3 | 8.1% | 9.7% |
| 5 | 11.0% | 11.0% |
| 7 | 14.5% | 16.7% |
| 8 | 17.0% | 19.6% |
| 10 (riskiest) | 29.1% | 29.6% |

- **The PD level is too low** in the test period: 13.3% predicted vs 14.1% observed (binomial p = 0.007). The gap sits mainly in deciles 6–9.
- **Ranking is preserved:** observed rates rise from decile 1 to 10.
- **Cause:** the test period was riskier than the training period (14.1% vs 12.6% default rate). This is a level shift, not a ranking problem.
- **Production fix:** recalibrate the PD intercept to recent default experience and monitor quarterly.

## 5.6 Threshold sensitivity (champion)
| PD threshold | Loans flagged | Precision | Recall |
|---|---|---|---|
| 0.100 | 61.5% | 0.186 | 0.809 |
| 0.125 (≈ chosen 0.126) | 45.6% | 0.212 | 0.685 |
| 0.150 | 32.6% | 0.234 | 0.540 |
| 0.200 | 16.3% | 0.274 | 0.314 |

The full grid is in `reports/tables/validation_threshold_sensitivity.csv`. You can try any threshold on the dashboard's Model Evaluation page.

## 5.7 Feature importance
- **Scorecard** (|coefficient| × standard deviation of WoE): revol_util 20%, purpose 17%, inquiries 16%, income 15%, term 14%, public record 10%, loan-to-income 8%.
- **XGBoost, permutation importance** (drop in test AUC when a feature is shuffled): revol_util 0.044, term 0.025, annual_inc 0.025, purpose 0.018, inquiries 0.018, loan_to_income 0.011, credit history 0.005. Several features contribute nothing or slightly negatively out-of-time (loan_amnt, emp_length_years, pub_rec), which suggests a leaner model would lose nothing.

## 5.8 Target-definition sensitivity
| Target | Test loans | Default rate | Champion AUC |
|---|---|---|---|
| Default within 36 months (main) | 12,458 | 14.1% | 0.681 |
| Charged Off vs Fully Paid, Current excluded | 11,501 | 16.7% | 0.714 |

The model ranks borrowers well under both definitions. The higher AUC on the original definition reflects that definition's maturity bias: dropping the surviving 60-month loans makes the classes look more separable than they are.

## 5.9 Conclusion
- **XGBoost is the PD champion** under the pre-registered rule, with a modest and real out-of-time advantage (Gini 0.361 vs 0.349).
- **The WoE scorecard is retained for decisioning.** It is transparent, stable and clearly better than LendingClub's sub-grade (Gini 0.349 vs 0.311).
- **Two items need monitoring:** the PD level under-predicts in the test period, and Random Forest overfits (it is not used).
