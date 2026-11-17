# 3. Methodology

The project runs in ten scripted phases (`python run_all.py`). Every fitted quantity is learned on the **training sample only**: WoE bins, outlier caps, feature selection, hyperparameters, calibration, band cut-offs and the decision threshold. The out-of-time test sample is used only for reporting.

## 1. Data understanding (Phase 1)
- Profile every column: type, missing %, cardinality, duplicates and constants.
- Validate that the file is the expected extract (required columns; warn if not 39,717 × 111).
- Tag each column by **when its value becomes known**: application, origination or after origination. That timing tag is how leakage is identified. A model that scores a borrower at origination may only use what is known at origination.

## 2. Target definition
**Bad = charged off, and the delinquency that led to charge-off began within the first 36 months on book.**

Why a fixed 36-month window, and not "Charged Off vs Fully Paid, Current excluded"?
- All 1,140 Current loans are 60-month loans issued in 2011, the test period.
- Excluding them keeps the 2011 60-month loans that defaulted early and drops the survivors. That inflates the test default rate (2011 60-month: 27.4% excluding Current vs 23.3% counting them as good).
- Every loan has at least 53 months of history at the May 2016 snapshot, so a 36-month window is **fully observed for every cohort**. Current loans, which have repaid for 53+ months, are correctly good.
- The pipeline refuses to run if any loan has less than 36 months of observation.

**Estimating when default began.** LendingClub gives only cumulative totals and the last payment date, not a payment history. The onset of the delinquency is taken as the **earliest** of three upper bounds:

| Bound | Captures |
|---|---|
| (a) First instalment not covered by principal repaid. For an annuity with principal P, monthly rate r and instalment A, the balance after k on-time payments is $B_k = A/r - (A/r - P)(1+r)^k$. Solving for k with $B_k = P - \text{principal repaid}$ gives the instalments covered; onset = ⌊k⌋ + 1 | Borrowers in arrears who kept making partial or late payments |
| (b) Month after the last payment received | Borrowers who simply stopped paying, including those who had prepaid |
| (c) Contractual term | Any balance left at maturity is a missed payment, so a 36-month charge-off is always inside its own 36 months |

**Why not the last payment date alone?** For 95.5% of charged-off loans, bound (a) is earlier than (b). The median gap is **2 months**, because borrowers in arrears often keep making partial or late payments. Using the last payment date alone therefore dates default too late. The earlier version of this project did that, and it also applied a special case for 36-month loans. The current rule:
- applies **identically to 36- and 60-month loans**;
- fixes an off-by-one at the window boundary (a last payment in month 36 means the first missed instalment is in month 37);
- reclassifies 47 late-paying 60-month loans as bad (5,149 → 5,196 bads).

Repayment fields are used **only** to build the label and are dropped before modelling. The original definition is kept as `default_spec` for a sensitivity check.

## 3. Preprocessing (Phase 2)
- **Chronological split** at 1 July 2011. A random split would mix loans from the same months into train and test, and overstate performance on future borrowers.
- **Format fixes:** percentages, terms and employment length converted to numbers; two-digit years corrected.
- **Feature engineering**, using only information known at origination:
  - credit-history length;
  - loan-to-income and revolving-balance-to-income;
  - missing-value flags, kept because Phase 1 showed that missingness itself carries information.
- **Winsorising** (capping) heavy-tailed variables at percentiles learned on train.

## 4. Exploratory analysis (Phase 3)
- Default rates by driver: univariate and bivariate views, distributions, a correlation matrix and two-way default-rate tables.
- **Vintage curves:** cumulative charge-off rate by months on book for each issue-year cohort, using the estimated onset month.
- EDA uses the training sample, so the test period cannot influence feature choices.

## 5. Weight of Evidence and Information Value (Phase 4)

$$WoE_i = \ln\left(\frac{\%\text{Good}_i}{\%\text{Bad}_i}\right), \qquad IV = \sum_i (\%\text{Good}_i - \%\text{Bad}_i)\times WoE_i$$

WoE converts every characteristic, numeric or categorical, to one scale that is linear in the log-odds. That is what logistic regression assumes. It also handles missing values and outliers.

| IV | Strength |
|---|---|
| < 0.02 | Not useful |
| 0.02–0.1 | Weak |
| 0.1–0.3 | Medium |
| 0.3–0.5 | Strong |
| > 0.5 | Suspicious (check for leakage) |

**Binning rules**
- **Numeric:** start with 20 quantile bins. Merge neighbours until the bad rate is monotonic, in the direction of the Spearman correlation. Then merge any bin with less than 5% of loans into the neighbour with the closest bad rate. Merging neighbours of a monotonic sequence keeps it monotonic.
- **Missing values:** these get their own bin. If that bin holds less than 5% of loans, it is pooled with the bin of closest bad rate. (`revol_util` has 44 missing loans in train; their own WoE would have cost 47 points on a sample that small.)
- **Categorical:** one bin per category; categories under 2% of loans are pooled into "RARE".
- **Unseen at scoring:** a value never seen in training is scored as the **riskiest** bin, the conservative choice.
- **Smoothing:** 0.5 is added to each good and bad count, and the distributions are re-normalised to sum to 1.

## 6. Scorecard (Phase 4)

**Feature selection**
1. Keep characteristics with 0.02 ≤ IV < 0.5.
2. Walk from the highest IV down, and drop any characteristic whose WoE correlates above 0.6 with one already kept.
3. Fit a logistic regression and remove, one at a time, characteristics with a **positive** coefficient (wrong sign, because a higher WoE must mean a lower PD) or p > 0.05.

**Logistic regression**

$$\text{logit}(PD) = \ln\frac{PD}{1-PD} = \beta_0 + \sum_k \beta_k\,WoE_k$$

**Scaling – Points to Double the Odds** (600 points = 50:1 odds, PDO = 20)

$$\text{Factor} = \frac{PDO}{\ln 2} = 28.85, \qquad \text{Offset} = 600 - \text{Factor}\times\ln 50 = 487.12$$

$$\text{Score} = \text{Offset} + \text{Factor}\times\ln(\text{odds of good}) = \text{Offset} - \text{Factor}\Big(\beta_0 + \sum_k \beta_k WoE_k\Big)$$

$$\text{Points}_{kj} = -\Big(\beta_k WoE_{kj} + \frac{\beta_0}{n}\Big)\text{Factor} + \frac{\text{Offset}}{n}, \qquad PD = \frac{1}{1 + e^{(\text{Score}-\text{Offset})/\text{Factor}}}$$

Tests check that the points add up to the scaled logit (up to rounding), that +20 points doubles the odds, and that score and PD convert back and forth exactly.

## 7. Risk classification (Phase 5)
- **Bands** come from the training-score distribution: High = worst 20%, Medium = next 40%, Low = best 40%. The cut-off scores (531 and 551) are then fixed.
- **Cut-off analysis:** for each candidate cut-off, measure the approval rate, the bad rate among approved loans, the % of defaulters rejected and the % of good borrowers rejected. The **KS cut-off** maximises (bads rejected − goods rejected). It is chosen on train (542) and reported on test.
- **Decision rule:** High band → Decline; score ≥ cut-off → Approve; otherwise Refer.

## 8. PD models (Phase 6)
| Model | Role |
|---|---|
| Logistic Regression (WoE) | Industry standard, interpretable; the scorecard's engine |
| Decision Tree | Simple rules |
| Random Forest | Averaged de-correlated trees; interactions |
| XGBoost | Gradient-boosted trees; usually the strongest on tabular data |

- **Time-series cross-validation (4 folds):** each fold trains on older loans and validates on the next block of newer loans.
- **No leakage inside cross-validation.** For logistic regression, `WoEEncoder` re-runs WoE binning **and** the full IV / correlation / p-value selection inside every fold. In the earlier version, the features were selected once on the whole training sample. That let the selection see the validation folds and inflated the logistic regression CV AUC (0.655 then vs 0.651 now). Because the champion is chosen on CV AUC, that version was biased towards logistic regression.
- **Tuning:** grid search for logistic regression (C) and the Decision Tree (depth, leaf size); random search for Random Forest (12 configurations) and XGBoost (15). The metric is ROC-AUC.
- **No class weights.** Re-weighting distorts probabilities. Imbalance is handled by the decision threshold (the training default rate).
- **Calibration:** Platt scaling, a logistic regression on the logit of the predicted PD. It is fitted on out-of-fold predictions and kept only if it lowers the Brier score by at least 1% in a **cross-fitted** comparison: each block of out-of-fold predictions is calibrated by a model fitted on the other blocks.

## 9. Evaluation (Phase 7)
| Metric | Meaning |
|---|---|
| ROC-AUC | P(a random defaulter gets a higher PD than a random non-defaulter) |
| Gini | 2 × AUC − 1 |
| KS | max over thresholds of (TPR − FPR) |
| Brier | mean((PD − outcome)²), the accuracy of the probabilities |
| Accuracy, Precision, Recall, F1, confusion matrix | At PD ≥ training default rate (12.6%) |

**Champion rule**, fixed in `config.py`: highest cross-validated AUC, unless logistic regression is within 0.01 AUC.

## 10. Borrower assessment (Phase 8)
Every application is validated before it is scored:
- impossible values raise an error (non-positive income, more open lines than total lines, bankruptcies exceeding public records, unknown categories);
- values outside the training range produce a warning.

The output is: score, scorecard PD, champion PD, band, decision, and **reason codes**, which are the characteristics that lost the most points against their best bin.

## 11. Portfolio analysis (Phase 9)
- Risk-band mix by count and by exposure.
- Predicted vs observed default rates.
- Default rates by grade, income, purpose, interest rate and term.
- **High-risk concentration:** a segment's share of High-risk loans divided by its share of the portfolio.

## 12. Validation (Phase 10)

**Population / Characteristic Stability Index**

$$PSI = \sum_b (\%A_b - \%E_b)\ln\frac{\%A_b}{\%E_b}$$

Buckets are the training-score deciles (or the WoE bins for CSI). PSI below 0.10 means stable, 0.10–0.25 means monitor, above 0.25 means a significant shift.

**Hosmer-Lemeshow**

$$HL = \sum_{g=1}^{10}\frac{(O_g - E_g)^2}{E_g(1 - E_g/n_g)}$$

On the development sample HL is compared with $\chi^2_{g-2}$. On an **independent validation sample**, where no parameters were estimated, it is compared with $\chi^2_{g}$. The earlier version used g − 2 on the test sample. That was corrected; it makes the test slightly more lenient, and the binomial test below was added so calibration is still checked strictly.

**Calibration-in-the-large (binomial test)**

$$z = \frac{\sum y - \sum PD}{\sqrt{\sum PD(1-PD)}}$$

This tests whether the overall PD level matches the observed default rate.

**Also covered:**
- train vs CV vs test AUC (overfitting);
- CV fold spread;
- tree importances and permutation importance (the drop in test AUC when a feature is shuffled);
- threshold sensitivity;
- AUC by issue quarter;
- AUC against the original target definition;
- rank agreement between scorecard PD and champion PD.

Every traffic-light rule is a named constant in `config.py`.
