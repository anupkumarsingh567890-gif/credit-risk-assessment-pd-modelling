# 6. Limitations and assumptions

> This is an educational analytical project. It is not a production lending system, credit policy, Basel regulatory capital model or IFRS 9 impairment model, and it has not been independently validated.

## Assumptions
| # | Assumption | Why it matters |
|---|---|---|
| A1 | The data snapshot is **May 2016**: the latest `last_pymnt_d` / `last_credit_pull_d` | Sets the observation window in the maturity check |
| A2 | The onset of the delinquency that led to charge-off is the **earliest** of: the first instalment not covered by principal repaid, the month after the last payment, and the term | Decides whether a 60-month charge-off falls inside the 36-month window. Each bound is an upper bound on the true onset, so the estimate can still be slightly late |
| A3 | Principal repaid is applied to scheduled instalments in order, on the stated annuity schedule (funded amount, interest rate, instalment) | Underlies bound (a); 0% of loans have an invalid schedule |
| A4 | Charge-off is used as the default event. LendingClub generally charges off loans at around 150 days past due, later than the Basel 90-days-past-due definition | The model predicts charge-off risk, not 90+ DPD |
| A5 | Bureau fields (`revol_util`, `inq_last_6mths`, `delinq_2yrs`, …) are **as at application** | The standard reading of this dataset; not independently verified. If any were refreshed later, they would leak information |
| A6 | A blank `mths_since_last_delinq` / `mths_since_last_record` means **no event on file** | Supported by the data: these loans default less |
| A7 | `earliest_cr_line` dates after the issue date are two-digit-year errors and are shifted back 100 years | Affects 90 loans |
| A8 | Current loans are **good** within the 36-month window | All have repaid for 53–62 months and were not charged off |
| A9 | Values unseen in training are scored as the riskiest bin | A conservative choice that can penalise genuinely new categories |
| A10 | The approval cut-off is the **KS point** | A statistical default only. In practice the cut-off comes from risk appetite and profitability |

## Limitations

**Data**
- **No FICO / bureau score**, so every characteristic is individually weak (IV < 0.1) and the Gini is about 0.35. Application scorecards that include a bureau score usually discriminate considerably better.
- **Approved loans only**, so there is through-the-door selection bias. Reject inference is impossible because rejected applications are not in the data.
- **No payment history.** Default timing is estimated from cumulative totals (A2–A3), not observed.
- **One lender and one period (2007–2011)**, covering the crisis and the recovery. Borrower mix shifted, for example more 60-month loans in late 2011.
- **Small early cohorts** (2007 has 251 loans), so early-quarter statistics are noisy.

**Modelling**
- **PD over a 36-month window**, not the 12-month PD used in Basel IRB or the lifetime PD used in IFRS 9. A conversion is possible but not done here.
- **No macroeconomic variables**, so there is no through-the-cycle, downturn or stressed PD.
- **Calibration drift:** the champion under-predicts the test-period default rate (13.3% vs 14.1%; binomial p = 0.007; Hosmer-Lemeshow p = 0.002). Production use would require recalibration.
- **The champion (XGBoost) is less transparent than the scorecard.** Its PD and the scorecard PD have a rank correlation of 0.90, so individual estimates can differ. The scorecard is used for decisions and reason codes.
- **Random Forest overfits** (train AUC 0.79 vs test 0.67). It is reported but not used.
- **The hyperparameter search is limited** (12–15 random configurations) to keep the runtime around 4 minutes.
- **Some features add nothing out-of-time** (permutation importance ≤ 0), so the champion could be simplified.

**Use**
- **No LGD, EAD, expected loss, capital or IFRS 9 staging is implemented.** These are listed as future extensions only.
- **Fair lending:** `zip_code` and `addr_state` are excluded from the models, but a formal fairness or adverse-impact review is out of scope.
- **Monitoring:** PSI / CSI and the calibration tests should be re-run on every new cohort.

## Reproducibility
- Fixed random seed (42) in every model, search and permutation test.
- Pinned library versions in `requirements.txt`; Python 3.12 (`.python-version`).
- `python run_all.py` regenerates every table, chart, model and report from `data/raw/loan.csv`, and `pytest` runs the test suite.
