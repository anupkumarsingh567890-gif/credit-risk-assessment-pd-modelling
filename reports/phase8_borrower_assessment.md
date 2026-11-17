# Phase 8 - Borrower assessment

Champion PD model: **XGBoost**. Approval cut-off score: 542.

## Example application

| input | value |
|---|---|
| loan_amnt | 12,000 |
| term_months | 36 |
| annual_inc | 60,000 |
| dti | 14.00 |
| emp_length_years | 5 |
| emp_title_provided | 1 |
| home_ownership | RENT |
| verification_status | Not Verified |
| purpose | debt_consolidation |
| delinq_2yrs | 0 |
| inq_last_6mths | 1 |
| open_acc | 9 |
| total_acc | 20 |
| pub_rec | 0 |
| pub_rec_bankruptcies | 0 |
| revol_bal | 10,000 |
| revol_util | 50.00 |
| mths_since_last_delinq | None |
| credit_history_years | 12.00 |

### Result

| credit_score | pd_pct | risk_category | decision | main_reasons | warnings | creditworthiness | model |
|---|---|---|---|---|---|---|---|
| 543 | 10.33 | Medium | Approve | purpose: 16 points below best; revol_util: 13 points below best; annual_inc: 11 points below best | [] | Acceptable - average risk, terms may need adjusting | XGBoost |

### Points by characteristic

| characteristic | points |
|---|---|
| revol_util | 77 |
| purpose | 76 |
| loan_to_income | 78 |
| term_months | 81 |
| inq_last_6mths | 75 |
| annual_inc | 77 |
| has_public_record | 79 |

## Typical test-sample borrower in each risk band

| id | credit_score | pd_pct | risk_category | decision | main_reasons | grade | actual_default |
|---|---|---|---|---|---|---|---|
| 1040029 | 560 | 7.26 | Low | Approve | revol_util: 26 points below best; purpose: 5 points below best; loan_to_income: 4 points below best | B | 0 |
| 1030552 | 541 | 11.30 | Medium | Refer for manual review | purpose: 16 points below best; annual_inc: 15 points below best; revol_util: 13 points below best | A | 0 |
| 851780 | 522 | 17.26 | High | Decline | revol_util: 26 points below best; annual_inc: 18 points below best; purpose: 16 points below best | D | 0 |