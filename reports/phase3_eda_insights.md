# Phase 3 - EDA insights (training sample, 2007-06 to 2011-06)

Training default rate (36-month window): **12.60%** on 27,259 loans.

## Default-rate drivers (lowest vs highest group)

- **grade:** default rate ranges from 5.3% (A) to 23.9% (F), 4.5x higher.
- **sub_grade:** default rate ranges from 2.9% (A1) to 21.7% (E2), 7.5x higher.
- **int_rate:** default rate ranges from 4.6% (<8%) to 24.5% (18%+), 5.3x higher.
- **annual_inc:** default rate ranges from 9.2% (100k+) to 16.7% (<30k), 1.8x higher.
- **loan_amnt:** default rate ranges from 11.2% (10-15k) to 19.3% (25k+), 1.7x higher.
- **dti:** default rate ranges from 8.5% (25+) to 14.4% (20-25), 1.7x higher.
- **emp_length_years:** default rate ranges from 11.6% (6) to 13.5% (7), 1.2x higher.
- **home_ownership:** default rate ranges from 12.0% (MORTGAGE) to 13.1% (OWN), 1.1x higher.
- **purpose:** default rate ranges from 7.8% (car) to 24.2% (small_business), 3.1x higher.
- **credit_history_years:** default rate ranges from 11.1% (30y+) to 16.1% (<5y), 1.5x higher.
- **revol_util:** default rate ranges from 8.9% (0-20%) to 17.2% (80%+), 1.9x higher.
- **inq_last_6mths:** default rate ranges from 10.4% (0) to 18.8% (3), 1.8x higher.
- **term_months:** default rate ranges from 11.1% (36) to 17.5% (60), 1.6x higher.
- **verification_status:** default rate ranges from 11.8% (Not Verified) to 13.8% (Verified), 1.2x higher.
- **loan_to_income:** default rate ranges from 10.4% (<10%) to 18.1% (30%+), 1.7x higher.

## Strongest linear relationships with default

- int_rate: r = +0.164
- revol_util: r = +0.089
- loan_to_income: r = +0.084
- term_months: r = +0.083
- inq_last_6mths: r = +0.075
- annual_inc: r = -0.057

## Highly correlated feature pairs (|r| > 0.6) - redundancy to manage

- pub_rec & pub_rec_bankruptcies: 0.83
- revol_bal & revol_bal_to_income: 0.78
- open_acc & total_acc: 0.69

Charts: `reports/figures/eda/`. Tables: `reports/tables/eda_*.csv`.