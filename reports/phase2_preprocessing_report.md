# Phase 2 - Preprocessing report

Clean dataset: **39,717 loans x 41 columns**; 24 model features.

Dropped 54 empty and 9 constant columns, plus repayment (leakage), identifier and free-text columns.

## Target: default within 36 months

| loan_status | loans | main_target_bad |
|---|---|---|
| Charged Off | 5,627 | 5,196 |
| Current | 1,140 | 0 |
| Fully Paid | 32,950 | 0 |

## Default rate by cohort under both definitions

| issue_year | term_months | loans | main_default_rate_pct | spec_default_rate_pct |
|---|---|---|---|---|
| 2007 | 36 | 251 | 17.93 | 17.93 |
| 2008 | 36 | 1,562 | 15.81 | 15.81 |
| 2009 | 36 | 4,716 | 12.60 | 12.60 |
| 2010 | 36 | 8,466 | 9.95 | 9.95 |
| 2010 | 60 | 3,066 | 16.76 | 20.97 |
| 2011 | 36 | 14,101 | 10.63 | 10.63 |
| 2011 | 60 | 7,555 | 19.26 | 27.39 |

## Chronological split

| sample | loans | first_issue | last_issue | default_rate_pct | spec_default_rate_pct |
|---|---|---|---|---|---|
| train | 27,259 | 2007-06 | 2011-06 | 12.60 | 13.70 |
| test | 12,458 | 2011-07 | 2011-12 | 14.14 | 16.68 |

## Remaining missing values in model features

| feature | n_missing | treatment |
|---|---|---|
| emp_length_years | 1,075 | Median in ML models / own WoE bin in scorecard |
| pub_rec_bankruptcies | 697 | Median in ML models / own WoE bin in scorecard |
| revol_util | 50 | Median in ML models / own WoE bin in scorecard |
| mths_since_last_delinq | 25,682 | Median in ML models / own WoE bin in scorecard |

## Outlier caps (learned on training sample)

| feature | lower_cap | upper_cap |
|---|---|---|
| annual_inc | 14,400.00 | 234,000.00 |
| revol_bal | 0.00 | 82,368.44 |
| loan_to_income | 0.00 | 0.51 |
| revol_bal_to_income | 0.00 | 0.85 |