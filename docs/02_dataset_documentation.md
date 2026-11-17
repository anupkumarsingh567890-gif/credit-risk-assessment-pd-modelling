# 2. Dataset documentation

## Source and scope
| Item | Value |
|---|---|
| File | `data/raw/loan.csv`: LendingClub loans issued 2007–2011 (public data; not redistributed in this repository) |
| How to obtain | Search Kaggle for "Lending Club Loan Dataset 2007-2011"; see [data/raw/README.md](../data/raw/README.md) |
| Records | 39,717 loans, one row per loan, no duplicates |
| Columns | 111, of which 54 are completely empty and 9 are constant |
| Issue dates | June 2007 – December 2011 |
| Snapshot | Not stated in the file; inferred as **May 2016** (latest payment and credit-pull month) |
| Loan status | Fully Paid 32,950 · Charged Off 5,627 · Current 1,140 |

Phase 1 checks that the required columns exist and warns if the file is not 39,717 × 111. The complete data dictionary (meaning, when the value is known, missing %, treatment for all 111 columns) is generated as `reports/phase1/data_dictionary.csv`. Example values are masked for identifiers and free text. The data quality report is `reports/phase1/data_quality_report.md`.

## Target variable
**`default` = 1 if the loan was charged off AND the delinquency that led to charge-off started within its first 36 months on book; otherwise 0.** The construction is explained in [03_methodology.md](03_methodology.md#2-target-definition).

| Sample | Issue dates | Loans | `default` rate | Original definition* |
|---|---|---|---|---|
| Train | Jun 2007 – Jun 2011 | 27,259 | 12.60% | 13.70% |
| Test (out-of-time) | Jul – Dec 2011 | 12,458 | 14.14% | 16.68% |
| All | | 39,717 | 13.08% | 14.59% |

\* Charged Off vs Fully Paid, with Current loans excluded. This is kept as `default_spec` for the sensitivity check in Phase 10.

Of the 5,627 charged-off loans, 5,196 are bad within the 36-month window. The other 431 are 60-month loans whose arrears began after month 36.

## Data dictionary – model features (24, all known at origination)

| Feature | Meaning | Type | Source |
|---|---|---|---|
| loan_amnt | Amount requested (USD) | numeric | Application |
| term_months | 36 or 60 monthly payments | numeric | Application |
| annual_inc | Self-reported annual income, capped at the train 1st / 99th percentiles | numeric | Application |
| dti | Monthly debt payments / monthly income × 100 (excluding mortgage and this loan) | numeric | Application + bureau |
| emp_length_years | Years employed, 0 (< 1) to 10 (10+) | numeric | Application |
| delinq_2yrs | 30+ days-past-due incidents in the last 2 years | numeric | Bureau |
| inq_last_6mths | Credit inquiries in the last 6 months | numeric | Bureau |
| open_acc | Open credit lines | numeric | Bureau |
| total_acc | Total credit lines ever | numeric | Bureau |
| pub_rec | Derogatory public records | numeric | Bureau |
| pub_rec_bankruptcies | Public-record bankruptcies | numeric | Bureau |
| revol_bal | Revolving balance (USD), capped at the train 99th percentile | numeric | Bureau |
| revol_util | Revolving balance / revolving limit (%) | numeric | Bureau |
| mths_since_last_delinq | Months since last delinquency (blank = none on file) | numeric | Bureau |
| credit_history_years | Years from earliest credit line to issue date | engineered | Bureau |
| loan_to_income | loan_amnt / annual_inc | engineered | Application |
| revol_bal_to_income | revol_bal / annual_inc | engineered | Bureau + application |
| emp_length_missing | 1 if employment length not given | flag | Application |
| emp_title_missing | 1 if job title not given | flag | Application |
| has_delinq_history | 1 if any delinquency on file | flag | Bureau |
| has_public_record | 1 if any public record on file | flag | Bureau |
| home_ownership | RENT, MORTGAGE, OWN, OTHER | categorical | Application |
| verification_status | Income Verified / Source Verified / Not Verified | categorical | Origination |
| purpose | Loan purpose (13 categories) | categorical | Application |

**Kept for analysis, never used as model inputs:**
- `grade`, `sub_grade`, `int_rate`, `installment`: LendingClub's own risk output, used as a benchmark;
- `issue_d`: used for the split and the vintages;
- `addr_state`: used for concentration analysis only;
- outcome fields: `loan_status`, `default`, `default_spec`, `default_onset_month`, `months_to_last_payment`, `months_observed`.

**Used only to build the label, then dropped:** `total_rec_prncp`, `last_pymnt_d`, `funded_amnt`, together with `installment` and `int_rate`, which feed the amortisation schedule.

**Removed:**
- 54 empty and 9 constant columns;
- 14 repayment columns (leakage);
- identifiers (`id` is kept only for tracking);
- free text (`desc`, `title`, `emp_title`);
- `zip_code` (fair-lending proxy risk);
- `funded_amnt` / `funded_amnt_inv` (set at issuance).

`check_no_leakage()` in `src/preprocessing.py` fails the pipeline if any outcome, repayment or pricing column enters the feature list. It is covered by tests.

## Data quality findings and treatment

| Issue | Treatment |
|---|---|
| `int_rate`, `revol_util` stored as text with % | Converted to float |
| `term`, `emp_length` stored as text | Converted to months / years |
| `earliest_cr_line` two-digit years (90 loans parsed into 2046–2068) | Dates after the issue date shifted back 100 years |
| `annual_inc` up to $6M (14 loans above $1M) | Capped at the training 1st / 99th percentile |
| `mths_since_last_delinq` 65% blank (blank = no delinquency; these loans default less) | `has_delinq_history` flag; own WoE bin |
| `emp_length` 2.7% blank (default 22% vs 14%) | `emp_length_missing` flag; median in ML models |
| `revol_util` 0.13% blank | Pooled into the riskiest utilisation bin (too few loans for a stable WoE) |
| `home_ownership` NONE (3) / `purpose` renewable_energy (103) | Merged into OTHER / other |
| Purposes with < 2% of loans | Pooled into one "RARE" WoE bin in the scorecard |
| No duplicates | – |

## What the dataset does not contain
- **No FICO / bureau score** in this extract.
- **No monthly payment history.** Default timing has to be estimated (see the methodology).
- **No rejected applications.** This creates through-the-door selection bias.
- **No macroeconomic variables and no collateral** (the loans are unsecured). Recovery timing is also missing, so a discounted LGD is not possible.
