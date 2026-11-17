# Phase 9 - Portfolio risk analysis (all 39,717 loans)

## Overview

| metric | value |
|---|---|
| loans | 39,717 |
| exposure_musd | 445.60 |
| default_rate_36m_pct | 13.08 |
| charged_off_rate_resolved_pct | 14.59 |
| average_pd_pct | 12.82 |
| expected_defaults | 5,092.57 |
| actual_defaults_36m | 5,196 |
| high_risk_loans_pct | 20.00 |

## Risk distribution

| risk_category | loans | exposure_musd | defaults | avg_pd_pct | default_rate_pct | avg_score | loan_share_pct | exposure_share_pct |
|---|---|---|---|---|---|---|---|---|
| Low | 15,464 | 157.20 | 1,015 | 7.31 | 6.56 | 562.22 | 38.94 | 35.28 |
| Medium | 16,308 | 185.32 | 2,271 | 13.21 | 13.93 | 541.13 | 41.06 | 41.59 |
| High | 7,945 | 103.08 | 1,910 | 22.77 | 24.04 | 520.27 | 20.00 | 23.13 |

## Default rate by grade
| group | loans | defaults | default_rate_pct | share_pct | avg_pd_pct |
|---|---|---|---|---|---|
| A | 10,085 | 592 | 5.87 | 25.39 | 8.00 |
| B | 12,020 | 1,336 | 11.11 | 30.26 | 11.72 |
| C | 8,098 | 1,249 | 15.42 | 20.39 | 14.56 |
| D | 5,307 | 1,019 | 19.20 | 13.36 | 16.15 |
| E | 2,842 | 628 | 22.10 | 7.16 | 19.10 |
| F | 1,049 | 278 | 26.50 | 2.64 | 21.31 |
| G | 316 | 94 | 29.75 | 0.80 | 23.36 |

## Default rate by annual_inc
| group | loans | defaults | default_rate_pct | share_pct | avg_pd_pct |
|---|---|---|---|---|---|
| <30k | 4,624 | 790 | 17.08 | 11.64 | 16.60 |
| 30-45k | 8,064 | 1,228 | 15.23 | 20.30 | 14.30 |
| 45-60k | 8,797 | 1,217 | 13.83 | 22.15 | 13.31 |
| 60-80k | 7,861 | 941 | 11.97 | 19.79 | 12.06 |
| 80-100k | 4,684 | 500 | 10.67 | 11.79 | 10.62 |
| 100k+ | 5,687 | 520 | 9.14 | 14.32 | 9.78 |

## Default rate by purpose
| group | loans | defaults | default_rate_pct | share_pct | avg_pd_pct |
|---|---|---|---|---|---|
| car | 1,549 | 132 | 8.52 | 3.90 | 8.83 |
| credit_card | 5,130 | 497 | 9.69 | 12.92 | 10.07 |
| debt_consolidation | 18,641 | 2,541 | 13.63 | 46.93 | 13.37 |
| educational | 325 | 53 | 16.31 | 0.82 | 13.45 |
| home_improvement | 2,976 | 318 | 10.69 | 7.49 | 11.18 |
| house | 381 | 54 | 14.17 | 0.96 | 12.94 |
| major_purchase | 2,187 | 210 | 9.60 | 5.51 | 10.49 |
| medical | 693 | 101 | 14.57 | 1.74 | 12.06 |
| moving | 583 | 89 | 15.27 | 1.47 | 12.18 |
| other | 4,096 | 613 | 14.97 | 10.31 | 13.72 |
| small_business | 1,828 | 451 | 24.67 | 4.60 | 23.18 |
| vacation | 381 | 47 | 12.34 | 0.96 | 11.89 |
| wedding | 947 | 90 | 9.50 | 2.38 | 11.16 |

## Default rate by int_rate
| group | loans | defaults | default_rate_pct | share_pct | avg_pd_pct |
|---|---|---|---|---|---|
| <8% | 8,233 | 431 | 5.24 | 20.73 | 7.79 |
| 8-10% | 4,160 | 373 | 8.97 | 10.47 | 10.20 |
| 10-12% | 8,576 | 1,003 | 11.70 | 21.59 | 11.80 |
| 12-14% | 7,469 | 1,087 | 14.55 | 18.81 | 13.90 |
| 14-16% | 5,410 | 927 | 17.13 | 13.62 | 15.57 |
| 16-18% | 3,357 | 719 | 21.42 | 8.45 | 18.01 |
| 18%+ | 2,512 | 656 | 26.11 | 6.32 | 21.06 |

## Default rate by term_months
| group | loans | defaults | default_rate_pct | share_pct | avg_pd_pct |
|---|---|---|---|---|---|
| 36 | 29,096 | 3,227 | 11.09 | 73.26 | 11.07 |
| 60 | 10,621 | 1,969 | 18.54 | 26.74 | 17.63 |

## High-risk concentration by purpose (top 10)

| purpose | loans | portfolio_share_pct | high_risk_share_pct | high_risk_rate_pct | exposure_musd | concentration_ratio |
|---|---|---|---|---|---|---|
| debt_consolidation | 18,641 | 46.93 | 58.17 | 24.79 | 236.65 | 1.24 |
| small_business | 1,828 | 4.60 | 15.24 | 66.25 | 24.80 | 3.31 |
| other | 4,096 | 10.31 | 11.16 | 21.66 | 33.11 | 1.08 |
| credit_card | 5,130 | 12.92 | 4.07 | 6.30 | 60.14 | 0.31 |
| home_improvement | 2,976 | 7.49 | 2.78 | 7.43 | 34.33 | 0.37 |
| medical | 693 | 1.74 | 1.76 | 20.20 | 5.73 | 1.01 |
| major_purchase | 2,187 | 5.51 | 1.42 | 5.17 | 17.84 | 0.26 |
| moving | 583 | 1.47 | 1.42 | 19.38 | 3.88 | 0.97 |
| house | 381 | 0.96 | 1.25 | 25.98 | 4.98 | 1.30 |
| educational | 325 | 0.82 | 0.91 | 22.15 | 2.21 | 1.11 |

## High-risk concentration by state (top 10)

| addr_state | loans | portfolio_share_pct | high_risk_share_pct | high_risk_rate_pct | exposure_musd | concentration_ratio |
|---|---|---|---|---|---|---|
| CA | 7,099 | 17.87 | 16.12 | 18.04 | 80.36 | 0.90 |
| NY | 3,812 | 9.60 | 8.31 | 17.31 | 42.97 | 0.87 |
| FL | 2,866 | 7.22 | 7.38 | 20.45 | 30.87 | 1.02 |
| TX | 2,727 | 6.87 | 6.08 | 17.71 | 31.90 | 0.89 |
| IL | 1,525 | 3.84 | 4.47 | 23.28 | 17.47 | 1.16 |
| NJ | 1,850 | 4.66 | 4.25 | 18.27 | 21.93 | 0.91 |
| OH | 1,223 | 3.08 | 4.05 | 26.33 | 13.30 | 1.32 |
| PA | 1,517 | 3.82 | 4.03 | 21.09 | 16.12 | 1.05 |
| GA | 1,398 | 3.52 | 3.27 | 18.60 | 15.81 | 0.93 |
| VA | 1,407 | 3.54 | 3.21 | 18.12 | 16.25 | 0.91 |

concentration_ratio > 1: the segment holds more than its fair share of High-risk loans.