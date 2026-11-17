# Phase 5 - Risk classification report

Band cut-offs (from training scores): High < 531 <= Medium < 551 <= Low.

## Risk bands

| sample | band | loans | defaults | avg_score | min_score | max_score | avg_pd_pct | share_pct | default_rate_pct |
|---|---|---|---|---|---|---|---|---|---|
| train | Low | 10,995 | 710 | 562.35 | 551 | 595 | 7.11 | 40.34 | 6.46 |
| train | Medium | 11,104 | 1,533 | 541.22 | 531 | 550 | 13.46 | 40.74 | 13.81 |
| train | High | 5,160 | 1,191 | 520.33 | 468 | 530 | 24.43 | 18.93 | 23.08 |

| sample | band | loans | defaults | avg_score | min_score | max_score | avg_pd_pct | share_pct | default_rate_pct |
|---|---|---|---|---|---|---|---|---|---|
| test | Low | 4,469 | 305 | 561.90 | 551 | 595 | 7.21 | 35.87 | 6.82 |
| test | Medium | 5,204 | 738 | 540.95 | 531 | 550 | 13.57 | 41.77 | 14.18 |
| test | High | 2,785 | 719 | 520.18 | 466 | 530 | 24.53 | 22.36 | 25.82 |

## Approval cut-off analysis (test sample)
KS-optimal cut-off chosen on the training sample: **score 542**.

| cutoff_score | approval_rate_pct | approved_bad_rate_pct | rejected_bad_rate_pct | bads_rejected_pct | goods_rejected_pct | ks_gap |
|---|---|---|---|---|---|---|
| 468.00 | 99.99 | 14.14 | 0.00 | 0.00 | 0.01 | -0.01 |
| 517.00 | 94.03 | 12.94 | 33.06 | 13.96 | 4.66 | 9.31 |
| 523.00 | 88.75 | 12.15 | 29.89 | 23.78 | 9.19 | 14.59 |
| 528.00 | 82.06 | 11.32 | 27.07 | 34.34 | 15.24 | 19.10 |
| 531.00 | 77.64 | 10.78 | 25.82 | 40.81 | 19.32 | 21.49 |
| 534.00 | 72.26 | 10.14 | 24.57 | 48.18 | 24.37 | 23.81 |
| 537.00 | 66.64 | 9.60 | 23.22 | 54.77 | 29.83 | 24.93 |
| 540.00 | 60.29 | 9.09 | 21.81 | 61.24 | 36.16 | 25.07 |
| 542.00 | 55.95 | 8.75 | 20.99 | 65.38 | 40.54 | 24.84 |
| 544.00 | 51.49 | 8.45 | 20.19 | 69.24 | 45.09 | 24.15 |
| 546.00 | 47.03 | 8.06 | 19.55 | 73.21 | 49.64 | 23.58 |
| 549.00 | 40.68 | 7.40 | 18.77 | 78.72 | 56.12 | 22.59 |
| 551.00 | 35.87 | 6.82 | 18.24 | 82.69 | 61.07 | 21.62 |
| 553.00 | 31.19 | 6.54 | 17.59 | 85.58 | 66.04 | 19.54 |
| 555.00 | 27.10 | 6.22 | 17.09 | 88.08 | 70.40 | 17.68 |
| 558.00 | 22.11 | 5.81 | 16.51 | 90.92 | 75.75 | 15.17 |
| 561.00 | 16.99 | 5.29 | 15.96 | 93.64 | 81.25 | 12.39 |
| 564.00 | 13.04 | 4.68 | 15.56 | 95.69 | 85.52 | 10.17 |
| 568.00 | 8.74 | 3.76 | 15.14 | 97.67 | 90.20 | 7.47 |

At the KS cut-off on test: approve 56.0% of applicants, approved bad rate 8.75% vs 14.14% if everyone is approved; 65.4% of defaulters rejected at the cost of 40.5% of good borrowers.