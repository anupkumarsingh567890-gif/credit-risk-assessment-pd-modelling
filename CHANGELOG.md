# Changelog

All notable changes to this project. Results are from the reference run (`python run_all.py`, Python 3.12, pinned versions, seed 42).

## [1.1.0] – 2026-11-17 – Audit, correctness fixes, tests and publication readiness

### Fixed – methodology
- **Target construction (`src/preprocessing.py`).** The default onset was dated from the last payment date, which lags the start of arrears (median gap 2 months, for 95.5% of charge-offs). 36-month loans were handled by a special-case rule, and there was an off-by-one at the window boundary (a last payment in month 36 was counted as default inside the window). The new `default_onset_month()` takes the earliest of three bounds: the first instalment not covered by principal repaid (closed-form annuity schedule), the month after the last payment, and the term. One uniform rule now applies to both terms. Bads: 5,149 → 5,196.
- **Feature-selection leakage in cross-validation (`src/models.py`).** The logistic regression's IV / correlation / p-value selection had been run once on the full training sample, so it had seen every CV validation fold. `WoEEncoder` now re-runs binning and selection inside each fold. Because the champion is chosen on CV AUC, the old setup biased that choice towards logistic regression (CV AUC 0.655 → 0.651).
- **Hosmer-Lemeshow degrees of freedom (`src/validation.py`).** The test used g − 2 on the independent test sample; out-of-sample it should be g. A binomial calibration-in-the-large test was added alongside it. The old check ("Pass" if the gap was under 1 percentage point) had no statistical basis.
- **Unstable WoE bins (`src/woe.py`).**
  - A 44-loan `revol_util` missing bin carried a 47-point penalty; missing bins under 5% are now pooled with the closest bin.
  - Purposes under 2% of loans are now pooled into "RARE".
  - Unseen values were scored as neutral (WoE 0); they are now scored as the riskiest bin, the conservative choice.
  - Smoothed distributions are re-normalised to sum to 1.
- **Calibration decision (`src/models.py`).** Raw and Platt-calibrated Brier scores were compared on the same out-of-fold predictions used to fit the calibrator. The comparison is now cross-fitted.
- **CV-stability status** in Phase 10 was hard-coded as "Pass". All traffic-light rules are now named constants in `config.py`.

### Fixed – code
- Text-to-number parsers returned pandas' nullable `Float64` (pd.NA); they now return float64.
- Month arithmetic failed on columns with only missing dates.
- The Phase 1 target helper wrote the original definition into a column named `default`, which clashed with the modelling target; it now writes `default_spec`.
- Portfolio overview numbers were printed unformatted.

### Changed
- **Champion PD model: Logistic Regression → XGBoost.** With leakage-free CV, XGBoost's lead (0.015 AUC) exceeds the 0.01 interpretability margin that was fixed in advance. The scorecard remains the decisioning tool.
- Removed duplicated code:
  - `months_between`, `md_table` and the sub-grade benchmark rank now each exist once;
  - model-file maps are centralised in `config.MODEL_FILES`;
  - portfolio band and concentration summaries live in a new `src/portfolio.py`, shared by Phase 9 and the dashboard.
- The vintage curves use the estimated default onset.
- **Dashboard** (all six pages kept):
  - consistent theme (`.streamlit/config.toml`), KPI cards and page headers, and a sidebar with model context and the disclaimer;
  - Altair charts with tooltips and correct average lines;
  - risk-band explanations and data-driven interpretation text;
  - input validation with clear error messages and warnings for values outside the training range;
  - scorecard PD and champion PD shown side by side, with reason codes;
  - validation flags surfaced.
- The data dictionary masks example values for identifiers and free text.
- The notebook is committed without outputs.

### Added
- `tests/`: a pytest suite of 68 tests covering the target (including boundary, arrears, prepayment and maturity cases), preprocessing and missing values, leakage safeguards, WoE/IV hand calculations, PDO scaling and points, PD ranges, risk bands and cut-offs, metrics (Gini, KS, Brier, confusion matrix), PSI/CSI, Hosmer-Lemeshow, the binomial test, threshold sensitivity, borrower-input validation and the trained-model outputs.
- `check_no_leakage()`, a guard called in Phase 6 and in the tests.
- `validate_raw()`, which checks required columns and warns if the raw file is not 39,717 × 111.
- `validate_application()` for borrower inputs.
- `require()`, which gives informative errors when an earlier phase has not been run.
- `LICENSE` (MIT), `.python-version`, `pyproject.toml` (pytest configuration), `CHANGELOG.md`, `data/raw/README.md` (how to obtain the data), and real dashboard screenshots in `assets/images/`.
- Expanded `.gitignore` (data, models, secrets, caches, logs).
- `pytest` and `altair` pinned in `requirements.txt`.

### Results: before → after
| Measure (out-of-time test unless stated) | 1.0.0 | 1.1.0 |
|---|---|---|
| Bad loans (36-month window) | 5,149 | 5,196 |
| Test default rate | 14.04% | 14.14% |
| Scorecard characteristics | 8 | 7 (`revol_bal_to_income` now p = 0.053) |
| Scorecard Gini | 0.342 | 0.349 |
| LendingClub sub-grade Gini (benchmark) | 0.309 | 0.311 |
| Logistic regression CV AUC | 0.655 (selection leakage) | 0.651 |
| XGBoost CV AUC | 0.664 | 0.665 |
| Champion | Logistic Regression | **XGBoost** (Gini 0.361) |
| Risk bands Low / Medium / High default rate | 7.1% / 14.0% / 25.4% | 6.8% / 14.2% / 25.8% |
| Maximum CSI | 0.136 (purpose, unpooled small categories) | 0.044 (term) |
| Hosmer-Lemeshow (test) | p = 0.031 (LR, wrong d.f.) | p = 0.002 (XGBoost, 10 d.f.): Monitor |
| Calibration-in-the-large | "Pass" (1 pp rule) | z = 2.71, p = 0.007: Monitor |

The calibration findings are reported, not tuned away: the test period was riskier than the training period, and the champion under-predicts PD by about 0.8 percentage points.

## [1.0.0] – 2026-11-17 – Initial end-to-end project
- Phases 1–10: data understanding, preprocessing, EDA, WoE/IV scorecard, risk bands, four PD models, evaluation, borrower assessment, portfolio analysis, validation.
- Six-page Streamlit app, documentation, VS Code launch configuration.
