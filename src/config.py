"""
Central project configuration.

Every path, seed and business rule used by more than one script lives here,
so a change (for example a different test period) is made in one place only.
"""
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # original file, never modified
INTERIM_DIR = DATA_DIR / "interim"      # unmodified Parquet copy
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, model-ready data

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PHASE1_DIR = REPORTS_DIR / "phase1"     # kept for the Phase 1 script
TABLES_DIR = REPORTS_DIR / "tables"

RAW_FILE = RAW_DIR / "loan.csv"
RAW_PARQUET = INTERIM_DIR / "loan_raw.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "loans_clean.parquet"
CLEAN_CSV = PROCESSED_DIR / "loans_clean.csv"
TEST_PREDICTIONS = PROCESSED_DIR / "test_predictions.parquet"
SCORED_PORTFOLIO = PROCESSED_DIR / "portfolio_scored.parquet"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
STATUS_COL = "loan_status"
TARGET_COL = "default"            # main target: default within the window
TARGET_SPEC_COL = "default_spec"  # original definition, kept for sensitivity

# Original definition: Charged Off = 1, Fully Paid = 0, Current excluded
TARGET_MAP = {"Charged Off": 1, "Fully Paid": 0}
EXCLUDED_STATUSES = ["Current"]

# Main definition (decided after Phase 1): a loan is "bad" if it was charged
# off AND stopped paying within its first 36 months. Every loan in the file
# has at least 53 months of history, so this outcome is fully observed for
# every cohort and 'Current' loans can be kept as goods.
PERFORMANCE_WINDOW_MONTHS = 36

# --------------------------------------------------------------------------- #
# Dates and sample split
# --------------------------------------------------------------------------- #
DATE_FORMAT = "%b-%y"   # LendingClub stores dates as 'Mon-YY', e.g. 'Dec-11'
DATE_COLS = ["issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]

# Inferred in Phase 1 from the latest payment / credit-pull month
SNAPSHOT_DATE = pd.Timestamp("2016-05-01")

# Chronological (out-of-time) split: train on older loans, test on newer ones
TEST_START_DATE = pd.Timestamp("2011-07-01")

# --------------------------------------------------------------------------- #
# Model features (origination-time only)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "loan_amnt",
    "term_months",
    "annual_inc",
    "dti",
    "emp_length_years",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "revol_bal",
    "revol_util",
    "mths_since_last_delinq",
    "credit_history_years",
    "loan_to_income",
    "revol_bal_to_income",
]
FLAG_FEATURES = [
    "emp_length_missing",
    "emp_title_missing",
    "has_delinq_history",
    "has_public_record",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]
ALL_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES

# LendingClub's own risk outputs: excluded from models, used as a benchmark
LC_PRICING_COLS = ["grade", "sub_grade", "int_rate", "installment"]

# Outlier caps (percentiles learned on the training sample only)
CAP_COLUMNS = {
    "annual_inc": (0.01, 0.99),
    "revol_bal": (0.0, 0.99),
    "loan_to_income": (0.0, 0.99),
    "revol_bal_to_income": (0.0, 0.99),
}

# --------------------------------------------------------------------------- #
# Scorecard scaling
# --------------------------------------------------------------------------- #
# A score of 600 means odds of 50 goods : 1 bad, and every 20 points
# doubles the odds (Points to Double the Odds).
BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20

# Minimum Information Value for a feature to enter the scorecard
MIN_IV = 0.02
# Maximum absolute correlation allowed between two WoE features
MAX_WOE_CORR = 0.6

# Risk bands: share of training applicants in each band (worst first)
RISK_BAND_SHARES = {"High": 0.20, "Medium": 0.40, "Low": 0.40}

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
# The champion is the model with the best cross-validated AUC, unless Logistic
# Regression is within 2 Gini points (0.01 AUC): then the interpretable model
# wins, because a tiny uplift does not justify a black-box model in credit.
INTERPRETABILITY_MARGIN_AUC = 0.01

# Saved model files (one place, used by scripts and the app)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}

# --------------------------------------------------------------------------- #
# Validation rules (Phase 10 traffic lights)
# --------------------------------------------------------------------------- #
PSI_STABLE, PSI_SHIFT = 0.10, 0.25   # < 0.10 stable, 0.10-0.25 monitor, > 0.25 shift
MAX_OVERFIT_GAP_AUC = 0.05           # train AUC - test AUC
MAX_CV_AUC_STD = 0.03                # spread of AUC across time-series folds
MIN_TEST_AUC = 0.65                  # minimum acceptable out-of-time AUC
CALIBRATION_ALPHA = 0.05             # significance level for calibration tests
