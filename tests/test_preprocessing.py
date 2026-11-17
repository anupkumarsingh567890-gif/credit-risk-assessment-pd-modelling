"""Cleaning, feature engineering, missing values, splitting and leakage safeguards."""
import numpy as np
import pandas as pd
import pytest

from src.config import ALL_FEATURES, LC_PRICING_COLS, TEST_START_DATE
from src.data.load import parse_lc_date
from src.preprocessing import (
    LABEL_COLS, POST_ORIGINATION_COLS, _emp_length_to_years, _percent_to_float, apply_caps,
    check_no_leakage, engineer_features, finalize_columns, fit_caps, split_by_date,
)


def test_percent_strings_become_numbers():
    out = _percent_to_float(pd.Series(["10.65%", " 7.5% ", None]))
    assert out.iloc[0] == pytest.approx(10.65)
    assert out.iloc[1] == pytest.approx(7.5)
    assert np.isnan(out.iloc[2])


def test_employment_length_parsing_keeps_missing():
    out = _emp_length_to_years(pd.Series(["< 1 year", "1 year", "5 years", "10+ years", None]))
    assert out.iloc[:4].tolist() == [0, 1, 5, 10]
    assert np.isnan(out.iloc[4])


def test_two_digit_years_are_fixed():
    issue = pd.Series(pd.to_datetime(["2011-12-01", "2011-12-01"]))
    out = parse_lc_date(pd.Series(["Dec-68", "Jan-05"]), reference=issue)
    assert out.dt.year.tolist() == [1968, 2005]


def test_missing_value_flags():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        "earliest_cr_line": pd.to_datetime(["2000-01-01", "1990-01-01"]),
        "loan_amnt": [10_000, 5_000], "annual_inc": [50_000, 100_000], "revol_bal": [5_000, 0],
        "emp_length_years": [np.nan, 3], "emp_title": [None, "Teacher"],
        "mths_since_last_delinq": [np.nan, 12], "mths_since_last_record": [np.nan, np.nan],
        "pub_rec": [0, 1],
    })
    out = engineer_features(df)
    assert out["emp_length_missing"].tolist() == [1, 0]
    assert out["emp_title_missing"].tolist() == [1, 0]
    assert out["has_delinq_history"].tolist() == [0, 1]
    assert out["has_public_record"].tolist() == [0, 1]
    assert out["credit_history_years"].tolist() == [10, 20]
    assert out["loan_to_income"].tolist() == [0.2, 0.05]


def test_chronological_split():
    df = pd.DataFrame({"id": range(4), "issue_d": pd.to_datetime(
        ["2011-08-01", "2010-01-01", "2011-06-01", "2011-07-01"])})
    out = split_by_date(df)
    assert out.loc[out["sample"] == "train", "issue_d"].max() < TEST_START_DATE
    assert out.loc[out["sample"] == "test", "issue_d"].min() >= TEST_START_DATE
    assert out["issue_d"].is_monotonic_increasing  # needed by TimeSeriesSplit


def test_caps_are_learned_on_train_only():
    train = pd.DataFrame({c: np.arange(1, 101, dtype=float) for c in
                          ["annual_inc", "revol_bal", "loan_to_income", "revol_bal_to_income"]})
    caps = fit_caps(train)
    test = pd.DataFrame({c: [1e9] for c in train.columns})
    capped = apply_caps(test, caps)
    assert caps["annual_inc"][1] == pytest.approx(np.quantile(train["annual_inc"], 0.99))
    assert capped["annual_inc"].iloc[0] == caps["annual_inc"][1]


def test_model_features_contain_no_leakage():
    check_no_leakage(ALL_FEATURES)  # must not raise
    forbidden = set(POST_ORIGINATION_COLS) | set(LABEL_COLS) | set(LC_PRICING_COLS)
    assert not forbidden & set(ALL_FEATURES)


@pytest.mark.parametrize("bad", ["total_pymnt", "recoveries", "default", "default_onset_month", "grade", "int_rate"])
def test_leakage_guard_rejects_forbidden_columns(bad):
    with pytest.raises(ValueError, match="Leakage"):
        check_no_leakage(ALL_FEATURES + [bad])


def test_finalize_drops_repayment_columns():
    df = pd.DataFrame({c: [1] for c in POST_ORIGINATION_COLS + ["loan_amnt"]})
    out = finalize_columns(df)
    assert list(out.columns) == ["loan_amnt"]


def test_real_data_split_and_columns(clean_data):
    train = clean_data[clean_data["sample"] == "train"]
    test = clean_data[clean_data["sample"] == "test"]
    assert train["issue_d"].max() < test["issue_d"].min()
    assert not set(POST_ORIGINATION_COLS) & set(clean_data.columns)
    assert set(ALL_FEATURES) <= set(clean_data.columns)
    # only the documented features may still contain missing values
    still_missing = set(clean_data[ALL_FEATURES].columns[clean_data[ALL_FEATURES].isna().any()])
    assert still_missing <= {"emp_length_years", "pub_rec_bankruptcies", "revol_util", "mths_since_last_delinq"}
