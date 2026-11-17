"""Target construction: default within 36 months of origination."""
import numpy as np
import pandas as pd
import pytest

from src.config import PERFORMANCE_WINDOW_MONTHS, TARGET_COL, TARGET_SPEC_COL
from src.preprocessing import add_targets, default_onset_month, instalments_covered
from tests.conftest import annuity_instalment, make_loan


def build(rows):
    return add_targets(pd.DataFrame(rows))


def test_fully_paid_and_current_are_good():
    out = build([
        make_loan("Fully Paid", 36, 36, 36),
        make_loan("Current", 60, 60, 60, issue="2011-06-01"),
    ])
    assert out[TARGET_COL].tolist() == [0, 0]
    assert out[TARGET_SPEC_COL].iloc[0] == 0
    assert np.isnan(out[TARGET_SPEC_COL].iloc[1])  # Current excluded in original definition


def test_charged_off_never_paid_defaults_in_month_one():
    out = build([make_loan("Charged Off", 36, None, 0)])
    assert out["default_onset_month"].iloc[0] == 1
    assert out[TARGET_COL].iloc[0] == 1


def test_late_60_month_default_is_outside_window():
    # paid 40 instalments on time, then stopped: first missed instalment = 41
    out = build([make_loan("Charged Off", 60, 40, 40)])
    assert out["default_onset_month"].iloc[0] == 41
    assert out[TARGET_COL].iloc[0] == 0


def test_arrears_are_dated_from_first_unpaid_instalment():
    # last payment in month 40, but principal repaid covers only 30 instalments:
    # the borrower was in arrears from instalment 31 -> inside the window
    out = build([make_loan("Charged Off", 60, 40, 30)])
    assert out["default_onset_month"].iloc[0] == 31
    assert out[TARGET_COL].iloc[0] == 1


def test_prepayment_does_not_hide_an_early_stop():
    # prepaid principal worth 45 instalments but stopped paying after month 20
    out = build([make_loan("Charged Off", 60, 20, 45)])
    assert out["default_onset_month"].iloc[0] == 21
    assert out[TARGET_COL].iloc[0] == 1


@pytest.mark.parametrize("paid, expected", [(35, 1), (36, 0)])
def test_window_boundary(paid, expected):
    # first missed instalment = paid + 1; inside the window only if <= 36
    out = build([make_loan("Charged Off", 60, paid, paid)])
    assert out["default_onset_month"].iloc[0] == paid + 1
    assert out[TARGET_COL].iloc[0] == expected


def test_36_month_charge_off_always_inside_its_term():
    # in arrears, last payment after maturity: balance left at month 36 = missed payment
    out = build([make_loan("Charged Off", 36, 40, 35.9)])
    assert out["default_onset_month"].iloc[0] <= 36
    assert out[TARGET_COL].iloc[0] == 1


def test_same_rule_for_both_terms():
    # identical behaviour (stop after 20 on-time payments) -> same label for 36m and 60m
    out = build([make_loan("Charged Off", 36, 20, 20), make_loan("Charged Off", 60, 20, 20)])
    assert out[TARGET_COL].tolist() == [1, 1]
    assert out["default_onset_month"].tolist() == [21, 21]


def test_insufficient_observation_raises():
    # issued 2014: fewer than 36 months before the May-2016 snapshot
    with pytest.raises(ValueError, match="not fully observed"):
        build([make_loan("Fully Paid", 36, 10, 10, issue="2014-01-01")])


def test_instalments_covered_limits():
    p, rate, n = 10_000, 12.0, 36
    a = annuity_instalment(p, rate, n)
    k = instalments_covered(pd.Series([0.0, p]), pd.Series([p, p]), pd.Series([a, a]), pd.Series([rate, rate]))
    assert k.iloc[0] == pytest.approx(0, abs=1e-9)
    assert k.iloc[1] == pytest.approx(n, abs=1e-6)


def test_onset_never_exceeds_term():
    df = pd.DataFrame([make_loan("Charged Off", 60, 70, 59.5)])
    assert default_onset_month(df).iloc[0] <= 60


def test_real_data_target(clean_data):
    """Checks on the actual pipeline output."""
    assert set(clean_data[TARGET_COL].unique()) <= {0, 1}
    assert clean_data.loc[clean_data["loan_status"] != "Charged Off", TARGET_COL].sum() == 0
    assert (clean_data["months_observed"] >= PERFORMANCE_WINDOW_MONTHS).all()
    bad = clean_data[clean_data[TARGET_COL] == 1]
    assert (bad["default_onset_month"] <= PERFORMANCE_WINDOW_MONTHS).all()
    assert 0.05 < clean_data[TARGET_COL].mean() < 0.25
