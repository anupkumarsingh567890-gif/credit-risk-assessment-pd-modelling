"""PSI / CSI, Hosmer-Lemeshow, calibration-in-the-large, threshold sensitivity."""
import numpy as np
import pandas as pd
import pytest

from src.validation import (
    auc_by_period, calibration_in_the_large, hosmer_lemeshow, psi, psi_categorical, psi_status,
    threshold_sensitivity,
)


def test_psi_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5000)
    value, table = psi(x, x)
    assert value == pytest.approx(0, abs=1e-12)
    assert table["expected_pct"].sum() == pytest.approx(100)


def test_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    value, _ = psi(rng.normal(size=5000), rng.normal(1, 1, size=5000))
    assert value > 0.25
    assert psi_status(value) == "Significant shift"


def test_psi_hand_example():
    # two buckets: 50/50 -> 30/70 gives PSI = 0.2*ln(70/50) - 0.2*ln(30/50)
    expected = np.r_[np.zeros(50), np.ones(50)]
    actual = np.r_[np.zeros(30), np.ones(70)]
    assert psi_categorical(pd.Series(expected), pd.Series(actual)) == pytest.approx(
        0.2 * np.log(0.7 / 0.5) - 0.2 * np.log(0.3 / 0.5))


def test_psi_status_bands():
    assert psi_status(0.05) == "Stable"
    assert psi_status(0.15) == "Monitor"
    assert psi_status(0.30) == "Significant shift"


def test_hosmer_lemeshow_calibrated_vs_miscalibrated():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.02, 0.4, 20_000)
    y_good = (rng.random(p.size) < p).astype(int)
    y_bad = (rng.random(p.size) < np.minimum(p * 1.5, 1)).astype(int)
    _, p_good, table = hosmer_lemeshow(y_good, p)
    _, p_bad, _ = hosmer_lemeshow(y_bad, p)
    assert p_good > 0.01
    assert p_bad < 0.001
    assert len(table) == 10


def test_hosmer_lemeshow_degrees_of_freedom():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.3, 5000)
    y = (rng.random(p.size) < p).astype(int)
    hl_out, p_out, _ = hosmer_lemeshow(y, p, out_of_sample=True)
    hl_in, p_in, _ = hosmer_lemeshow(y, p, out_of_sample=False)
    assert hl_out == pytest.approx(hl_in)
    assert p_out > p_in  # more degrees of freedom -> larger p-value for the same statistic


def test_calibration_in_the_large():
    p = np.full(1000, 0.1)
    on_target = calibration_in_the_large(np.r_[np.ones(100), np.zeros(900)], p)
    assert on_target["z"] == pytest.approx(0)
    too_low = calibration_in_the_large(np.r_[np.ones(200), np.zeros(800)], p)
    assert too_low["z"] > 0 and too_low["p_value"] < 0.001


def test_threshold_sensitivity_is_monotonic():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = threshold_sensitivity(y, p, [0.1, 0.3, 0.5, 0.7])
    assert t["flagged_pct"].is_monotonic_decreasing
    assert t["recall"].is_monotonic_decreasing


def test_auc_by_period_skips_single_class_periods():
    df = pd.DataFrame({"q": ["A"] * 4 + ["B"] * 2, "y": [0, 1, 0, 1, 0, 0],
                       "p": [0.1, 0.9, 0.2, 0.8, 0.5, 0.5]})
    out = auc_by_period(df, "q", "y", ["p"])
    assert out["period"].tolist() == ["A"]
    assert out["auc_p"].iloc[0] == 1.0
