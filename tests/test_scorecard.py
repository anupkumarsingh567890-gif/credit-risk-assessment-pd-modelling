"""Scorecard scaling (PDO), points, PD conversion, risk bands and cut-offs."""
import numpy as np
import pandas as pd
import pytest

from src import scorecard as sc
from src.config import BASE_ODDS, BASE_SCORE, PDO
from src.risk_bands import assign_band, band_summary, cutoff_table, fit_band_cutoffs


def test_scaling_constants():
    assert sc.FACTOR == pytest.approx(PDO / np.log(2))
    assert sc.OFFSET == pytest.approx(BASE_SCORE - sc.FACTOR * np.log(BASE_ODDS))


def test_base_score_means_base_odds():
    # score 600 -> odds 50:1 -> PD = 1/51
    assert sc.score_to_pd(BASE_SCORE) == pytest.approx(1 / (1 + BASE_ODDS))


def test_pdo_doubles_the_odds():
    p1, p2 = sc.score_to_pd(BASE_SCORE), sc.score_to_pd(BASE_SCORE + PDO)
    assert ((1 - p2) / p2) / ((1 - p1) / p1) == pytest.approx(2)


def test_score_and_pd_are_inverse():
    pds = np.array([0.01, 0.1, 0.3, 0.7])
    assert sc.score_to_pd(sc.pd_to_score(pds)) == pytest.approx(pds)


@pytest.fixture
def card(credit_sample):
    df, y = credit_sample
    return sc.build_scorecard(df, y, numeric=["x1", "x2"], categorical=["cat"]), df, y


def test_coefficients_have_negative_sign(card):
    built, _, _ = card
    assert all(b < 0 for b in built["coefficients"].values())


def test_points_add_up_to_the_scaled_logit(card):
    built, df, _ = card
    score = sc.score_from_points(df, built)
    woe_df = sc.woe.transform(df, built["specs"], built["features"])
    logit = built["intercept"] + sum(built["coefficients"][f] * woe_df[f] for f in built["features"])
    exact = sc.OFFSET - sc.FACTOR * logit
    # each characteristic's points are rounded to an integer
    assert (score - exact).abs().max() <= 0.5 * len(built["features"]) + 1e-9


def test_higher_score_means_lower_pd(card):
    built, df, y = card
    score = sc.score_from_points(df, built)
    pd_hat = sc.score_to_pd(score)
    assert np.all(np.diff(pd_hat[np.argsort(score.to_numpy())]) <= 1e-12)
    assert pd_hat.min() > 0 and pd_hat.max() < 1
    # the score ranks real risk: defaulters score lower on average
    assert score[y == 1].mean() < score[y == 0].mean()


def test_risk_bands():
    scores = pd.Series(np.arange(100))
    cut = fit_band_cutoffs(scores)
    bands = assign_band(scores, cut)
    assert (bands == "High").mean() == pytest.approx(0.20, abs=0.02)
    assert (bands == "Low").mean() == pytest.approx(0.40, abs=0.02)
    assert assign_band([cut["high_below"] - 1], cut)[0] == "High"
    assert assign_band([cut["low_from"]], cut)[0] == "Low"
    summary = band_summary(scores, (scores < 30).astype(int), np.full(100, 0.1), cut)
    assert summary["band"].tolist() == ["Low", "Medium", "High"]
    assert summary["default_rate_pct"].is_monotonic_increasing


def test_cutoff_table_tradeoff():
    scores = np.arange(100)
    y = (scores < 20).astype(int)
    t = cutoff_table(scores, y, [0, 10, 20, 50])
    assert t["approval_rate_pct"].tolist() == [100, 90, 80, 50]
    assert t["approval_rate_pct"].is_monotonic_decreasing
    assert t.loc[t.cutoff_score == 20, "approved_bad_rate_pct"].iloc[0] == 0
    assert t.loc[t.cutoff_score == 20, "bads_rejected_pct"].iloc[0] == 100
