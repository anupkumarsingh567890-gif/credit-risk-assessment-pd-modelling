"""Weight of Evidence binning and Information Value."""
import numpy as np
import pandas as pd
import pytest

from src import woe


def test_woe_and_iv_match_hand_calculation():
    # bin A: 80 goods, 20 bads; bin B: 20 goods, 30 bads
    keys = pd.Series(["A"] * 100 + ["B"] * 50)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 30)
    t = woe.woe_table(keys, y).set_index("key")

    g = np.array([80.5, 20.5]) / 101   # smoothed, re-normalised goods
    b = np.array([20.5, 30.5]) / 51    # smoothed, re-normalised bads
    expected_woe = np.log(g / b)
    assert t["woe"].to_numpy() == pytest.approx(expected_woe)
    assert t["iv"].sum() == pytest.approx(((g - b) * expected_woe).sum())
    assert t["dist_good"].sum() == pytest.approx(1)
    assert t["dist_bad"].sum() == pytest.approx(1)
    assert t.loc["A", "woe"] > 0 > t.loc["B", "woe"]  # safer bin has positive WoE


def test_iv_is_non_negative(credit_sample):
    df, y = credit_sample
    for col, kind in [("x1", "numeric"), ("x2", "numeric"), ("cat", "categorical")]:
        assert woe.fit_feature(df[col], y, kind)["iv"] >= 0


def test_numeric_bins_are_monotonic_and_large_enough(credit_sample):
    df, y = credit_sample
    spec = woe.fit_feature(df["x1"], y, "numeric")
    t = spec["table"]
    assert t["bad_rate_pct"].is_monotonic_increasing  # risk rises with x1
    assert (t["share_pct"] >= woe.MIN_BIN_SHARE * 100 - 1e-9).all()
    assert len(t) >= 3


def test_constant_feature_gives_single_bin():
    edges = woe.find_monotonic_edges(pd.Series([1.0] * 50), pd.Series([0, 1] * 25))
    assert edges == [-np.inf, np.inf]


def test_small_missing_bin_is_pooled(credit_sample):
    df, y = credit_sample
    x = df["x1"].copy()
    x.iloc[:20] = np.nan  # 0.5% missing -> too small for its own bin
    spec = woe.fit_feature(x, y, "numeric")
    assert woe.MISSING in spec["key_map"]
    assert woe.MISSING not in spec["woe_map"]
    assert not woe.transform(pd.DataFrame({"x1": x}), {"x1": spec}).isna().any().any()


def test_large_missing_bin_keeps_its_own_woe(credit_sample):
    df, y = credit_sample
    x = df["x1"].copy()
    x.iloc[:800] = np.nan  # 20% missing
    spec = woe.fit_feature(x, y, "numeric")
    assert woe.MISSING in spec["woe_map"]


def test_rare_categories_are_pooled():
    x = pd.Series(["a"] * 600 + ["b"] * 380 + ["r1"] * 10 + ["r2"] * 10, name="cat")
    y = pd.Series(([0, 1] * 500))
    spec = woe.fit_feature(x, y, "categorical")
    assert spec["key_map"] == {"r1": woe.RARE, "r2": woe.RARE}
    assert woe.RARE in spec["woe_map"]


def test_unseen_values_get_riskiest_bin(credit_sample):
    df, y = credit_sample
    spec = woe.fit_feature(df["cat"], y, "categorical")
    out = woe.transform(pd.DataFrame({"cat": ["never_seen"]}), {"cat": spec})
    assert out["cat"].iloc[0] == min(spec["woe_map"].values())


def test_iv_strength_labels():
    assert woe.iv_strength(0.01) == "Not useful"
    assert woe.iv_strength(0.05) == "Weak"
    assert woe.iv_strength(0.2) == "Medium"
    assert woe.iv_strength(0.4) == "Strong"
    assert woe.iv_strength(0.7) == "Suspicious"
