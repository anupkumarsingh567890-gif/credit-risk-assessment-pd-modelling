"""
Weight of Evidence (WoE) and Information Value (IV).

For every bin i of a characteristic:
    dist_good_i = goods in bin i / all goods
    dist_bad_i  = bads in bin i  / all bads
    WoE_i       = ln(dist_good_i / dist_bad_i)
    IV          = sum_i (dist_good_i - dist_bad_i) * WoE_i

WoE > 0 : bin is safer than average.   WoE < 0 : bin is riskier than average.
IV rule of thumb: <0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium,
                  0.3-0.5 strong, >0.5 suspiciously strong (check for leakage).

Binning rules
  * Numeric: start from 20 quantile bins and merge neighbours until the bad
    rate moves in one direction only (monotonic) and every bin holds at least
    5% of loans.
  * Missing values form their own bin. If that bin holds less than 5% of
    loans, its WoE is too unstable, so it is merged into the bin with the
    closest bad rate.
  * Categorical: one bin per category; categories with less than 2% of loans
    are pooled into a single "RARE" bin.
  * A value never seen in training (new category, or missing when training
    had no missing values) is scored as the RISKIEST bin - a conservative
    choice for a credit decision.
  * 0.5 is added to every good and bad count (then re-normalised) so an empty
    cell never produces log(0).
"""
import numpy as np
import pandas as pd

MIN_BIN_SHARE = 0.05       # minimum share of loans in a numeric or missing bin
RARE_CATEGORY_SHARE = 0.02  # categories below this share are pooled as "RARE"
START_BINS = 20
MISSING = -1               # key used for missing numeric values
RARE = "RARE"


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Not useful"
    if iv < 0.1:
        return "Weak"
    if iv < 0.3:
        return "Medium"
    if iv < 0.5:
        return "Strong"
    return "Suspicious"


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def find_monotonic_edges(x: pd.Series, y: pd.Series) -> list[float]:
    """Monotonic binning in three steps.

    1. Start with up to 20 quantile bins.
    2. Decide the direction of risk from the Spearman correlation of x with y.
    3. Merge neighbouring bins while the bad rate breaks that direction,
       then merge any bin holding < 5% of loans into its closest neighbour
       (merging two neighbours of a monotonic sequence keeps it monotonic).
    """
    x, y = x[x.notna()], y[x.notna()]
    if x.nunique() < 2:  # constant feature (can happen in a small CV fold)
        return [-np.inf, np.inf]
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, START_BINS + 1))[1:-1])
    edges = [-np.inf, *cuts.tolist(), np.inf]
    increasing = x.corr(y, method="spearman") >= 0

    def bin_stats(e):
        groups = y.groupby(pd.cut(x, e), observed=False).agg(["size", "mean"])
        return groups["size"].to_numpy(), groups["mean"].to_numpy()

    # 3a. merge bins that break the monotonic direction
    while len(edges) > 2:
        _, rates = bin_stats(edges)
        diffs = np.diff(rates)
        breaks = np.where(diffs < 0)[0] if increasing else np.where(diffs > 0)[0]
        if len(breaks) == 0:
            break
        del edges[breaks[0] + 1]  # merge bin i with bin i+1

    # 3b. merge small bins into the neighbour with the closest bad rate
    while len(edges) > 2:
        sizes, rates = bin_stats(edges)
        shares = sizes / sizes.sum()
        i = int(np.argmin(shares))
        if shares[i] >= MIN_BIN_SHARE:
            break
        if i == 0:
            del edges[1]
        elif i == len(sizes) - 1:
            del edges[-2]
        elif abs(rates[i] - rates[i - 1]) <= abs(rates[i] - rates[i + 1]):
            del edges[i]
        else:
            del edges[i + 1]
    return edges


def raw_keys(x: pd.Series, kind: str, edges: list[float] | None = None) -> pd.Series:
    """Raw bin keys: interval index for numeric (-1 = missing), the text value for categorical."""
    if kind == "numeric":
        return pd.cut(x, edges, labels=False).fillna(MISSING).astype(int)
    return x.astype("string").fillna("Missing").astype(str)


def keys_for(x: pd.Series, spec: dict) -> pd.Series:
    """Final bin keys for scoring: applies rare/missing pooling and the unseen-value rule."""
    keys = raw_keys(x, spec["kind"], spec["edges"])
    if spec["key_map"]:
        keys = keys.replace(spec["key_map"])
    known = keys.isin(list(spec["woe_map"]))
    return keys.where(known, spec["fallback_key"])


def bin_label(key, edges: list[float] | None) -> str:
    if edges is None:
        return str(key)
    if key == MISSING:
        return "Missing"
    lo, hi = edges[key], edges[key + 1]
    lo_txt = "-inf" if np.isinf(lo) else f"{lo:,.4g}"
    hi_txt = "inf" if np.isinf(hi) else f"{hi:,.4g}"
    return f"({lo_txt}, {hi_txt}]"


# --------------------------------------------------------------------------- #
# WoE table
# --------------------------------------------------------------------------- #
def woe_table(keys: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"key": keys, "bad": y}).groupby("key")["bad"].agg(loans="size", bads="sum")
    t["goods"] = t["loans"] - t["bads"]
    t["share_pct"] = t["loans"] / t["loans"].sum() * 100
    t["bad_rate_pct"] = t["bads"] / t["loans"] * 100
    # 0.5 smoothing, then re-normalise so each distribution sums to 1
    smooth_good, smooth_bad = t["goods"] + 0.5, t["bads"] + 0.5
    t["dist_good"] = smooth_good / smooth_good.sum()
    t["dist_bad"] = smooth_bad / smooth_bad.sum()
    t["woe"] = np.log(t["dist_good"] / t["dist_bad"])
    t["iv"] = (t["dist_good"] - t["dist_bad"]) * t["woe"]
    return t.reset_index()


def _pool_small_missing(keys: pd.Series, y: pd.Series) -> dict:
    """If the missing bin is too small, map it to the bin with the closest bad rate."""
    is_missing = keys == MISSING
    if not is_missing.any() or is_missing.mean() >= MIN_BIN_SHARE or (~is_missing).sum() == 0:
        return {}
    rates = y[~is_missing].groupby(keys[~is_missing]).mean()
    target = (rates - y[is_missing].mean()).abs().idxmin()
    return {MISSING: int(target)}


def _pool_rare_categories(keys: pd.Series) -> dict:
    shares = keys.value_counts(normalize=True)
    rare = shares[shares < RARE_CATEGORY_SHARE].index.tolist()
    return {level: RARE for level in rare} if len(rare) >= 2 else {}


def fit_feature(x: pd.Series, y: pd.Series, kind: str) -> dict:
    """Learn bins and WoE values for one characteristic on training data."""
    edges = find_monotonic_edges(x, y) if kind == "numeric" else None
    keys = raw_keys(x, kind, edges)
    key_map = _pool_small_missing(keys, y) if kind == "numeric" else _pool_rare_categories(keys)
    if key_map:
        keys = keys.replace(key_map)

    table = woe_table(keys, y)
    labels = [bin_label(k, edges) for k in table["key"]]
    if kind == "numeric" and MISSING in key_map:
        labels = [f"{lab} or Missing" if k == key_map[MISSING] else lab
                  for k, lab in zip(table["key"], labels)]
    if kind == "categorical" and key_map:
        pooled = ", ".join(sorted(map(str, key_map)))
        labels = [f"RARE ({pooled})" if k == RARE else lab for k, lab in zip(table["key"], labels)]
    table["bin"] = labels

    return {
        "feature": x.name,
        "kind": kind,
        "edges": edges,
        "key_map": key_map,
        "woe_map": dict(zip(table["key"], table["woe"])),
        "fallback_key": table.loc[table["woe"].idxmin(), "key"],  # riskiest bin
        "iv": float(table["iv"].sum()),
        "table": table,
    }


def fit_all(df: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]) -> dict:
    specs = {}
    for col in numeric:
        specs[col] = fit_feature(df[col], y, "numeric")
    for col in categorical:
        specs[col] = fit_feature(df[col], y, "categorical")
    return specs


def transform(df: pd.DataFrame, specs: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Replace raw values by their WoE."""
    features = features or list(specs)
    out = pd.DataFrame(index=df.index)
    for col in features:
        spec = specs[col]
        out[col] = keys_for(df[col], spec).map(spec["woe_map"]).astype(float)
    return out


def iv_summary(specs: dict) -> pd.DataFrame:
    t = pd.DataFrame(
        [{"feature": f, "iv": s["iv"], "n_bins": len(s["table"])} for f, s in specs.items()]
    ).sort_values("iv", ascending=False)
    t["strength"] = t["iv"].apply(iv_strength)
    return t.reset_index(drop=True)


def bins_long_table(specs: dict, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        t = specs[f]["table"].copy()
        t.insert(0, "feature", f)
        rows.append(t[["feature", "bin", "loans", "share_pct", "bads", "bad_rate_pct", "woe", "iv"]])
    return pd.concat(rows, ignore_index=True)
