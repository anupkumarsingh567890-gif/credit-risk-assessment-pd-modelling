"""
Phase 4 - Credit scorecard: WoE logistic regression + points scaling.

Logistic regression on WoE features
    logit(PD) = b0 + b1*WoE_1 + ... + bk*WoE_k
Because WoE is positive for safe bins, every coefficient should be NEGATIVE
(safer bin -> lower PD). A positive sign means the feature is fighting other
features (multicollinearity), so it is removed.

Score scaling (Points to Double the Odds)
    factor = PDO / ln(2)
    offset = BASE_SCORE - factor * ln(BASE_ODDS)
    score  = offset + factor * ln(odds of good)
           = offset - factor * (b0 + sum_i b_i * WoE_i)
Points for bin j of feature i (so that the points add up to the score):
    points_ij = -(b_i * WoE_ij + b0 / k) * factor + offset / k
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import woe
from src.config import BASE_ODDS, BASE_SCORE, MAX_WOE_CORR, MIN_IV, PDO

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def select_by_iv(iv_table: pd.DataFrame) -> list[str]:
    keep = iv_table[(iv_table["iv"] >= MIN_IV) & (iv_table["iv"] < 0.5)]
    return keep["feature"].tolist()  # already sorted by IV, strongest first


def drop_correlated(woe_df: pd.DataFrame, ordered_features: list[str]) -> tuple[list[str], list[str]]:
    """Walk features from highest IV down; skip any too correlated with one already kept."""
    corr = woe_df[ordered_features].corr().abs()
    kept, dropped = [], []
    for f in ordered_features:
        if all(corr.loc[f, k] < MAX_WOE_CORR for k in kept):
            kept.append(f)
        else:
            dropped.append(f)
    return kept, dropped


def fit_logit(woe_df: pd.DataFrame, y: pd.Series, features: list[str]):
    X = sm.add_constant(woe_df[features])
    return sm.Logit(y, X).fit(disp=0)


def backward_elimination(woe_df: pd.DataFrame, y: pd.Series, features: list[str],
                         p_max: float = 0.05) -> tuple[list[str], list[dict]]:
    """Remove, one at a time, the worst feature with a wrong sign or p-value > p_max."""
    features = list(features)
    log = []
    while len(features) > 1:
        model = fit_logit(woe_df, y, features)
        coefs, pvals = model.params.drop("const"), model.pvalues.drop("const")
        wrong_sign = coefs[coefs > 0]
        not_significant = pvals[pvals > p_max]
        if len(wrong_sign):
            worst = pvals[wrong_sign.index].idxmax()
            reason = "positive coefficient (wrong sign)"
        elif len(not_significant):
            worst = not_significant.idxmax()
            reason = f"p-value {not_significant.max():.3f} > {p_max}"
        else:
            return features, log
        features.remove(worst)
        log.append({"feature": worst, "reason": reason})
    return features, log


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def points_table(specs: dict, model, features: list[str]) -> pd.DataFrame:
    b0 = model.params["const"]
    k = len(features)
    rows = []
    for f in features:
        beta = model.params[f]
        for _, r in specs[f]["table"].iterrows():
            pts = -(beta * r["woe"] + b0 / k) * FACTOR + OFFSET / k
            rows.append(
                {"feature": f, "bin": r["bin"], "key": r["key"], "woe": r["woe"],
                 "bad_rate_pct": r["bad_rate_pct"], "share_pct": r["share_pct"],
                 "coefficient": beta, "points": int(round(pts))}
            )
    return pd.DataFrame(rows)


def score_from_points(df: pd.DataFrame, card: dict, return_breakdown: bool = False):
    """Total score = sum of the points of the bin each characteristic falls into."""
    points, specs = card["points"], card["specs"]
    breakdown = pd.DataFrame(index=df.index)
    for f in card["features"]:
        keys = woe.keys_for(df[f], specs[f])  # unseen values -> riskiest bin
        rows = points[points["feature"] == f]
        breakdown[f] = keys.map(dict(zip(rows["key"], rows["points"]))).astype(int)
    score = breakdown.sum(axis=1)
    return (score, breakdown) if return_breakdown else score


def score_to_pd(score) -> np.ndarray:
    """Invert the scaling: PD = 1 / (1 + exp((score - offset) / factor))."""
    return 1 / (1 + np.exp((np.asarray(score, dtype=float) - OFFSET) / FACTOR))


def pd_to_score(pd_values) -> np.ndarray:
    p = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
    return OFFSET + FACTOR * np.log((1 - p) / p)


# --------------------------------------------------------------------------- #
# Full build
# --------------------------------------------------------------------------- #
def select_features(train: pd.DataFrame, y: pd.Series, numeric: list[str],
                    categorical: list[str]) -> dict:
    """WoE binning + the three selection steps. Used for the final scorecard AND
    inside every cross-validation fold, so CV scores are not inflated by a
    feature selection that has already seen the validation data."""
    specs = woe.fit_all(train, y, numeric, categorical)
    iv_table = woe.iv_summary(specs)
    iv_selected = select_by_iv(iv_table)
    if not iv_selected:
        raise ValueError("No characteristic reached the minimum IV; cannot build a scorecard.")
    woe_train = woe.transform(train, specs, iv_selected)
    corr_kept, corr_dropped = drop_correlated(woe_train, iv_selected)
    final, elimination_log = backward_elimination(woe_train, y, corr_kept)
    return {"specs": specs, "iv_table": iv_table, "iv_selected": iv_selected,
            "corr_dropped": corr_dropped, "elimination_log": elimination_log,
            "features": final, "woe_train": woe_train}


def build_scorecard(train: pd.DataFrame, y: pd.Series, numeric: list[str],
                    categorical: list[str]) -> dict:
    sel = select_features(train, y, numeric, categorical)
    final = sel["features"]
    model = fit_logit(sel["woe_train"], y, final)
    points = points_table(sel["specs"], model, final)

    return {
        "specs": sel["specs"],
        "iv_table": sel["iv_table"],
        "iv_selected": sel["iv_selected"],
        "corr_dropped": sel["corr_dropped"],
        "elimination_log": sel["elimination_log"],
        "features": final,
        "model": model,
        "intercept": float(model.params["const"]),
        "coefficients": model.params.drop("const").to_dict(),
        "points": points,
        "factor": FACTOR,
        "offset": OFFSET,
    }
