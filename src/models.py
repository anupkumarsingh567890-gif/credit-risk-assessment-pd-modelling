"""
Phase 6 - Probability of Default models.

Four models, all predicting the probability of default within 36 months:

1. Logistic Regression  - on WoE features (the statistical scorecard model)
2. Decision Tree        - simple, readable rules
3. Random Forest        - many de-correlated trees, averaged
4. XGBoost              - gradient-boosted trees, each tree fixes the last one's errors

Design choices
* Cross-validation uses TimeSeriesSplit: each fold trains on older loans and
  validates on the next block of newer loans, the same direction as the real
  out-of-time test. A random K-fold would let the model "see the future".
* No class weighting: weighting the rare class distorts probabilities, and
  for PD the probability itself matters (pricing, expected loss). Imbalance
  is handled by choosing a sensible decision threshold instead.
* Probability calibration (Platt scaling) is fitted on out-of-fold
  predictions and kept only if it lowers the Brier score by at least 1% in a
  cross-fitted comparison (smaller gains are noise, not worth the extra step).
"""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import woe
from src.config import CATEGORICAL_FEATURES, FLAG_FEATURES, NUMERIC_FEATURES, RANDOM_SEED
from src.scorecard import select_features

N_SPLITS = 4
MIN_BRIER_GAIN = 0.01  # calibrate only if out-of-fold Brier improves by >= 1%
CV = TimeSeriesSplit(n_splits=N_SPLITS)


# --------------------------------------------------------------------------- #
# WoE encoder (so WoE bins are re-learned inside every CV fold)
# --------------------------------------------------------------------------- #
class WoEEncoder(BaseEstimator, TransformerMixin):
    """WoE binning AND scorecard feature selection, learned inside each fit.

    Within cross-validation this runs separately in every fold, so neither the
    bins nor the IV / correlation / p-value selection ever see the fold that
    is being validated.
    """

    def __init__(self, numeric=None, categorical=None):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        sel = select_features(X, y, self.numeric, self.categorical)
        self.specs_, self.features_ = sel["specs"], sel["features"]
        return self

    def transform(self, X):
        return woe.transform(X, self.specs_, self.features_).to_numpy()


def tree_preprocessor() -> ColumnTransformer:
    """Median-impute numbers, one-hot encode categories."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES + FLAG_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


# --------------------------------------------------------------------------- #
# Model definitions and search spaces
# --------------------------------------------------------------------------- #
def model_specs() -> dict:
    return {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("woe", WoEEncoder(numeric=NUMERIC_FEATURES,
                                   categorical=FLAG_FEATURES + CATEGORICAL_FEATURES)),
                ("model", LogisticRegression(max_iter=1000)),
            ]),
            "params": {"model__C": [0.01, 0.1, 1.0, 10.0]},
            "search": "grid",
        },
        "Decision Tree": {
            "pipeline": Pipeline([
                ("prep", tree_preprocessor()),
                ("model", DecisionTreeClassifier(random_state=RANDOM_SEED)),
            ]),
            "params": {
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__min_samples_leaf": [50, 100, 200, 400],
            },
            "search": "grid",
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("prep", tree_preprocessor()),
                ("model", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)),
            ]),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, 8, 10],
                "model__min_samples_leaf": [20, 50, 100],
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
            "search": "random",
            "n_iter": 12,
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("prep", tree_preprocessor()),
                ("model", XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1,
                                        eval_metric="logloss", tree_method="hist")),
            ]),
            "params": {
                "model__n_estimators": [200, 400, 600],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3, 4, 5],
                "model__min_child_weight": [1, 5, 10, 20],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__reg_lambda": [1, 5, 10],
            },
            "search": "random",
            "n_iter": 15,
        },
    }


def tune(spec: dict, X: pd.DataFrame, y: pd.Series):
    """Hyperparameter search with time-series cross-validation (ROC-AUC)."""
    if spec["search"] == "grid":
        search = GridSearchCV(spec["pipeline"], spec["params"], cv=CV, scoring="roc_auc", n_jobs=1)
    else:
        search = RandomizedSearchCV(spec["pipeline"], spec["params"], n_iter=spec["n_iter"], cv=CV,
                                    scoring="roc_auc", random_state=RANDOM_SEED, n_jobs=1)
    search.fit(X, y)
    return search


def out_of_fold(estimator, X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Refit the tuned model fold by fold; return OOF predictions and fold scores."""
    oof = pd.Series(np.nan, index=X.index)
    folds = []
    for k, (tr, va) in enumerate(CV.split(X), start=1):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof.iloc[va] = p
        folds.append({"fold": k, "train_loans": len(tr), "valid_loans": len(va),
                      "valid_auc": roc_auc_score(y.iloc[va], p),
                      "valid_brier": brier_score_loss(y.iloc[va], p)})
    return oof, pd.DataFrame(folds)


# --------------------------------------------------------------------------- #
# Calibration (Platt scaling)
# --------------------------------------------------------------------------- #
def _logit(p):
    return logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)


def fit_platt(oof: pd.Series, y: pd.Series) -> LogisticRegression:
    mask = oof.notna()
    return LogisticRegression(C=1e6).fit(_logit(oof[mask]), y[mask])


def decide_calibration(oof: pd.Series, y: pd.Series) -> tuple[LogisticRegression | None, dict]:
    """Keep Platt scaling only if it improves the Brier score out-of-sample.

    The out-of-fold predictions come in time-ordered blocks (one per CV fold).
    Each block is calibrated with a Platt model fitted on the OTHER blocks, so
    the comparison of raw vs calibrated Brier is itself out-of-sample.
    """
    mask = oof.notna()
    p, t = oof[mask], y[mask]
    blocks = np.array_split(np.arange(len(p)), N_SPLITS)
    calibrated = np.empty(len(p))
    for idx in blocks:
        rest = np.setdiff1d(np.arange(len(p)), idx)
        platt = fit_platt(p.iloc[rest], t.iloc[rest])
        calibrated[idx] = platt.predict_proba(_logit(p.iloc[idx]))[:, 1]
    raw = brier_score_loss(t, p)
    cal = brier_score_loss(t, calibrated)
    use = cal < raw * (1 - MIN_BRIER_GAIN)
    final = fit_platt(p, t) if use else None
    return final, {"oof_brier_raw": raw, "oof_brier_calibrated": cal, "calibrated": bool(use)}


# --------------------------------------------------------------------------- #
# Prediction helper used everywhere (scripts and Streamlit app)
# --------------------------------------------------------------------------- #
def predict_pd(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    if bundle.get("calibrator") is not None:
        p = bundle["calibrator"].predict_proba(_logit(p))[:, 1]
    return p
