"""PD model components, PD ranges and the borrower assessment."""
import numpy as np
import pandas as pd
import pytest

from src.assessment import EXAMPLE_APPLICATION, assess_borrower, validate_application
from src.config import ALL_FEATURES, MODEL_FILES, TARGET_COL
from src.models import decide_calibration, predict_pd


def test_woe_encoder_learns_only_from_its_fit_data(credit_sample, monkeypatch):
    from src import models
    df, y = credit_sample
    monkeypatch.setattr(models, "select_features", lambda X, yy, n, c: {
        "specs": models.woe.fit_all(X, yy, n, c), "features": n + c})
    enc = models.WoEEncoder(numeric=["x1"], categorical=["cat"]).fit(df.iloc[:2000], y.iloc[:2000])
    out = enc.transform(pd.DataFrame({"x1": [0.0], "cat": ["unseen"]}))
    assert out.shape == (1, 2)
    assert not np.isnan(out).any()


def test_calibration_decision_returns_out_of_sample_briers():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0.05, 0.3, 4000))
    y = pd.Series((rng.random(4000) < p * 0.5).astype(int))  # predictions twice too high
    calibrator, info = decide_calibration(p, y)
    assert info["calibrated"] is True and calibrator is not None
    assert info["oof_brier_calibrated"] < info["oof_brier_raw"]


def test_validation_rejects_impossible_inputs():
    bad = dict(EXAMPLE_APPLICATION, annual_inc=0, open_acc=30, total_acc=10)
    with pytest.raises(ValueError) as err:
        validate_application(bad)
    assert "income" in str(err.value) and "Open credit lines" in str(err.value)


def test_validation_warns_outside_training_range():
    warnings = validate_application(dict(EXAMPLE_APPLICATION, loan_amnt=50_000))
    assert any("loan_amnt" in w for w in warnings)
    assert validate_application(EXAMPLE_APPLICATION) == []


def test_validation_requires_all_fields():
    app = dict(EXAMPLE_APPLICATION)
    app.pop("dti")
    with pytest.raises(ValueError, match="Missing"):
        validate_application(app)


# ---- tests on the trained models (skipped if the pipeline has not been run) ----
def test_all_models_give_valid_pds(clean_data, artifacts):
    import joblib
    from src.config import MODELS_DIR
    test = clean_data[clean_data["sample"] == "test"].head(2000)
    for name, file in MODEL_FILES.items():
        bundle = joblib.load(MODELS_DIR / f"{file}.joblib")
        p = predict_pd(bundle, test[ALL_FEATURES])
        assert np.isfinite(p).all(), name
        assert ((p > 0) & (p < 1)).all(), name
        # better than random on the out-of-time sample
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(test[TARGET_COL], p) > 0.55, name


def test_borrower_assessment_output(artifacts):
    r = assess_borrower(EXAMPLE_APPLICATION, artifacts)
    pts = artifacts["card"]["points"].groupby("feature")["points"]
    assert pts.min().sum() <= r["credit_score"] <= pts.max().sum()
    assert 0 < r["probability_of_default"] < 1
    assert r["risk_category"] in {"Low", "Medium", "High"}
    assert r["decision"] in {"Approve", "Refer for manual review", "Decline"}
    assert sum(r["points_breakdown"].values()) == r["credit_score"]


def test_riskier_application_scores_lower(artifacts):
    safe = assess_borrower(EXAMPLE_APPLICATION, artifacts)
    risky = assess_borrower(dict(EXAMPLE_APPLICATION, revol_util=95.0, inq_last_6mths=4,
                                 purpose="small_business", term_months=60, annual_inc=25_000,
                                 pub_rec=1), artifacts)
    assert risky["credit_score"] < safe["credit_score"]
    assert risky["probability_of_default"] > safe["probability_of_default"]
