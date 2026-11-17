"""
Phase 8 - Individual borrower assessment.

Takes the information on a loan application, turns it into the model
features exactly as in Phase 2, and returns:
    * credit score (scorecard points) and the points per characteristic
    * probability of default (champion model)
    * risk category (Low / Medium / High)
    * a creditworthiness decision with the main reasons behind it
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, MODELS_DIR
from src.models import predict_pd
from src.preprocessing import apply_caps
from src.risk_bands import assign_band
from src.scorecard import score_from_points

# Example application used as default values in the app
EXAMPLE_APPLICATION = {
    "loan_amnt": 12000,
    "term_months": 36,
    "annual_inc": 60000,
    "dti": 14.0,
    "emp_length_years": 5,
    "emp_title_provided": True,
    "home_ownership": "RENT",
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 9,
    "total_acc": 20,
    "pub_rec": 0,
    "pub_rec_bankruptcies": 0,
    "revol_bal": 10000,
    "revol_util": 50.0,
    "mths_since_last_delinq": None,
    "credit_history_years": 12.0,
}


# Allowed values and the range seen in the training data
HOME_OWNERSHIP = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION = ["Not Verified", "Source Verified", "Verified"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
            "small_business", "car", "wedding", "medical", "moving", "house", "vacation",
            "educational", "other"]
COUNT_FIELDS = ["delinq_2yrs", "inq_last_6mths", "open_acc", "total_acc", "pub_rec",
                "pub_rec_bankruptcies"]
TRAINING_RANGE = {"loan_amnt": (500, 35000), "dti": (0, 30), "revol_util": (0, 100),
                  "credit_history_years": (3, 66)}


def validate_application(app: dict) -> list[str]:
    """Check an application before scoring.

    Raises ValueError listing every impossible value; returns warnings for
    values that are possible but outside the range the model was trained on.
    """
    missing = [k for k in EXAMPLE_APPLICATION if k not in app]
    if missing:
        raise ValueError(f"Missing application fields: {missing}")

    errors = []
    if app["loan_amnt"] <= 0:
        errors.append("Loan amount must be positive.")
    if app["annual_inc"] <= 0:
        errors.append("Annual income must be positive.")
    if app["term_months"] not in (36, 60):
        errors.append("Term must be 36 or 60 months.")
    if app["dti"] < 0:
        errors.append("Debt-to-income cannot be negative.")
    if not 0 <= app["revol_util"] <= 150:
        errors.append("Revolving utilisation must be between 0% and 150%.")
    if app["revol_bal"] < 0:
        errors.append("Revolving balance cannot be negative.")
    if app["credit_history_years"] < 0:
        errors.append("Credit history length cannot be negative.")
    for f in COUNT_FIELDS:
        if app[f] < 0 or int(app[f]) != app[f]:
            errors.append(f"{f} must be a whole number >= 0.")
    if app["open_acc"] > app["total_acc"]:
        errors.append("Open credit lines cannot exceed total credit lines.")
    if app["pub_rec_bankruptcies"] > app["pub_rec"]:
        errors.append("Bankruptcies are public records, so they cannot exceed derogatory public records.")
    emp = app.get("emp_length_years")
    if emp is not None and not 0 <= emp <= 10:
        errors.append("Employment length must be 0-10 years (10 = 10 or more).")
    delinq = app.get("mths_since_last_delinq")
    if delinq is not None and delinq < 0:
        errors.append("Months since last delinquency cannot be negative.")
    if app["home_ownership"] not in HOME_OWNERSHIP:
        errors.append(f"Home ownership must be one of {HOME_OWNERSHIP}.")
    if app["verification_status"] not in VERIFICATION:
        errors.append(f"Verification status must be one of {VERIFICATION}.")
    if app["purpose"] not in PURPOSES:
        errors.append(f"Purpose must be one of {PURPOSES}.")
    if errors:
        raise ValueError(" ".join(errors))

    warnings = []
    for f, (lo, hi) in TRAINING_RANGE.items():
        if not lo <= app[f] <= hi:
            warnings.append(f"{f} = {app[f]} is outside the training range ({lo}-{hi}); "
                            "the estimate is an extrapolation.")
    return warnings


def load_artifacts() -> dict:
    """Everything needed to score a borrower, produced by Phases 2-7."""
    needed = [MODELS_DIR / n for n in ("champion.json", "scorecard.joblib", "preprocessing.json")]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Model files not found in {MODELS_DIR}: {missing}. "
                                "Run the pipeline first: python run_all.py")
    champion = json.loads((MODELS_DIR / "champion.json").read_text())
    return {
        "card": joblib.load(MODELS_DIR / "scorecard.joblib"),
        "bundle": joblib.load(MODELS_DIR / champion["file"]),
        "champion": champion,
        "caps": json.loads((MODELS_DIR / "preprocessing.json").read_text())["caps"],
    }


def application_to_features(app: dict, caps: dict) -> pd.DataFrame:
    """Build the model features for one application (same rules as Phase 2)."""
    a = dict(app)
    emp_len = a.get("emp_length_years")
    row = {
        "loan_amnt": a["loan_amnt"],
        "term_months": a["term_months"],
        "annual_inc": a["annual_inc"],
        "dti": a["dti"],
        "emp_length_years": np.nan if emp_len is None else emp_len,
        "delinq_2yrs": a["delinq_2yrs"],
        "inq_last_6mths": a["inq_last_6mths"],
        "open_acc": a["open_acc"],
        "total_acc": a["total_acc"],
        "pub_rec": a["pub_rec"],
        "pub_rec_bankruptcies": a["pub_rec_bankruptcies"],
        "revol_bal": a["revol_bal"],
        "revol_util": a["revol_util"],
        "mths_since_last_delinq": np.nan if a.get("mths_since_last_delinq") is None
        else a["mths_since_last_delinq"],
        "credit_history_years": a["credit_history_years"],
        "loan_to_income": a["loan_amnt"] / a["annual_inc"],
        "revol_bal_to_income": a["revol_bal"] / a["annual_inc"],
        "emp_length_missing": int(emp_len is None),
        "emp_title_missing": int(not a.get("emp_title_provided", True)),
        "has_delinq_history": int(a.get("mths_since_last_delinq") is not None),
        "has_public_record": int(a["pub_rec"] > 0),
        "home_ownership": a["home_ownership"],
        "verification_status": a["verification_status"],
        "purpose": a["purpose"],
    }
    df = pd.DataFrame([row])
    return apply_caps(df, caps)[ALL_FEATURES]


def decision(score: float, band: str, cutoff: float) -> str:
    if band == "High":
        return "Decline"
    if score >= cutoff:
        return "Approve"
    return "Refer for manual review"


def reason_codes(breakdown: pd.Series, card: dict, top_n: int = 3) -> list[str]:
    """Characteristics that cost the most points vs the best possible bin."""
    best = card["points"].groupby("feature")["points"].max()
    lost = (best[breakdown.index] - breakdown).sort_values(ascending=False)
    return [f"{f}: {int(v)} points below best" for f, v in lost.head(top_n).items() if v > 0]


def assess_features(features: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Score one or many rows that already have model features."""
    card = art["card"]
    score, breakdown = score_from_points(features, card, return_breakdown=True)
    pd_hat = predict_pd(art["bundle"], features[ALL_FEATURES])
    band = assign_band(score, card["band_cutoffs"])
    out = pd.DataFrame({
        "credit_score": score.to_numpy(),
        "probability_of_default": pd_hat,
        "risk_category": band,
    }, index=features.index)
    out["decision"] = [decision(s, b, card["approval_cutoff"]) for s, b in zip(out["credit_score"], band)]
    out["main_reasons"] = ["; ".join(reason_codes(breakdown.loc[i], card)) or "-" for i in features.index]
    return out


def assess_borrower(app: dict, art: dict | None = None) -> dict:
    warnings = validate_application(app)  # raises ValueError on impossible inputs
    art = art or load_artifacts()
    features = application_to_features(app, art["caps"])
    result = assess_features(features, art).iloc[0].to_dict()
    result["warnings"] = warnings
    _, breakdown = score_from_points(features, art["card"], return_breakdown=True)
    result["points_breakdown"] = breakdown.iloc[0].to_dict()
    result["creditworthiness"] = {
        "Low": "Creditworthy - low expected risk",
        "Medium": "Acceptable - average risk, terms may need adjusting",
        "High": "Not creditworthy at current terms - high expected risk",
    }[result["risk_category"]]
    result["model"] = art["champion"]["model"]
    return result
