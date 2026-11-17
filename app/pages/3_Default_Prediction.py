"""Page 4: Default Prediction - assess an individual borrower."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RISK_EXPLANATIONS, kpi_row, load_artifacts, load_portfolio, rate_bar, risk_badge, setup_page,
)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.assessment import EXAMPLE_APPLICATION as EX  # noqa: E402
from src.assessment import HOME_OWNERSHIP, PURPOSES, VERIFICATION, assess_borrower  # noqa: E402
from src.scorecard import score_to_pd  # noqa: E402

setup_page("Default Prediction - Borrower Assessment",
           "Enter an application to get a credit score, probability of default, risk category and decision")
art = load_artifacts()
card = art["card"]
portfolio_pd = load_portfolio()["probability_of_default"].mean()

with st.form("application"):
    st.subheader("Loan request")
    a1, a2, a3 = st.columns(3)
    loan_amnt = a1.number_input("Loan amount ($)", min_value=500, max_value=100_000, value=EX["loan_amnt"],
                                step=500, help="LendingClub range 500-35,000; larger amounts are extrapolation")
    term = a2.selectbox("Term (months)", [36, 60])
    purpose = a3.selectbox("Purpose", PURPOSES)

    st.subheader("Borrower")
    b1, b2, b3, b4 = st.columns(4)
    annual_inc = b1.number_input("Annual income ($)", min_value=1_000, max_value=5_000_000,
                                 value=EX["annual_inc"], step=1_000)
    verif = b1.selectbox("Income verification", VERIFICATION)
    dti = b2.number_input("Debt-to-income (%)", min_value=0.0, max_value=100.0, value=EX["dti"], step=0.5,
                          help="Monthly debt payments excluding mortgage / monthly income")
    home = b2.selectbox("Home ownership", HOME_OWNERSHIP)
    emp_known = b3.checkbox("Employment length known", True)
    emp_len = b3.slider("Employment length (years, 10 = 10+)", 0, 10, EX["emp_length_years"])
    emp_title = b4.checkbox("Job title provided", True)

    st.subheader("Credit bureau")
    c1, c2, c3, c4 = st.columns(4)
    revol_util = c1.number_input("Revolving utilisation (%)", 0.0, 150.0, EX["revol_util"], step=1.0)
    revol_bal = c1.number_input("Revolving balance ($)", 0, 2_000_000, EX["revol_bal"], step=500)
    inq = c2.number_input("Inquiries, last 6 months", 0, 30, EX["inq_last_6mths"])
    delinq = c2.number_input("Delinquencies, last 2 years", 0, 30, EX["delinq_2yrs"])
    history = c2.number_input("Credit history (years)", 0.0, 80.0, EX["credit_history_years"], step=0.5)
    ever_delinq = c3.checkbox("Any past delinquency on file", False)
    mths_delinq = c3.number_input("Months since last delinquency (if any)", 0, 200, 24)
    pub_rec = c3.number_input("Derogatory public records", 0, 20, EX["pub_rec"])
    bankrupt = c4.number_input("Public-record bankruptcies", 0, 20, EX["pub_rec_bankruptcies"])
    open_acc = c4.number_input("Open credit lines", 0, 100, EX["open_acc"])
    total_acc = c4.number_input("Total credit lines", 0, 200, EX["total_acc"])

    submitted = st.form_submit_button("Assess borrower", type="primary")

if not submitted:
    st.info("Fill in the application and press **Assess borrower**. The defaults are a typical application.")
    st.stop()

app = {
    "loan_amnt": loan_amnt, "term_months": term, "annual_inc": annual_inc, "dti": dti,
    "emp_length_years": emp_len if emp_known else None, "emp_title_provided": emp_title,
    "home_ownership": home, "verification_status": verif, "purpose": purpose,
    "delinq_2yrs": delinq, "inq_last_6mths": inq, "open_acc": open_acc, "total_acc": total_acc,
    "pub_rec": pub_rec, "pub_rec_bankruptcies": bankrupt, "revol_bal": revol_bal,
    "revol_util": revol_util, "mths_since_last_delinq": mths_delinq if ever_delinq else None,
    "credit_history_years": history,
}
try:
    r = assess_borrower(app, art)
except ValueError as exc:
    st.error(f"Please correct the application: {exc}")
    st.stop()

for w in r["warnings"]:
    st.warning(w)

st.divider()
band = r["risk_category"]
scorecard_pd = float(score_to_pd(r["credit_score"]))
kpi_row([
    ("Credit score", int(r["credit_score"]), "Scorecard points; 600 = 50:1 odds, +20 points doubles the odds"),
    ("Scorecard PD", f"{scorecard_pd:.1%}", "PD implied by the score (WoE logistic regression)"),
    (f"Model PD ({r['model']})", f"{r['probability_of_default']:.1%}",
     f"Champion PD model; portfolio average {portfolio_pd:.1%}"),
    ("Decision", r["decision"]),
])

left, right = st.columns([2, 3])
with left:
    st.markdown(f"### {risk_badge(band)}", unsafe_allow_html=True)
    st.markdown(f"**{r['creditworthiness']}.**")
    st.write(RISK_EXPLANATIONS[band])
    cut = card["band_cutoffs"]
    st.caption(f"Bands: High < {cut['high_below']:.0f} <= Medium < {cut['low_from']:.0f} <= Low. "
               f"Approval cut-off: {card['approval_cutoff']:.0f}. Decision rule: High -> Decline; "
               "score at or above the cut-off -> Approve; otherwise Refer.")
    st.markdown("**Main factors lowering the score**")
    reasons = [x for x in r["main_reasons"].split("; ") if x and x != "-"]
    st.markdown("\n".join(f"- {x}" for x in reasons) or "- None: every characteristic is in its best bin.")
with right:
    pts = pd.DataFrame(list(r["points_breakdown"].items()), columns=["characteristic", "points"])
    st.markdown("**Points by characteristic**")
    st.altair_chart(rate_bar(pts, "characteristic", "points", "Characteristic", "Points", horizontal=True,
                             sort="-x"), use_container_width=True)

st.caption("The scorecard drives the decision and the reason codes because it is transparent. The champion "
           "model provides the PD estimate; the two are strongly but not perfectly aligned (see the "
           "validation summary), so individual estimates can differ.")
