"""Page 3: Credit Scoring - WoE / IV, scorecard points and approval cut-offs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    figure, kpi_row, load_artifacts, load_table, load_test_predictions, rate_bar, setup_page,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

setup_page("Credit Scoring", "Weight of Evidence scorecard: how each characteristic earns points")
art = load_artifacts()
card = art["card"]

st.markdown(
    f"**Score = {card['offset']:.1f} + {card['factor']:.2f} x ln(odds of good).** A score of 600 means "
    "odds of 50 good borrowers to 1 bad; every 20 points doubles the odds. Higher score = lower risk."
)

st.subheader("Information Value - which characteristics separate good and bad borrowers")
iv = load_table("scorecard_iv_table")
iv["in_scorecard"] = iv["feature"].isin(card["features"])
c1, c2 = st.columns([2, 3])
c1.dataframe(iv.round(4), hide_index=True, use_container_width=True)
c1.caption("IV < 0.02 not useful | 0.02-0.1 weak | 0.1-0.3 medium | > 0.3 strong. All drivers here are "
           "weak individually - expected without a bureau score; the scorecard combines them.")
with c2:
    figure("scorecard", "01_information_value")

st.subheader("Weight of Evidence and points by characteristic")
feature = st.selectbox("Characteristic", card["features"])
pts = card["points"][card["points"]["feature"] == feature][
    ["bin", "share_pct", "bad_rate_pct", "woe", "points"]].round(3)
a, b = st.columns(2)
a.dataframe(pts, hide_index=True, use_container_width=True)
a.caption("WoE > 0: safer than average, earns more points. Bins hold at least 5% of loans; rare "
          "categories are pooled.")
b.altair_chart(rate_bar(pts, "bin", "points", "Bin", "Points", sort=pts["bin"].tolist()),
               use_container_width=True)

with st.expander("Full scorecard table"):
    st.dataframe(load_table("scorecard_points"), hide_index=True, use_container_width=True)
with st.expander("Logistic regression coefficients (all negative, as expected with WoE inputs)"):
    st.dataframe(load_table("scorecard_coefficients"), hide_index=True, use_container_width=True)

st.subheader("Approval cut-off (out-of-time test sample)")
preds = load_test_predictions()
lo, hi = int(preds["score"].min()), int(preds["score"].max())
cutoff = st.slider("Approve applicants with a score at or above", lo, hi, int(card["approval_cutoff"]))
approved = preds["score"] >= cutoff
y = preds["default"]
kpi_row([
    ("Approval rate", f"{approved.mean():.1%}"),
    ("Bad rate - approved", f"{y[approved].mean():.2%}" if approved.any() else "-",
     f"Approving everyone: {y.mean():.2%}"),
    ("Defaulters rejected", f"{y[~approved].sum() / y.sum():.1%}"),
    ("Good borrowers rejected", f"{(1 - y[~approved]).sum() / (1 - y).sum():.1%}"),
])
st.caption(f"KS-optimal cut-off chosen on the training sample: {card['approval_cutoff']:.0f}. Note these loans "
           "were already approved by LendingClub, so a strict cut-off would reject many applicants that "
           "were in fact funded; in practice the cut-off comes from risk appetite and profitability.")

curve = []
for c in np.arange(lo, hi + 1, 2):
    app = preds["score"] >= c
    if app.sum() >= 50:
        curve.append({"approval_rate_pct": app.mean() * 100, "approved_bad_rate_pct": y[app].mean() * 100})
st.line_chart(pd.DataFrame(curve), x="approval_rate_pct", y="approved_bad_rate_pct",
              x_label="Approval rate (%)", y_label="Bad rate among approved (%)", color="#2a78d6")

figure("scorecard", "03_score_distribution", "Score distribution of goods and bads (test)")
