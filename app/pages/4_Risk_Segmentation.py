"""Page 5: Risk Segmentation - risk bands and portfolio concentration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RISK_EXPLANATIONS, band_bar, figure, kpi_row, load_portfolio, setup_page  # noqa: E402

import streamlit as st  # noqa: E402

from src.portfolio import band_summary, concentration  # noqa: E402

setup_page("Risk Segmentation", "Where does the risk sit in the loan book?")
pf = load_portfolio()

f1, f2, f3 = st.columns(3)
purposes = f1.multiselect("Purpose", sorted(pf["purpose"].unique()), placeholder="All purposes")
grades = f2.multiselect("LendingClub grade", sorted(pf["grade"].unique()), placeholder="All grades")
sample = f3.selectbox("Sample", ["All loans", "Train", "Test (out-of-time)"])

view = pf
if purposes:
    view = view[view["purpose"].isin(purposes)]
if grades:
    view = view[view["grade"].isin(grades)]
if sample != "All loans":
    view = view[view["sample"] == sample.split()[0].lower()]
if view.empty:
    st.warning("No loans match these filters.")
    st.stop()

kpi_row([
    ("Loans", f"{len(view):,}"),
    ("High-risk share", f"{(view['risk_category'] == 'High').mean():.1%}"),
    ("Average model PD", f"{view['probability_of_default'].mean():.2%}"),
    ("Observed 36-month default rate", f"{view['default'].mean():.2%}"),
])
if sample == "Train":
    st.caption("Training-period PDs are in-sample; use the out-of-time sample to judge accuracy.")

bands = band_summary(view)
st.subheader("Risk bands")
a, b = st.columns(2)
with a:
    st.altair_chart(band_bar(bands, "loan_share_pct", "Share of loans (%)"), use_container_width=True)
with b:
    st.altair_chart(band_bar(bands, "default_rate_pct", "Observed default rate (%)"), use_container_width=True)
st.dataframe(bands, hide_index=True, use_container_width=True)
for band, text in RISK_EXPLANATIONS.items():
    st.markdown(f"- **{band}:** {text}")

st.subheader("Where are the high-risk loans?")
dim = st.selectbox("Break down by", ["purpose", "addr_state", "grade", "home_ownership", "term_months"])
st.dataframe(concentration(view, dim, top=None), hide_index=True, use_container_width=True)
st.caption("Concentration ratio = share of High-risk loans / share of the portfolio. Above 1: the segment "
           "holds more than its fair share of High-risk loans.")

st.subheader("Our risk band vs LendingClub's grade (default rate %)")
cross = (view.pivot_table(index="risk_category", columns="grade", values="default", aggfunc="mean") * 100)
st.dataframe(cross.reindex([b for b in ["Low", "Medium", "High"] if b in cross.index]).round(1),
             use_container_width=True)
st.caption("Read down each column: where the default rate rises from Low to High within the same grade, "
           "the scorecard adds information that LendingClub's grade does not capture. Small cells "
           "(grades F-G) are noisy.")

figure("portfolio", "02_predicted_vs_observed_by_band", "Predicted vs observed default rate by band (all loans)")
