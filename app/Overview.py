"""
Streamlit app - page 1: Overview.

Run from the project root:   streamlit run app/Overview.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DISCLAIMER, RISK_EXPLANATIONS, band_bar, champion, figure, kpi_row, load_portfolio, load_table,
    setup_page,
)

import streamlit as st  # noqa: E402

from src.portfolio import band_summary  # noqa: E402

setup_page("Credit Risk Modelling - Default Prediction & Credit Scoring",
           "LendingClub personal loans issued June 2007 - December 2011 | 36-month default definition | "
           "out-of-time validation on July - December 2011 loans")

pf = load_portfolio()
champ = champion()
metrics = load_table("eval_metrics")
score_perf = load_table("scorecard_performance")
validation = load_table("validation_summary")

test = metrics[metrics["sample"] == "test"].set_index("model")
sc_test = score_perf[(score_perf["sample"] == "test")].set_index("model")

kpi_row([
    ("Loans", f"{len(pf):,}"),
    ("Exposure", f"${pf['loan_amnt'].sum() / 1e6:,.0f}M"),
    ("Default rate", f"{pf['default'].mean():.1%}", "Charged off, with arrears starting in the first 36 months"),
    ("PD champion", champ["model"]),
    ("Champion Gini", f"{test.loc[champ['model'], 'gini']:.3f}", "Out-of-time test, Jul-Dec 2011"),
    ("Scorecard Gini", f"{sc_test.loc['Scorecard (WoE logistic)', 'gini']:.3f}",
     f"Out-of-time test. LendingClub sub-grade benchmark: "
     f"{sc_test.loc['Benchmark: LendingClub sub-grade', 'gini']:.3f}"),
])

st.info(DISCLAIMER)

left, right = st.columns([3, 2])
with left:
    st.subheader("What the project does")
    st.markdown(
        """
| Stage | Approach |
|---|---|
| Target | Charged off **and** arrears began within the first 36 months (fully observed for every cohort) |
| Leakage control | Repayment fields and LendingClub's own grade / rate excluded from all models |
| Credit scorecard | Monotonic WoE binning, IV selection, logistic regression, PDO scaling (600 = 50:1, PDO 20) |
| PD models | Logistic Regression, Decision Tree, Random Forest, XGBoost; time-series cross-validation |
| Decisioning | Low / Medium / High risk bands and a KS-based approval cut-off |
| Validation | Out-of-time Gini / KS, PSI / CSI, Hosmer-Lemeshow, binomial calibration test, stability by quarter |
"""
    )
    st.subheader("Champion selection")
    st.write(champ["reason"])
    st.caption("Selection uses cross-validated AUC on the training period; the out-of-time test sample is "
               "used only for reporting.")
with right:
    st.subheader("Risk bands across the portfolio")
    bands = band_summary(pf)
    st.altair_chart(band_bar(bands, "default_rate_pct", "Observed default rate (%)"), use_container_width=True)
    for band in ["Low", "Medium", "High"]:
        row = bands.set_index("risk_category").loc[band]
        st.markdown(f"**{band}** - {row['loan_share_pct']:.0f}% of loans, "
                    f"{row['default_rate_pct']:.1f}% default rate. {RISK_EXPLANATIONS[band]}")

st.subheader("Validation status")
if not validation.empty:
    st.dataframe(validation, hide_index=True, use_container_width=True)
    flagged = validation[validation["status"] != "Pass"]["check"].tolist()
    if flagged:
        st.warning("Items to monitor: " + "; ".join(flagged) + ". See the Model Evaluation page.")

figure("eda", "23_vintage_curves", "Cumulative charge-offs by issue year (onset of arrears)")
