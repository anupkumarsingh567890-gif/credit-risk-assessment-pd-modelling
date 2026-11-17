"""Page 2: Credit Risk Analysis - interactive default-rate drivers (EDA)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import figure, kpi_row, load_clean, rate_bar, setup_page  # noqa: E402

import streamlit as st  # noqa: E402

from src.eda import default_rate_table, insight_line  # noqa: E402

setup_page("Credit Risk Analysis", "Which borrower and loan characteristics drive 36-month default?")
df = load_clean()

DRIVERS = {
    "Loan grade (LendingClub)": "grade", "Sub-grade (LendingClub)": "sub_grade",
    "Interest rate": "int_rate", "Annual income": "annual_inc", "Loan amount": "loan_amnt",
    "Debt-to-income": "dti", "Employment length": "emp_length_years",
    "Home ownership": "home_ownership", "Loan purpose": "purpose",
    "Credit history length": "credit_history_years", "Revolving utilisation": "revol_util",
    "Inquiries (6 months)": "inq_last_6mths", "Term (months)": "term_months",
    "Income verification": "verification_status", "Loan-to-income": "loan_to_income",
}

f1, f2, f3, f4 = st.columns(4)
all_years = sorted(df["issue_year"].unique())
years = f1.multiselect("Issue year", all_years, default=all_years)
terms = f2.multiselect("Term (months)", [36, 60], default=[36, 60])
sample = f3.selectbox("Sample", ["All loans", "Train", "Test"])
driver_label = f4.selectbox("Risk driver", list(DRIVERS))

view = df[df["issue_year"].isin(years) & df["term_months"].isin(terms)]
if sample != "All loans":
    view = view[view["sample"] == sample.lower()]
if view.empty:
    st.warning("No loans match these filters. Select at least one issue year and one term.")
    st.stop()

kpi_row([
    ("Loans", f"{len(view):,}"),
    ("Defaults (36 months)", f"{int(view['default'].sum()):,}"),
    ("Default rate", f"{view['default'].mean():.2%}"),
    ("Exposure", f"${view['loan_amnt'].sum() / 1e6:,.1f}M"),
])

col = DRIVERS[driver_label]
table = default_rate_table(view, col)
table["group"] = table["group"].astype(str)
order = table["group"].tolist()

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Default rate by {driver_label}")
    st.altair_chart(rate_bar(table, "group", "default_rate_pct", driver_label, "Default rate (%)",
                             average=view["default"].mean() * 100, sort=order), use_container_width=True)
    st.caption("Dashed line: average default rate of the selected loans.")
with right:
    st.subheader("Interpretation")
    st.markdown(insight_line(table, driver_label))
    st.caption("Groups with less than 1% of the selected loans are ignored in this sentence.")
    st.dataframe(table, hide_index=True, use_container_width=True)

if col in ("grade", "sub_grade", "int_rate"):
    st.info("Grade, sub-grade and interest rate are LendingClub's own risk assessment. They are shown "
            "for analysis and used as a benchmark, but are deliberately excluded from the models.")

st.divider()
st.subheader("Multivariate and time views (training sample)")
a, b = st.columns(2)
with a:
    figure("eda", "21_grade_x_term", "Default rate by grade and term")
    figure("eda", "20_correlation_heatmap", "Correlation of numeric features")
with b:
    figure("eda", "22_income_x_dti", "Default rate by income and DTI")
    figure("eda", "24_default_by_issue_quarter", "Default rate by issue quarter (all loans)")
