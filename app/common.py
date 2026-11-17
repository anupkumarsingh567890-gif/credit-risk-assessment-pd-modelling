"""Shared helpers for the Streamlit pages: page layout, cached loading, charts, explanations."""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CLEAN_PARQUET, FIGURES_DIR, MODEL_FILES, MODELS_DIR, SCORED_PORTFOLIO, TABLES_DIR, TEST_PREDICTIONS,
)

REQUIRED = [CLEAN_PARQUET, TEST_PREDICTIONS, SCORED_PORTFOLIO, MODELS_DIR / "champion.json",
            MODELS_DIR / "scorecard.joblib", TABLES_DIR / "eval_metrics.csv"]

MODEL_COLUMNS = {name: f"pd_{file}" for name, file in MODEL_FILES.items()}

BLUE = "#2a78d6"
MUTED = "#52514e"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
RISK_EXPLANATIONS = {
    "Low": "Best 40% of training applicants by credit score; the lowest observed default rate. "
           "Approve on standard terms.",
    "Medium": "Middle 40% of applicants; default rate close to the portfolio average. Approve if the "
              "score clears the approval cut-off, otherwise refer for manual review or adjust terms.",
    "High": "Worst 20% of applicants by score; the highest observed default rate. Decline, or lend "
            "only with risk-based pricing and tighter limits.",
}
DISCLAIMER = ("Educational analytics project on public LendingClub data (2007-2011). Not a production "
              "lending system, credit policy or regulatory capital model.")


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
def setup_page(title: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"Credit Risk | {title}", page_icon=":bar_chart:", layout="wide")
    st.markdown(
        """<style>
        div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e5e4df;
            border-radius:8px; padding:12px 16px;}
        div[data-testid="stMetricLabel"] p {color:#52514e; font-size:0.85rem;}
        div[data-testid="stMetricValue"] {font-size:1.7rem;}
        .risk-badge {display:inline-block; padding:4px 14px; border-radius:14px;
            color:#fff; font-weight:600;}
        </style>""",
        unsafe_allow_html=True,
    )
    missing = [p.name for p in REQUIRED if not p.exists()]
    if missing:
        st.error(f"Model outputs not found ({', '.join(missing)}). Run the pipeline first from the "
                 "project folder:\n\n`python run_all.py`")
        st.stop()
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    with st.sidebar:
        champ = champion()
        st.markdown("**Credit Risk Modelling**")
        st.caption(f"LendingClub loans 2007-2011  \nPD champion: {champ['model']}  \n"
                   "Target: default within 36 months")
        st.caption(DISCLAIMER)


def kpi_row(items: list[tuple]) -> None:
    """items: (label, value) or (label, value, help text)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *help_text = item
        col.metric(label, value, help=help_text[0] if help_text else None)


def risk_badge(band: str) -> str:
    return f'<span class="risk-badge" style="background:{RISK_COLORS[band]}">{band} risk</span>'


# --------------------------------------------------------------------------- #
# Cached loading
# --------------------------------------------------------------------------- #
@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_PARQUET)


@st.cache_data
def load_portfolio() -> pd.DataFrame:
    return pd.read_parquet(SCORED_PORTFOLIO)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(TEST_PREDICTIONS)


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = TABLES_DIR / f"{name}.csv"
    if not path.exists():
        st.warning(f"Table `{name}.csv` not found; re-run `python run_all.py`.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_artifacts() -> dict:
    from src.assessment import load_artifacts as _load
    try:
        return _load()
    except Exception as exc:  # show a readable message instead of a stack trace
        st.error(f"Could not load the models: {exc}")
        st.stop()


def champion() -> dict:
    return json.loads((MODELS_DIR / "champion.json").read_text())


def figure(subdir: str, name: str, caption: str | None = None) -> None:
    path = FIGURES_DIR / subdir / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Chart `{name}.png` not found; re-run `python run_all.py`.")


# --------------------------------------------------------------------------- #
# Charts (Altair, consistent style)
# --------------------------------------------------------------------------- #
def rate_bar(table: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
             average: float | None = None, horizontal: bool = False, sort=None) -> alt.Chart:
    """Single-colour bar chart with tooltips and an optional dashed average line."""
    angle = 0 if table[x].astype(str).str.len().max() * len(table) < 90 else -35
    if horizontal:
        cat = alt.Y(f"{x}:N", title=x_title, sort=sort)
        val = alt.X(f"{y}:Q", title=y_title)
    else:
        cat = alt.X(f"{x}:N", title=x_title, sort=sort, axis=alt.Axis(labelAngle=angle))
        val = alt.Y(f"{y}:Q", title=y_title)
    bars = alt.Chart(table).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(cat, val, tooltip=list(table.columns))
    if average is None:
        return bars.properties(height=320)
    line = alt.Chart(pd.DataFrame({"average": [average]})).mark_rule(color=MUTED, strokeDash=[5, 4], size=1.5)
    line = line.encode(x="average:Q") if horizontal else line.encode(y="average:Q")
    return (bars + line).properties(height=320)


def band_bar(table: pd.DataFrame, y: str, y_title: str) -> alt.Chart:
    """Bars coloured by risk band (always labelled, never colour alone)."""
    return alt.Chart(table).mark_bar(cornerRadiusEnd=3).encode(
        alt.X("risk_category:N", title="Risk band", sort=["Low", "Medium", "High"],
              axis=alt.Axis(labelAngle=0)),
        alt.Y(f"{y}:Q", title=y_title),
        alt.Color("risk_category:N", scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values())),
                  legend=None),
        tooltip=list(table.columns),
    ).properties(height=300)
