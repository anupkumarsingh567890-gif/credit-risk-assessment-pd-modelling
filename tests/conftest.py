"""Shared fixtures. Synthetic data keeps most tests fast and independent of loan.csv;
tests that need the real pipeline outputs are skipped when those files are absent."""
import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN_PARQUET, MODELS_DIR


def annuity_instalment(principal: float, annual_rate_pct: float, n_months: int) -> float:
    r = annual_rate_pct / 1200
    return principal * r / (1 - (1 + r) ** -n_months)


def principal_after(principal: float, annual_rate_pct: float, n_months: int, k: int) -> float:
    """Principal repaid after k on-time instalments."""
    r = annual_rate_pct / 1200
    a = annuity_instalment(principal, annual_rate_pct, n_months)
    balance = a / r - (a / r - principal) * (1 + r) ** k
    return principal - balance


def make_loan(status: str, term: int, last_pay_mob: float | None, instalments_paid: float,
              issue: str = "2010-01-01", principal: float = 10_000, rate: float = 12.0) -> dict:
    """One synthetic loan with the columns needed to build the target."""
    issue_d = pd.Timestamp(issue)
    last = None if last_pay_mob is None else issue_d + pd.DateOffset(months=int(last_pay_mob))
    return {
        "loan_status": status,
        "issue_d": issue_d,
        "last_pymnt_d": last,
        "term_months": term,
        "funded_amnt": principal,
        "int_rate": rate,
        "installment": annuity_instalment(principal, rate, term),
        "total_rec_prncp": principal_after(principal, rate, term, instalments_paid),
    }


@pytest.fixture
def credit_sample():
    """Synthetic borrowers whose default probability rises with x1 and with category 'b'."""
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    logit = -2 + 0.9 * x1 + 0.5 * (cat == "b") - 0.2 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    return df, pd.Series(y, name="default")


@pytest.fixture(scope="session")
def clean_data():
    if not CLEAN_PARQUET.exists():
        pytest.skip("Pipeline outputs not found; run `python run_all.py` first.")
    return pd.read_parquet(CLEAN_PARQUET)


@pytest.fixture(scope="session")
def artifacts():
    if not (MODELS_DIR / "champion.json").exists():
        pytest.skip("Model files not found; run `python run_all.py` first.")
    from src.assessment import load_artifacts
    return load_artifacts()
