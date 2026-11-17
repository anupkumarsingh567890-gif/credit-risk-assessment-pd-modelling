"""
Data dictionary for the LendingClub 2007-2011 loan file.

Each populated column is tagged with:
  * description  - what the field means (from LendingClub's published dictionary)
  * group        - business grouping used throughout the project
  * timing       - WHEN the value becomes known, which decides leakage:
        Identifier        : row keys, no predictive meaning
        Application       : supplied by the borrower or pulled from the credit
                            bureau when the application is submitted
        Origination       : set by LendingClub / investors when the loan is
                            priced or funded (known at issuance, not at application)
        Post-origination  : produced by repayment behaviour after issuance -> LEAKAGE
        Outcome           : the label itself
  * recommendation - Phase 1 view on how the column should be treated

The 54 fully-empty columns are bureau/joint-application fields that LendingClub
introduced after 2011; they are listed with a generic description.
"""
import pandas as pd

from src.config import STATUS_COL

ID, APP, ORIG, POST, OUT = (
    "Identifier",
    "Application",
    "Origination",
    "Post-origination",
    "Outcome",
)

# column: (description, group, timing)
COLUMN_INFO: dict[str, tuple[str, str, str]] = {
    # ---- identifiers -------------------------------------------------------
    "id": ("Unique LendingClub loan listing ID", "Identifier", ID),
    "member_id": ("Unique LendingClub borrower ID", "Identifier", ID),
    "url": ("Web link to the loan listing", "Identifier", ID),
    # ---- loan terms ----------------------------------------------------------
    "loan_amnt": ("Loan amount requested by the borrower (USD)", "Loan terms", APP),
    "funded_amnt": ("Amount committed to the loan at issuance; can be lower than requested", "Loan terms", ORIG),
    "funded_amnt_inv": ("Portion of the funded amount committed by investors", "Loan terms", ORIG),
    "term": ("Number of monthly payments: 36 or 60", "Loan terms", APP),
    "purpose": ("Borrower-selected loan purpose category", "Loan terms", APP),
    "title": ("Loan title typed by the borrower (free text)", "Loan terms", APP),
    "desc": ("Loan description typed by the borrower (free text)", "Loan terms", APP),
    "issue_d": ("Month the loan was funded (issue date)", "Loan terms", ORIG),
    # ---- LendingClub pricing -------------------------------------------------
    "grade": ("LendingClub-assigned risk grade A (best) to G (worst)", "LC pricing", ORIG),
    "sub_grade": ("LendingClub-assigned sub-grade A1 to G5", "LC pricing", ORIG),
    "int_rate": (
        "Interest rate set by LendingClub from the sub-grade (stored as text, e.g. '10.65%')",
        "LC pricing", ORIG),
    "installment": ("Monthly payment owed; a function of funded amount, rate and term", "LC pricing", ORIG),
    # ---- borrower profile ----------------------------------------------------
    "emp_title": ("Job title supplied by the borrower (free text)", "Borrower", APP),
    "emp_length": ("Employment length: '< 1 year' to '10+ years'", "Borrower", APP),
    "home_ownership": ("Home ownership: RENT, OWN, MORTGAGE, OTHER, NONE", "Borrower", APP),
    "annual_inc": ("Self-reported annual income (USD)", "Borrower", APP),
    "verification_status": (
        "Whether LendingClub verified the income: Verified, Source Verified, Not Verified",
        "Borrower", APP),
    "zip_code": ("First 3 digits of the borrower's zip code", "Borrower", APP),
    "addr_state": ("Borrower's US state", "Borrower", APP),
    "dti": (
        "Debt-to-income: monthly debt payments (excl. mortgage and this loan) / monthly income x 100",
        "Borrower", APP),
    # ---- credit bureau at application ---------------------------------------
    "delinq_2yrs": ("Number of 30+ days-past-due incidents in the last 2 years", "Credit bureau", APP),
    "earliest_cr_line": ("Month the borrower's earliest credit line was opened", "Credit bureau", APP),
    "inq_last_6mths": ("Credit inquiries in the last 6 months (excl. auto and mortgage)", "Credit bureau", APP),
    "mths_since_last_delinq": (
        "Months since the last delinquency; blank usually means none on file",
        "Credit bureau", APP),
    "mths_since_last_record": (
        "Months since the last public record; blank usually means none on file",
        "Credit bureau", APP),
    "open_acc": ("Number of open credit lines", "Credit bureau", APP),
    "pub_rec": ("Number of derogatory public records", "Credit bureau", APP),
    "pub_rec_bankruptcies": ("Number of public-record bankruptcies", "Credit bureau", APP),
    "revol_bal": ("Total revolving credit balance (USD)", "Credit bureau", APP),
    "revol_util": (
        "Revolving utilisation: balance / total revolving limit (stored as text, e.g. '83.7%')",
        "Credit bureau", APP),
    "total_acc": ("Total number of credit lines ever in the file", "Credit bureau", APP),
    "collections_12_mths_ex_med": ("Collections in the last 12 months, excluding medical", "Credit bureau", APP),
    "acc_now_delinq": ("Number of accounts currently delinquent", "Credit bureau", APP),
    "chargeoff_within_12_mths": ("Charge-offs in the last 12 months", "Credit bureau", APP),
    "delinq_amnt": ("Past-due amount owed on delinquent accounts (USD)", "Credit bureau", APP),
    "tax_liens": ("Number of tax liens", "Credit bureau", APP),
    # ---- platform / admin ----------------------------------------------------
    "policy_code": ("1 = loan under publicly available credit policy", "Platform", ORIG),
    "application_type": ("Individual or joint application", "Platform", APP),
    "initial_list_status": ("Initial listing status: w = whole, f = fractional", "Platform", ORIG),
    # ---- repayment performance (post-origination) ---------------------------
    "pymnt_plan": ("Whether a hardship payment plan has been put in place", "Repayment", POST),
    "out_prncp": ("Outstanding principal at the snapshot date", "Repayment", POST),
    "out_prncp_inv": ("Outstanding principal funded by investors", "Repayment", POST),
    "total_pymnt": ("Total payments received to date", "Repayment", POST),
    "total_pymnt_inv": ("Total payments received to date, investor portion", "Repayment", POST),
    "total_rec_prncp": ("Principal received to date", "Repayment", POST),
    "total_rec_int": ("Interest received to date", "Repayment", POST),
    "total_rec_late_fee": ("Late fees received to date", "Repayment", POST),
    "recoveries": ("Gross recoveries after charge-off", "Repayment", POST),
    "collection_recovery_fee": ("Collection fee on post-charge-off recoveries", "Repayment", POST),
    "last_pymnt_d": ("Month the last payment was received", "Repayment", POST),
    "last_pymnt_amnt": ("Amount of the last payment received", "Repayment", POST),
    "next_pymnt_d": ("Next scheduled payment month", "Repayment", POST),
    "last_credit_pull_d": ("Most recent month LendingClub pulled the borrower's credit", "Repayment", POST),
    # ---- outcome -------------------------------------------------------------
    STATUS_COL: ("Loan status at the snapshot: Fully Paid, Charged Off, Current", "Outcome", OUT),
}

# Columns that need a specific treatment decision rather than a generic rule
SPECIAL_RECOMMENDATIONS = {
    "issue_d": "Keep as split / vintage key - not a predictor",
    "funded_amnt": "Exclude - set at issuance; near-duplicate of loan_amnt",
    "funded_amnt_inv": "Exclude - investor funding, unknown at application",
    "grade": "Decision needed - LendingClub's own risk output",
    "sub_grade": "Decision needed - LendingClub's own risk output",
    "int_rate": "Decision needed - LendingClub's own risk output",
    "installment": "Decision needed - derived from int_rate and amount",
    "emp_title": "Exclude - free text, ~28.8k unique values",
    "title": "Exclude - free text, duplicates `purpose`",
    "desc": "Exclude - free text, 33% missing",
    "zip_code": "Exclude - fair-lending proxy risk, 823 levels",
    "addr_state": "Candidate predictor - use with caution (geographic)",
    STATUS_COL: "Target source",
}

MASKED_EXAMPLES = {"id", "member_id", "url", "desc", "title", "emp_title", "zip_code"}

EMPTY_DESCRIPTION = (
    "Field not populated for 2007-2011 loans (introduced by LendingClub later)"
)


def _recommend(col: str, timing: str, missing_pct: float, n_unique: int) -> str:
    if missing_pct == 100:
        return "Drop - 100% missing"
    if col in SPECIAL_RECOMMENDATIONS:
        return SPECIAL_RECOMMENDATIONS[col]
    if timing == ID:
        return "Drop - identifier"
    if timing == POST:
        return "Drop - post-origination leakage"
    if n_unique <= 1:
        return "Drop - constant value"
    return "Candidate predictor"


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Combine static descriptions with observed statistics for every column."""
    rows = []
    for col in df.columns:
        s = df[col]
        missing_pct = round(s.isna().mean() * 100, 2)
        n_unique = int(s.nunique(dropna=True))
        desc, group, timing = COLUMN_INFO.get(col, (EMPTY_DESCRIPTION, "Not populated", "n/a"))
        # No raw values for identifiers or free text (privacy, and the raw file
        # itself is not redistributed with the repository)
        masked = col in MASKED_EXAMPLES
        example = None if masked or s.isna().all() else s.dropna().iloc[0]
        rows.append(
            {
                "column": col,
                "description": desc,
                "group": group,
                "timing": timing,
                "dtype": str(s.dtype),
                "missing_pct": missing_pct,
                "n_unique": n_unique,
                "example": example,
                "recommendation": _recommend(col, timing, missing_pct, n_unique),
            }
        )
    return pd.DataFrame(rows)
