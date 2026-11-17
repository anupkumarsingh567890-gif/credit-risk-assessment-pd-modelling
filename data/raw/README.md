# Raw data (not included in the repository)

Place the LendingClub 2007-2011 loan file here as **`data/raw/loan.csv`**.

**How to obtain it**
1. Search Kaggle for **"Lending Club Loan Dataset 2007-2011"** (the `loan.csv` file used in the widely
   published *Lending Club case study*). LendingClub originally released this data publicly as its
   2007-2011 loan statistics file.
2. Download `loan.csv` and copy it to this folder.

**Check it is the right file**: 39,717 rows x 111 columns, loans issued Jun-2007 to Dec-2011,
`loan_status` values Fully Paid (32,950), Charged Off (5,627) and Current (1,140).
Phase 1 (`python -m scripts.run_phase1`) validates the columns and prints a warning if the shape differs.

The data is not redistributed here; check the licence / terms of the source you download it from.
