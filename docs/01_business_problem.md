# 1. Business problem statement

> Educational project on public historical data. It is not a production lending system, credit policy or regulatory model.

## Context
LendingClub is a peer-to-peer lender. It offered unsecured personal loans of $500–$35,000 over 36 or 60 months. Between 2007 and 2011 it issued the 39,717 loans in this dataset, worth $445.6M. About 13% of them defaulted within 36 months of origination (as defined in the methodology).

## The problem
At the moment a borrower applies, the lender has to decide **whether to lend, how much and at what price**. It can only use the information it has at that point: the application and the credit bureau report. Two kinds of error have different costs:

| Error | Cost |
|---|---|
| Approve a borrower who defaults | Most of the principal is lost, recoveries are small and slow |
| Reject a borrower who would have repaid | The interest margin and the customer are lost |

## Objectives
1. **Find the factors** at origination that drive default.
2. **Build a statistical credit score**: transparent, explainable to applicants, and the industry standard (WoE logistic scorecard).
3. **Estimate the Probability of Default (PD)** for each borrower, for pricing, expected loss and limits.
4. **Compare** the statistical scorecard with machine-learning models (Decision Tree, Random Forest, XGBoost).
5. **Classify borrowers** into Low / Medium / High risk and set an **approval cut-off**, showing the trade-off between approval volume and bad rate.
6. **Validate** that the model works on *future* borrowers: out-of-time performance, stability, calibration and overfitting.
7. **Deliver an interactive tool** that credit officers and risk managers can use.

## Users and decisions supported

| User | Decision |
|---|---|
| Credit officer | Approve / refer / decline an individual application; explain the reasons |
| Credit risk manager | Set the approval cut-off; monitor risk-band mix and concentrations |
| Pricing / finance | Use PD for risk-based pricing and expected-loss provisioning |
| Model validation / audit | Review discrimination, stability, calibration and limitations |

## Success criteria
- Out-of-time Gini clearly above random (0) and competitive with LendingClub's own sub-grade.
- Monotonic, well-separated risk bands.
- Stable score distribution (PSI < 0.10) and no material overfitting.
- Every modelling choice documented and reproducible.
