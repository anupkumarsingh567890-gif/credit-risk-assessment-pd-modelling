"""
Phase 8 - Borrower assessment demo.

Run from the project root:   python -m scripts.run_phase8

Scores the example application plus three real test-sample borrowers (one
typical borrower from each risk band) and shows the actual outcome next to
the assessment.
Output: reports/tables/assessment_examples.csv, reports/phase8_borrower_assessment.md
"""
import pandas as pd

from src.assessment import EXAMPLE_APPLICATION, assess_borrower, assess_features, load_artifacts
from src.config import ALL_FEATURES, REPORTS_DIR, TARGET_COL
from src.utils import load_clean, md_table, save_table, train_test


def main() -> None:
    art = load_artifacts()
    _, test = train_test(load_clean())

    example = assess_borrower(EXAMPLE_APPLICATION, art)
    print("Example application:")
    for k, v in example.items():
        print(f"  {k}: {v}")

    scored = assess_features(test[ALL_FEATURES], art)
    scored = scored.join(test[["id", "grade", TARGET_COL]])
    picks = []
    for band in ["Low", "Medium", "High"]:
        g = scored[scored["risk_category"] == band].sort_values("credit_score")
        picks.append(g.iloc[len(g) // 2])  # median borrower of the band
    picks = pd.DataFrame(picks)
    picks["probability_of_default"] = (picks["probability_of_default"] * 100).round(2)
    picks = picks.rename(columns={"probability_of_default": "pd_pct", TARGET_COL: "actual_default"})
    cols = ["id", "credit_score", "pd_pct", "risk_category", "decision", "main_reasons", "grade",
            "actual_default"]
    save_table(picks[cols], "assessment_examples")

    ex = pd.DataFrame([{k: v for k, v in example.items() if k != "points_breakdown"}])
    ex["probability_of_default"] = (ex["probability_of_default"] * 100).round(2)
    breakdown = pd.DataFrame(list(example["points_breakdown"].items()), columns=["characteristic", "points"])
    report = [
        "# Phase 8 - Borrower assessment\n",
        f"Champion PD model: **{art['champion']['model']}**. Approval cut-off score: "
        f"{art['card']['approval_cutoff']:.0f}.\n",
        "## Example application\n", md_table(pd.DataFrame([EXAMPLE_APPLICATION]).T.reset_index()
                                             .rename(columns={"index": "input", 0: "value"})),
        "\n### Result\n", md_table(ex.rename(columns={"probability_of_default": "pd_pct"})),
        "\n### Points by characteristic\n", md_table(breakdown),
        "\n## Typical test-sample borrower in each risk band\n", md_table(picks[cols]),
    ]
    (REPORTS_DIR / "phase8_borrower_assessment.md").write_text("\n".join(report), encoding="utf-8")
    print(md_table(picks[cols]))


if __name__ == "__main__":
    main()
