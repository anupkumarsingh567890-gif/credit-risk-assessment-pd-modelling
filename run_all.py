"""
Run the complete credit risk pipeline, Phase 1 to Phase 10.

From the project folder (in the VS Code terminal):
    python run_all.py              # everything, about 4-6 minutes
    python run_all.py --from 4     # restart from Phase 4 (earlier outputs must exist)

Then start the app:
    streamlit run app/Overview.py
"""
import argparse
import time

from scripts import (
    run_phase1, run_phase2, run_phase3, run_phase4, run_phase5, run_phase6, run_phase7,
    run_phase8, run_phase9, run_phase10,
)

PHASES = [
    (1, "Data understanding", run_phase1.main),
    (2, "Data preprocessing", run_phase2.main),
    (3, "Exploratory data analysis", run_phase3.main),
    (4, "Scorecard development", run_phase4.main),
    (5, "Risk classification", run_phase5.main),
    (6, "PD model development (slowest step)", run_phase6.main),
    (7, "Model evaluation", run_phase7.main),
    (8, "Borrower assessment", run_phase8.main),
    (9, "Portfolio risk analysis", run_phase9.main),
    (10, "Model validation", run_phase10.main),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the credit risk pipeline")
    parser.add_argument("--from", dest="start", type=int, default=1, help="first phase to run")
    args = parser.parse_args()

    total = time.time()
    for number, name, run in PHASES:
        if number < args.start:
            continue
        print(f"\n{'=' * 70}\nPhase {number}: {name}\n{'=' * 70}", flush=True)
        start = time.time()
        run()
        print(f"Phase {number} finished in {time.time() - start:.0f}s", flush=True)
    print(f"\nAll done in {(time.time() - total) / 60:.1f} minutes.")
    print("Start the app with:  streamlit run app/Overview.py")


if __name__ == "__main__":
    main()
