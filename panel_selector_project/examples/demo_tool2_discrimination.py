"""
Demo -- Tool 2: which OBPs best separate healthy from diseased vines.

Synthetic GC-MS-style replicate profiles for 4 compounds:
  - MeSA: up in diseased plants
  - C6_aldehyde: down in diseased plants
  - fungal_alcohol: appears (low in healthy, higher in diseased)
  - unrelated_compound: no difference between classes at all

OBP-Decoy binds unrelated_compound very tightly (s=0.5, far stronger than
any other OBP/compound pair in the table) but that compound doesn't move
with disease state -- so despite being the "strongest" binder available,
the greedy search should not need it, and should instead prefer
DcitOBP3/OBP-B, which are sensitive to the compounds that actually shift.

Run: python demo_tool2_discrimination.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from panel_selector import SensitivityMatrix, greedy_panel_for_discrimination

DATA = Path(__file__).parent / "example_sensitivity.csv"


def synthetic_profiles(n=12, seed=0):
    rng = np.random.default_rng(seed)
    compounds = ["MeSA", "C6_aldehyde", "fungal_alcohol", "unrelated_compound"]

    healthy = pd.DataFrame(
        {
            "MeSA": rng.normal(1.0, 0.2, n),
            "C6_aldehyde": rng.normal(5.0, 0.5, n),
            "fungal_alcohol": rng.normal(0.1, 0.05, n),
            "unrelated_compound": rng.normal(3.0, 0.3, n),
        }
    )[compounds].clip(lower=0)

    diseased = pd.DataFrame(
        {
            "MeSA": rng.normal(4.0, 0.5, n),
            "C6_aldehyde": rng.normal(2.0, 0.4, n),
            "fungal_alcohol": rng.normal(2.0, 0.3, n),
            "unrelated_compound": rng.normal(3.0, 0.3, n),  # same as healthy on purpose
        }
    )[compounds].clip(lower=0)

    return healthy, diseased


def main():
    S = SensitivityMatrix.from_csv(DATA)
    healthy, diseased = synthetic_profiles()

    print("=== Greedy search (Fisher-ratio separation) ===")
    result = greedy_panel_for_discrimination(
        S, class_a_profiles=healthy, class_b_profiles=diseased, metric="fisher", min_gain=1.0
    )
    for step in result.result.trace:
        gain = "n/a (first pick)" if step.gain == float("inf") else f"{step.gain:.3f}"
        print(f"  + {step.added:<10} -> Fisher J={step.score:.3f}  (gain={gain})")
    print(f"Final panel: {result.panel}")
    print(f"Stop reason: {result.stop_reason}")
    print("-> Notice OBP-Decoy (strongest raw binder in the table) isn't picked: it")
    print("   only sees unrelated_compound, which doesn't differ between classes.")

    try:
        print("\n=== Cross-checking with cross-validated logistic regression ===")
        result_cv = greedy_panel_for_discrimination(
            S, class_a_profiles=healthy, class_b_profiles=diseased, metric="cv_accuracy", min_gain=0.01
        )
        for step in result_cv.result.trace:
            gain = "n/a (first pick)" if step.gain == float("inf") else f"{step.gain:.3f}"
            print(f"  + {step.added:<10} -> CV accuracy={step.score:.3f}  (gain={gain})")
        print(f"Final panel: {result_cv.panel}  (stop: {result_cv.stop_reason})")
    except NotImplementedError as e:
        print(f"(skipped: {e})")


if __name__ == "__main__":
    main()
