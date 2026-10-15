"""
Demo -- Tool 1: minimum panel so nothing imitates MeSA.

Reproduces the hand-worked example from IDEA_SOFTWARE.pdf:
  - DcitOBP3 alone: residual = 0 (a single sensor is a single equation --
    nonanal alone can be scaled to match it exactly).
  - DcitOBP3 + OBP-B: residual jumps to ~0.171 (limonene is the closest
    interferent recipe now, but it can't reach both sensors at once).
  - Adding a third, near-redundant protein (OBP-C, deliberately built to
    look like OBP-B) barely moves the residual -- diminishing returns.

Run: python demo_tool1_specificity.py
"""
from pathlib import Path

from panel_selector import (
    SensitivityMatrix,
    best_imitation_recipe,
    best_imitation_residual,
    greedy_panel_for_specificity,
)

DATA = Path(__file__).parent / "example_sensitivity.csv"


def main():
    S = SensitivityMatrix.from_csv(DATA)
    target = "MeSA"
    interferents = ["nonanal", "hexanol", "limonene"]
    noise = 0.05  # sensor can't distinguish anything closer than this

    print("=== Step by step (matches the worked example in IDEA_SOFTWARE.pdf) ===")
    for panel in (["DcitOBP3"], ["DcitOBP3", "OBP-B"], ["DcitOBP3", "OBP-B", "OBP-C"]):
        c_star, residual = best_imitation_recipe(S, panel, target, interferents)
        recipe = ", ".join(f"{c:.3f} {name}" for c, name in zip(c_star, interferents) if c > 1e-6)
        print(f"panel={panel}")
        print(f"  best fake recipe: {recipe or '(none needed / none found)'}")
        print(f"  residual = {residual:.4f}  (noise floor = {noise})")

    print("\n=== Greedy search from scratch ===")
    result = greedy_panel_for_specificity(
        S, target=target, interferents=interferents, noise=noise, min_gain=0.01
    )
    for step in result.result.trace:
        gain = "n/a (first pick)" if step.gain == float("inf") else f"{step.gain:.4f}"
        print(f"  + {step.added:<10} -> residual={step.score:.4f}  (gain={gain})")
    print(f"Final panel: {result.panel}")
    print(f"Stop reason: {result.stop_reason}")
    if result.stop_reason == "target_crossed":
        print(f"-> MeSA is safely un-fakeable by {interferents} on this panel (residual clears noise).")
    elif result.stop_reason == "library_exhausted":
        print("-> Ran out of candidate OBPs without clearing the noise floor: this library isn't enough.")
    elif result.stop_reason == "diminishing_returns":
        print("-> Stopped early: the next OBP barely helps (redundant with what's already in the panel).")


if __name__ == "__main__":
    main()
