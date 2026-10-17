"""
Sanity tests: check the module reproduces the hand-worked numbers from
IDEA_SOFTWARE.pdf, and that the greedy loop's three stop conditions behave
as described.
"""
import math
from pathlib import Path

import pandas as pd
import pytest

from panel_selector import (
    SensitivityMatrix,
    best_imitation_residual,
    greedy_panel_for_specificity,
    greedy_panel_for_discrimination,
    fisher_ratio,
)

DATA = Path(__file__).parent.parent / "examples" / "example_sensitivity.csv"


@pytest.fixture
def S():
    return SensitivityMatrix.from_csv(DATA)


def test_single_sensor_residual_is_zero(S):
    # "with a single protein the residual was 0" -- one equation, one degree
    # of freedom in nonanal is enough to hit it exactly.
    r = best_imitation_residual(S, ["DcitOBP3"], "MeSA", ["nonanal", "hexanol", "limonene"])
    assert r == pytest.approx(0.0, abs=1e-9)


def test_two_sensor_residual_matches_worked_example(S):
    # "with two it jumped to 0.17" (text gives 0.1718 with an unrounded x*).
    r = best_imitation_residual(S, ["DcitOBP3", "OBP-B"], "MeSA", ["nonanal", "hexanol", "limonene"])
    assert r == pytest.approx(0.1718, abs=0.005)


def test_third_redundant_sensor_barely_helps(S):
    # "with all three, 0.1731 -- practically no gain."
    r2 = best_imitation_residual(S, ["DcitOBP3", "OBP-B"], "MeSA", ["nonanal", "hexanol", "limonene"])
    r3 = best_imitation_residual(S, ["DcitOBP3", "OBP-B", "OBP-C"], "MeSA", ["nonanal", "hexanol", "limonene"])
    assert r3 - r2 < 0.01


def test_greedy_specificity_stops_on_target_crossed(S):
    result = greedy_panel_for_specificity(
        S, target="MeSA", interferents=["nonanal", "hexanol", "limonene"], noise=0.05, min_gain=0.01
    )
    assert result.panel == ["DcitOBP3", "OBP-B"]
    assert result.stop_reason == "target_crossed"


def test_greedy_specificity_reports_library_exhausted_when_never_safe(S):
    # An impossibly strict noise floor -- no panel of these 4 OBPs can clear it.
    result = greedy_panel_for_specificity(
        S, target="MeSA", interferents=["nonanal", "hexanol", "limonene"], noise=999.0, min_gain=-1
    )
    assert result.stop_reason in ("library_exhausted", "diminishing_returns")


def test_fisher_ratio_zero_when_classes_identical():
    df_a = pd.DataFrame({"x": [1.0, 1.1, 0.9]})
    df_b = pd.DataFrame({"x": [1.0, 1.05, 0.95]})
    import numpy as np

    assert fisher_ratio(df_a.values, df_b.values) == pytest.approx(0.0, abs=0.5)


def test_greedy_discrimination_avoids_pure_decoy(S):
    import numpy as np

    rng = np.random.default_rng(1)
    compounds = ["MeSA", "C6_aldehyde", "fungal_alcohol", "unrelated_compound"]
    healthy = pd.DataFrame(
        {
            "MeSA": rng.normal(1.0, 0.2, 10),
            "C6_aldehyde": rng.normal(5.0, 0.5, 10),
            "fungal_alcohol": rng.normal(0.1, 0.05, 10),
            "unrelated_compound": rng.normal(3.0, 0.3, 10),
        }
    )[compounds]
    diseased = pd.DataFrame(
        {
            "MeSA": rng.normal(4.0, 0.5, 10),
            "C6_aldehyde": rng.normal(2.0, 0.4, 10),
            "fungal_alcohol": rng.normal(2.0, 0.3, 10),
            "unrelated_compound": rng.normal(3.0, 0.3, 10),
        }
    )[compounds]

    result = greedy_panel_for_discrimination(
        S, class_a_profiles=healthy, class_b_profiles=diseased, metric="fisher", min_gain=0.05
    )
    assert "OBP-Decoy" not in result.panel


def test_start_with_forces_first_obp_then_finds_complement(S):
    # Start from OBP-B (as if SCENTINEL v3 had chosen it): the search must keep
    # it first and add DcitOBP3 as the complement, reaching the same 0.17 residual.
    res = greedy_panel_for_specificity(
        S, target="MeSA", interferents=["nonanal", "hexanol", "limonene"],
        noise=0.05, initial=["OBP-B"],
    )
    assert res.panel[0] == "OBP-B"
    assert res.panel == ["OBP-B", "DcitOBP3"]
    assert math.isnan(res.result.trace[0].gain)
    assert res.stop_reason == "target_crossed"
    assert res.residual_trace[-1] == pytest.approx(0.1718, abs=0.005)


def test_start_with_does_not_add_duplicate(S):
    res = greedy_panel_for_specificity(
        S, target="MeSA", interferents=["nonanal", "hexanol", "limonene"],
        noise=0.05, initial=["DcitOBP3", "DcitOBP3"],
    )
    assert res.panel.count("DcitOBP3") == 1
