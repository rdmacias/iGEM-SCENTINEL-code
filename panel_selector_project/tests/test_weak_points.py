"""Weak points: single-cell changes that bring the residual back under noise."""
import numpy as np
import pandas as pd
import pytest

from panel_selector import (
    SensitivityMatrix,
    best_imitation_residual,
    residual_with_cell,
    breaking_ki_range,
    measured_tolerance,
)
from panel_selector.specificity import DEFAULT_KI_GRID

PANEL = ["A", "B"]
INTERF = ["x", "y", "z"]


@pytest.fixture
def S():
    # target t = (1, 0.1); x = (1, 1) measured on both.
    # y: unknown on A, 1 on B.   z: unknown on A, 0 on B (invisible on B).
    df = pd.DataFrame({"t": [1.0, 0.1], "x": [1.0, 1.0], "y": [np.nan, 1.0], "z": [np.nan, 0.0]},
                      index=PANEL)
    return SensitivityMatrix(df)


def test_baseline_passes(S):
    # only x can imitate: best c = 0.55, residual = |(0.45, -0.45)| = 0.636
    assert best_imitation_residual(S, PANEL, "t", INTERF) == pytest.approx(0.45 * np.sqrt(2), rel=1e-6)


def test_residual_with_cell_unknown_strong_binder_breaks(S):
    # y binding A very strongly (Ki 0.01 -> s 100) lets y + x fit t almost exactly
    assert residual_with_cell(S, PANEL, "t", INTERF, "A", "y", 0.01) < 0.05
    # very weak binding: back to the baseline
    assert residual_with_cell(S, PANEL, "t", INTERF, "A", "y", 1e6) == pytest.approx(0.636, abs=1e-3)


def test_breaking_range_is_low_ki_only(S):
    lo, hi = breaking_ki_range(S, PANEL, "t", INTERF, "A", "y", noise=0.05)
    assert lo == pytest.approx(DEFAULT_KI_GRID[0])
    assert hi < 1.0                         # only strong binding of y to A breaks it


def test_invisible_interferent_breaks_at_any_ki(S):
    # z gives 0 on B, so any signal on A can be scaled up to fit t's A component
    lo, hi = breaking_ki_range(S, PANEL, "t", INTERF, "A", "z", noise=0.05)
    assert lo == pytest.approx(DEFAULT_KI_GRID[0]) and hi == pytest.approx(DEFAULT_KI_GRID[-1])


def test_cell_that_cannot_break(S):
    # x on B: whatever its value, t keeps a component no mixture reaches? check no false alarm
    lo, hi = breaking_ki_range(S, PANEL, "t", ["x"], "B", "x", noise=0.05)
    # x = (1, s): fits t = (1, 0.1) exactly when s = 0.1 -> Ki = 10 breaks; others do not
    assert lo is not None and lo <= 10 <= hi and hi < 100


def test_measured_tolerance(S):
    # with noise 0.6 the margin is thin: residual = (s_tA - 0.1)/sqrt(2) with only x,
    # it drops under 0.6 when s_tA < 0.9485, i.e. Ki_tA up by ~5.4 %
    up_down = measured_tolerance(S, PANEL, "t", ["x"], "A", "t", noise=0.6)
    down, up = up_down
    assert down is None
    assert up == pytest.approx(1 / 0.9485 - 1, abs=0.01)


def test_bet_text_wording():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).parent.parent / "scentinel_tool1_specificity.py"
    spec = importlib.util.spec_from_file_location("tool1", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    g = DEFAULT_KI_GRID
    assert "does NOT bind" in m._bet_text("x", "A", g[0], g[-1], g)
    assert "Ki > 16" in m._bet_text("x", "A", g[0], 16.0, g)
    assert "Ki < 20" in m._bet_text("x", "A", 20.0, g[-1], g)
    assert "below 14" in m._bet_text("x", "A", 14.0, 50.0, g)


def test_complete_coverage_adds_obp_that_knows_the_invisible_interferent():
    from panel_selector import complete_coverage
    # z has no value on A or B; only C knows it
    df = pd.DataFrame({"t": [1.0, 0.1, 0.5], "x": [1.0, 1.0, 0.2], "z": [np.nan, np.nan, 0.3]},
                      index=["A", "B", "C"])
    S = SensitivityMatrix(df)
    added, uncovered = complete_coverage(S, ["A", "B"], "t", ["x", "z"], ["A", "B", "C"])
    assert [o for o, _, _ in added] == ["C"] and added[0][1] == ["z"] and uncovered == []
    # residual never goes down when an OBP is added
    assert added[0][2] >= best_imitation_residual(S, ["A", "B"], "t", ["x", "z"]) - 1e-12


def test_complete_coverage_reports_unknowable():
    from panel_selector import complete_coverage
    df = pd.DataFrame({"t": [1.0, 0.1], "x": [1.0, 1.0], "z": [np.nan, np.nan]}, index=["A", "B"])
    added, uncovered = complete_coverage(SensitivityMatrix(df), ["A"], "t", ["x", "z"], ["A", "B"])
    assert added == [] and uncovered == ["z"]


def test_pick_key_prefers_better_measured_useful_obp():
    from panel_selector import greedy_select
    # score: B raises more than C, both above min_gain; pick_key prefers C
    gains = {"B": 0.5, "C": 0.2, "D": 0.0}
    score = lambda panel: sum(gains.get(o, 0.0) for o in panel)
    plain = greedy_select(["B", "C", "D"], score, target_score=0.1, initial=["A"])
    keyed = greedy_select(["B", "C", "D"], score, target_score=0.1, initial=["A"],
                          pick_key=lambda o, g, s: {"B": 0.1, "C": 0.9, "D": 1.0}[o])
    assert plain.panel == ["A", "B"]
    assert keyed.panel == ["A", "C"]        # D has the best key but adds nothing -> not eligible


def test_tool2_hidden_noisy_compound_lowers_J():
    from panel_selector import summary_j_function, j_cell_scan
    # one OBP reads MeSA (changes 1 -> 3); limonene (5 +/- 2 in both) unknown on it
    comps = ["MeSA", "Lim"]
    A = pd.DataFrame({"mean": [1, 5], "sd": [0.2, 2]}, index=comps)
    B = pd.DataFrame({"mean": [3, 5], "sd": [0.5, 2]}, index=comps)
    J = summary_j_function(A, B, comps, sensor_noise=0.0)
    A0 = np.array([[6.0, 0.0]])
    J0 = J(A0)
    assert J0 == pytest.approx(144 / (36 * 0.04 + 36 * 0.25))       # 12^2 / (1.44 + 9) = 13.8
    Js = j_cell_scan(J, A0, 0, 1, np.array([0.0, 5.0]))
    assert Js[0] == pytest.approx(J0)
    assert Js[1] == pytest.approx(144 / (1.44 + 9 + 2 * 25 * 4))     # limonene noise 5*2 per class -> 0.68


def test_error_from_j_matches_table():
    from panel_selector import error_from_j, j_for_error
    assert error_from_j(1.0) == pytest.approx(0.24, abs=0.005)
    assert error_from_j(4.0) == pytest.approx(0.079, abs=0.002)
    assert j_for_error(0.05) == pytest.approx(5.41, abs=0.01)
    assert error_from_j(j_for_error(0.05)) == pytest.approx(0.05)
