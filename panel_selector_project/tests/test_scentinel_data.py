"""
Tests for the SCENTINEL adapter: Tool 1 / Tool 2 reading the same CSVs as
main.py (Compound_OBP_binding*.csv + tanimoto_imputation_diagnostic.csv).
Uses a tiny synthetic database written to tmp_path, plus an optional smoke
test on the real files if they sit in ./data.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panel_selector import (
    ScentinelData,
    CompoundAmbiguous,
    CompoundNotFound,
    convert_ki_to_float,
    diag_confidence,
    greedy_panel_for_specificity,
    greedy_panel_for_discrimination,
    greedy_select,
)

HEADER = ["CAS-number", "Compound name", "OBP_A", "OBP_B", "OBP_C"]
ROWS_EXP = [
    ["119-36-8", "methyl salicylate", "5", "", ">20"],
    ["66-25-1", "hexanal", "", "10", "8"],
    ["78-70-6", "3,7-dimethyl-1,6-octadien-3-ol / DL linalool", "40", "", ""],
    ["7212-44-4", "(+/-) nerolidol", "", "", "12"],
]


def _write(tmp_path, name, rows):
    pd.DataFrame(rows, columns=HEADER).to_csv(tmp_path / name, index=False)


@pytest.fixture
def db(tmp_path):
    _write(tmp_path, "Compound_OBP_binding.csv", ROWS_EXP)
    pred = [r[:] for r in ROWS_EXP]
    low = [r[:] for r in ROWS_EXP]
    pred[0][3], low[0][3] = "7", "5"     # methyl salicylate / OBP_B  imputed
    pred[1][2], low[1][2] = "30", "20"   # hexanal / OBP_A            imputed
    pred[2][3], low[2][3] = "9", "6"     # linalool / OBP_B           imputed, NOT in diag
    for rows in (pred, low):
        rows[0][4] = "22.0"              # '>20' resolved numerically in the imputed CSVs
    _write(tmp_path, "Compound_OBP_binding_imputed_pred.csv", pred)
    _write(tmp_path, "Compound_OBP_binding_imputed_lower.csv", low)
    pd.DataFrame({
        "row_idx": [0, 1, 3],
        "OBP": ["OBP_B", "OBP_A", "OBP_A"],          # last one points to an empty cell
        "Ki_pred_uM": [7, 30, 1], "Ki_lower_uM": [5, 20, 1],
        "T_max": [0.9, 0.4, 0.5], "sigma_log": [0.0, 0.5, 0.0], "n_donors": [3, 1, 1],
        "decision": ["confident", "exploratory", "confident"],
    }).to_csv(tmp_path / "tanimoto_imputation_diagnostic.csv", index=False)
    return ScentinelData.load(str(tmp_path))


def test_convert_ki_matches_main_rule():
    assert convert_ki_to_float(">20") == pytest.approx(22.0)
    assert convert_ki_to_float("> 30") == pytest.approx(33.0)
    assert np.isnan(convert_ki_to_float(""))


def test_diag_confidence_formula():
    d = pd.DataFrame({"T_max": [0.9], "sigma_log": [0.0], "n_donors": [3]})
    assert diag_confidence(d).iloc[0] == pytest.approx(0.9 * 1.0 * 0.75)


def test_resolve_by_cas_name_synonym(db):
    assert db.resolve("119-36-8") == "methyl salicylate"
    assert db.resolve("Methyl Salicylate") == "methyl salicylate"
    assert db.resolve("DL linalool").startswith("3,7-dimethyl")
    assert db.resolve("(+/-) nerolidol") == "(+/-) nerolidol"   # '/' inside a name is not a synonym split
    with pytest.raises(CompoundNotFound):
        db.resolve("unobtainium")


def test_ambiguous_queries_expose_numbered_choices(db):
    with pytest.raises(CompoundAmbiguous) as exc:
        db.resolve("methyl")
    assert "methyl" in str(exc.value)
    assert len(exc.value.matches) > 1
    assert any("methyl salicylate" in m.lower() for m in exc.value.matches)


def test_short_names(db):
    assert db.short(db.resolve("78-70-6")) == "DL linalool"   # long IUPAC -> common synonym
    assert db.short("(+/-) nerolidol") == "(+/-) nerolidol"


def test_measured_vs_imputed_comes_from_experimental_csv(db):
    conf, dec = db.confidence, db.decision
    assert dec.loc["methyl salicylate", "OBP_A"] == "measured"
    assert conf.loc["methyl salicylate", "OBP_B"] == pytest.approx(0.9 * 0.75)
    assert dec.loc["methyl salicylate", "OBP_B"] == "confident"
    # imputed cell with no diag row -> fallback 0.5, never "measured"
    lin = db.resolve("DL linalool")
    assert dec.loc[lin, "OBP_B"] == "no_diag"
    assert conf.loc[lin, "OBP_B"] == pytest.approx(0.5)
    assert db.diag_report["diag_on_empty_cell"] == 1
    assert db.diag_report["imputed_without_diag"] == 1


def test_sensitivity_is_transposed_inverse_ki(db):
    S = db.sensitivity("experimental").S
    assert list(S.index) == ["OBP_A", "OBP_B", "OBP_C"]
    assert S.loc["OBP_A", "methyl salicylate"] == pytest.approx(1 / 5)
    assert S.loc["OBP_C", "methyl salicylate"] == pytest.approx(1 / 22)
    assert np.isnan(S.loc["OBP_B", "methyl salicylate"])


def test_min_confidence_drops_weak_imputations_only(db):
    ki = db.ki_table("pred", min_confidence=0.6)
    assert ki.loc["methyl salicylate", "OBP_B"] == pytest.approx(7)   # c1 = 0.675 kept
    assert np.isnan(ki.loc["hexanal", "OBP_A"])                       # c1 = 0.4*0.5*0.5 dropped
    assert ki.loc["hexanal", "OBP_B"] == pytest.approx(10)            # measured always kept
    ki_c = db.ki_table("pred", only_confident=True)
    assert np.isnan(ki_c.loc[db.resolve("DL linalool"), "OBP_B"])     # 'no_diag' is not 'confident'


def test_specificity_matrix_uses_pred_for_target_lower_for_interferents(db):
    S = db.specificity_matrix("methyl salicylate", ["hexanal"], source="predicted").S
    assert S.loc["OBP_B", "methyl salicylate"] == pytest.approx(1 / 7)   # pred
    assert S.loc["OBP_A", "hexanal"] == pytest.approx(1 / 20)            # lower


def test_tool1_runs_on_scentinel_matrix(db):
    S = db.specificity_matrix("methyl salicylate", ["hexanal", db.resolve("DL linalool")], "predicted")
    res = greedy_panel_for_specificity(S, "methyl salicylate", ["hexanal", db.resolve("DL linalool")],
                                       noise=0.01, candidates=["OBP_A", "OBP_B", "OBP_C"])
    assert res.panel and res.stop_reason in {"target_crossed", "diminishing_returns", "library_exhausted"}


def test_tool2_profiles_resolve_cas_and_names(db, tmp_path):
    prof = pd.DataFrame({
        "sample": ["a1", "a2", "a3", "b1", "b2", "b3"],
        "class": ["h", "h", "h", "d", "d", "d"],
        "119-36-8": [1, 1.1, 0.9, 4, 4.2, 3.8],
        "hexanal": [3, 3.1, 2.9, 3, 3.1, 2.9],
    })
    prof.to_csv(tmp_path / "prof.csv", index=False)
    classes, errors = db.load_profiles(str(tmp_path / "prof.csv"))
    assert errors == []
    assert list(classes["h"].columns) == ["methyl salicylate", "hexanal"]
    S = db.sensitivity("experimental")
    res = greedy_panel_for_discrimination(S, classes["h"], classes["d"], min_gain=0.01)
    assert res.panel[0] in ("OBP_A", "OBP_C")   # the ones that see methyl salicylate


def test_lookahead_escapes_flat_plateau():
    # score only rises once BOTH x and y are in: plain greedy stops, lookahead=2 finds it
    def score(panel):
        return 1.0 if {"x", "y"} <= set(panel) else 0.0
    plain = greedy_select(["w", "x", "y"], score, min_gain=0.1)
    assert plain.stop_reason == "diminishing_returns" and len(plain.panel) == 1
    la = greedy_select(["w", "x", "y"], score, min_gain=0.1, lookahead=2, target_score=0.5)
    assert {"x", "y"} <= set(la.panel) and la.stop_reason == "target_crossed"


REAL = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(not (REAL / "Compound_OBP_binding.csv").is_file(), reason="real data not present")
def test_real_files_load_and_align():
    D = ScentinelData.load(str(REAL))
    assert D.has_predicted
    assert D.resolve("119-36-8") == "methyl salicylate"
    # measured values are identical in all three Ki tables
    m = D.ki_exp.notna()
    assert np.allclose(D.ki_exp.values[m.values], D.ki_pred.values[m.values], rtol=1e-3)  # imputed CSVs round to 1 decimal


def test_censored_cells_become_zero_sensitivity(db):
    # '>20' on OBP_C for methyl salicylate: keep -> 1/22, zero -> 0 (measured, no binding)
    assert db.censored.loc["methyl salicylate", "OBP_C"]
    keep = db.sensitivity("experimental").S
    zero = db.sensitivity("experimental", censored="zero").S
    assert keep.loc["OBP_C", "methyl salicylate"] == pytest.approx(1 / 22)
    assert zero.loc["OBP_C", "methyl salicylate"] == 0.0
    assert zero.loc["OBP_A", "methyl salicylate"] == pytest.approx(1 / 5)   # uncensored untouched


def test_anchored_scale_matches_s1(db):
    S = db.sensitivity("experimental", scale="anchored").S
    # Ki 5 uM -> log(40/5)/log(40) = 0.564, same as s1 in main.py
    assert S.loc["OBP_A", "methyl salicylate"] == pytest.approx(0.5637, abs=1e-3)
    # Ki 40 uM (linalool on OBP_A) -> 0 ; '>20' -> 22 uM -> 0.162 with keep, 0 with zero
    assert S.loc["OBP_A", db.resolve("DL linalool")] == pytest.approx(0.0)
    assert S.loc["OBP_C", "methyl salicylate"] == pytest.approx(np.log10(40 / 22) / np.log10(40), abs=1e-6)
    Sz = db.sensitivity("experimental", scale="anchored", censored="zero").S
    assert Sz.loc["OBP_C", "methyl salicylate"] == 0.0
    assert np.isnan(S.loc["OBP_B", "methyl salicylate"])                # unmeasured stays unknown


# ---------------- summary mode (published mean +/- SE, nothing invented) ----------------
from panel_selector import summary_moments, fisher_from_summary, greedy_panel_from_summary, SensitivityMatrix


def test_summary_single_obp_equals_hand_formula():
    S = SensitivityMatrix(pd.DataFrame({"x": [0.5], "y": [0.2]}, index=["O1"]))
    mean_a, sd_a = pd.Series({"x": 10.0, "y": 5.0}), pd.Series({"x": 2.0, "y": 1.0})
    mean_b, sd_b = pd.Series({"x": 20.0, "y": 5.0}), pd.Series({"x": 3.0, "y": 1.0})
    ma, ca = summary_moments(S, ["O1"], mean_a, sd_a)
    mb, cb = summary_moments(S, ["O1"], mean_b, sd_b)
    d = 0.5 * 10
    va = 0.25 * 4 + 0.04 * 1
    vb = 0.25 * 9 + 0.04 * 1
    assert fisher_from_summary(ma, ca, mb, cb, sensor_noise=0.0) == pytest.approx(d ** 2 / (va + vb))


def test_summary_panel_can_cancel_shared_noise():
    # toy from the notes: nonanal is noisy and unrelated to disease; OBP-1 reads a lot of it.
    S = SensitivityMatrix(pd.DataFrame({"mesa": [0.30, 0.50], "non": [0.20, 0.05], "lim": [0.10, 0.10]},
                                       index=["OBP-1", "OBP-2"]))
    A = pd.DataFrame({"mean": [1.0, 4.0, 2.0], "sd": [0.2, 1.7, 0.0]}, index=["mesa", "non", "lim"])
    B = pd.DataFrame({"mean": [4.0, 4.0, 2.0], "sd": [0.4, 1.7, 0.0]}, index=["mesa", "non", "lim"])
    def J(panel):
        ma, ca = summary_moments(S, panel, A["mean"], A["sd"]); mb, cb = summary_moments(S, panel, B["mean"], B["sd"])
        return fisher_from_summary(ma, ca, mb, cb, sensor_noise=0.01)
    assert J(["OBP-1", "OBP-2"]) > J(["OBP-2"]) > J(["OBP-1"])   # OBP-1 useless alone, useful as noise reference
    res = greedy_panel_from_summary(S, A, B, sensor_noise=0.01, min_gain=0.01)
    assert set(res.panel) == {"OBP-1", "OBP-2"}


# ---------------- input table: CAS | name | mean healthy | SE healthy | mean diseased | SE diseased ----------------
def test_input_table_minimal(db, tmp_path):
    pd.DataFrame({
        "CAS": ["119-36-8", "66-25-1", "", "123-45-6"],
        "name": ["MeSA", "hexanal", "unknown 1", "not in Ki table"],
        "mean healthy": [178.0, "n.d.", 5, 1], "SE healthy": [82.5, "n.d.", 1, 1],
        "mean diseased": ["1,449E+03", 41.6, 6, 1], "SE diseased": [231.0, "\u00ad", 1, 1],
    }).to_csv(tmp_path / "in.csv", index=False)
    H, D, rep = db.load_input(str(tmp_path / "in.csv"), n=3)
    assert list(H.index) == ["methyl salicylate", "hexanal"]
    assert H.loc["methyl salicylate", "sd"] == pytest.approx(82.5 * np.sqrt(3))
    assert D.loc["methyl salicylate", "mean"] == pytest.approx(1449)          # comma decimal
    assert H.loc["hexanal", "mean"] == 0 and D.loc["hexanal", "sd"] == 0      # n.d. -> 0
    assert len(rep["nd"]) == 3 and len(rep["skipped"]) == 2


def test_input_table_name_is_optional_cas_is_not(db, tmp_path):
    pd.DataFrame({"CAS": ["119-36-8"], "mean healthy": [1], "SE healthy": [1],
                  "mean diseased": [2], "SE diseased": [1]}).to_csv(tmp_path / "a.csv", index=False)
    H, D, _ = db.load_input(str(tmp_path / "a.csv"))
    assert list(D.index) == ["methyl salicylate"]
    pd.DataFrame({"name": ["MeSA"], "mean healthy": [1], "SE healthy": [1],
                  "mean diseased": [2], "SE diseased": [1]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="CAS"):
        db.load_input(str(tmp_path / "b.csv"))

