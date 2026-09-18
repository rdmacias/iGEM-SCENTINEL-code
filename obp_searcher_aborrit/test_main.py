"""
Unit tests for main.py - OBP Finder (iGEM URV 2025)
Run with: python -m pytest test_main.py -v
"""
 
import math
import numpy as np
import pandas as pd
import pytest
 
# Import the functions we want to test
from main import (
    DEFAULT_WEIGHTS,
    TYPE_SCORES,
    convert_ki_to_float,
    # find_voc_rows - removed, replaced by validate_voc
    read_interferent_file,
    compute_s1_affinity,
    compute_s2_selectivity,
    compute_s4_stability,
    compute_s5_promiscuity,
    compute_final_score,
    build_obp_ranking,
    select_binding_source,
    infer_s1_source_label,
)
 
 
# --------------------------------------------
# 1. convert_ki_to_float
# --------------------------------------------
 
class TestConvertKiToFloat:
 
    def test_plain_numeric_value(self):
        assert convert_ki_to_float("3.5") == pytest.approx(3.5)
 
    def test_integer_value(self):
        assert convert_ki_to_float("100") == pytest.approx(100.0)
 
    def test_value_with_greater_than_sign(self):
        # ">10" -> 10 * 1.1 = 11.0
        result = convert_ki_to_float(">10")
        assert result == pytest.approx(11.0)
 
    def test_large_value_with_greater_than_sign(self):
        # ">500" -> 550.0
        result = convert_ki_to_float(">500")
        assert result == pytest.approx(550.0)
 
    def test_nan_returns_nan(self):
        assert math.isnan(convert_ki_to_float(np.nan))
 
    def test_invalid_text_returns_nan(self):
        assert math.isnan(convert_ki_to_float("no_es_un_numero"))
 
    def test_invalid_greater_than_returns_nan(self):
        # ">abc" -> no number can be extracted -> NaN
        result = convert_ki_to_float(">abc")
        assert math.isnan(result)
 
    def test_whitespace(self):
        assert convert_ki_to_float("  7.2  ") == pytest.approx(7.2)
 
    def test_non_breaking_space(self):
        # '\xa0' is a non-breaking space
        assert convert_ki_to_float("\xa0 5.0 \xa0") == pytest.approx(5.0)
 
    def test_float_is_already_float(self):
        assert convert_ki_to_float(42.0) == pytest.approx(42.0)
 
    def test_empty_string_returns_nan(self):
        assert math.isnan(convert_ki_to_float(""))
 
    def test_zero_value_returns_zero(self):
        # 0.0 is a valid value, not NaN
        assert convert_ki_to_float("0.0") == pytest.approx(0.0)
 
    def test_none_treated_as_nan(self):
        # None is treated as NaN by pd.isna
        assert math.isnan(convert_ki_to_float(None))
 
    @pytest.mark.parametrize("val,expected", [
        ("1.0",   1.0),
        ("50",    50.0),
        (">100",  110.0),
        (">1000", 1100.0),
        ("0.5",   0.5),
    ])
    def test_parametrized_conversions(self, val, expected):
        assert convert_ki_to_float(val) == pytest.approx(expected)
 
 
# --------------------------------------------
# 2. select_binding_source
# --------------------------------------------
 
class TestSelectBindingSource:

    def test_selects_experimental_data(self, monkeypatch):
        answers = iter(["1"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        binding_file, source_label = select_binding_source()
        assert binding_file == "Compound_OBP_binding.csv"
        assert source_label == "experimental"

    def test_selects_predicted_data(self, monkeypatch):
        answers = iter(["2"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        binding_file, source_label = select_binding_source()
        assert binding_file == "Compound_OBP_binding_imputed_lower.csv"
        assert source_label == "predicted"


# --------------------------------------------
# 3. read_interferent_file
# --------------------------------------------
 
class TestReadInterferentFile:
 
    def test_reads_valid_lines(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("Ethanol\nAcetone\nBenzaldehyde\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone', 'Benzaldehyde']
 
    def test_ignores_comments(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("# comentari\nEthanol\n# altre\nAcetone\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone']
 
    def test_ignores_blank_lines(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("\nEthanol\n\nAcetone\n\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone']
 
    def test_empty_file(self, tmp_path):
        f = tmp_path / "buit.txt"
        f.write_text("", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == []
 
    def test_only_comments(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("# line 1\n# line 2\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == []
 
 
# --------------------------------------------
# 3. compute_s1_affinity
# --------------------------------------------
 
class TestInferS1SourceLabel:

    def test_predicted_value_matching_experimental_is_experimental(self):
        assert infer_s1_source_label(
            selected_value=2.0,
            experimental_value=2.0,
            binding_source_label="predicted",
        ) == "experimental"

    def test_predicted_value_without_experimental_is_new_tanimoto(self):
        assert infer_s1_source_label(
            selected_value=2.0,
            experimental_value=np.nan,
            binding_source_label="predicted",
        ) == "new (tanimoto)"

    def test_experimental_source_is_experimental(self):
        assert infer_s1_source_label(
            selected_value=2.0,
            experimental_value=np.nan,
            binding_source_label="experimental",
        ) == "experimental"


class TestComputeS1Affinity:
 
    def test_ki_equal_to_min_returns_1(self):
        # ki_target == ki_min → log_diana == log_best → s1 = 1
        s1 = compute_s1_affinity(ki_target=1.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(1.0)
 
    def test_ki_equal_to_max_returns_0(self):
        s1 = compute_s1_affinity(ki_target=100.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(0.0)
 
    def test_ki_nan_returns_0(self):
        s1 = compute_s1_affinity(ki_target=np.nan, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_zero_returns_0(self):
        s1 = compute_s1_affinity(ki_target=0.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_min_zero_returns_0(self):
        s1 = compute_s1_affinity(ki_target=5.0, ki_min_matrix=0.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_min_equal_to_max_returns_0(self):
        # denominator = 0 → retorna 0
        s1 = compute_s1_affinity(ki_target=5.0, ki_min_matrix=10.0, ki_max_matrix=10.0)
        assert s1 == 0.0
 
    def test_intermediate_value_between_0_and_1(self):
        s1 = compute_s1_affinity(ki_target=10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert 0.0 < s1 < 1.0
 
    def test_result_always_clipped_between_0_and_1(self):
        # Very large Ki (worse than the maximum in the matrix)
        s1 = compute_s1_affinity(ki_target=999999.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert 0.0 <= s1 <= 1.0
 
    def test_monotonic_decreasing_ki_increasing_s1(self):
        # The smaller the Ki, the better the affinity -> larger s1
        ki_vals = [100.0, 50.0, 10.0, 1.0]
        s1_vals = [
            compute_s1_affinity(ki, ki_min_matrix=1.0, ki_max_matrix=100.0)
            for ki in ki_vals
        ]
        assert s1_vals == sorted(s1_vals)  # creixent quan Ki decreix
 
    def test_logarithmic_symmetry(self):
        # ki_min=1, ki_max=100 -> geometric centre = sqrt(1*100) = 10 -> s1 = 0.5
        s1 = compute_s1_affinity(ki_target=10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(0.5, abs=1e-6)
 
    def test_consistent_across_different_ranges(self):
        # Same relative Ki -> same s1 regardless of the absolute range
        s1_a = compute_s1_affinity(10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        s1_b = compute_s1_affinity(100.0, ki_min_matrix=10.0, ki_max_matrix=1000.0)
        assert s1_a == pytest.approx(s1_b, abs=1e-6)
 
 
# --------------------------------------------
# 4. compute_s2_selectivity
# --------------------------------------------
 
class TestComputeS2Selectivity:
 
    def test_no_interferents_returns_neutral(self):
        # No interferent measured -> neutral score (0.5), not optimistic
        s2 = compute_s2_selectivity(ki_target=5.0, ki_min_interferent=np.nan)
        assert s2 == 0.5
 
    def test_target_ki_nan_returns_neutral(self):
        # Invalid target Ki -> neutral score (0.5)
        s2 = compute_s2_selectivity(ki_target=np.nan, ki_min_interferent=50.0)
        assert s2 == 0.5
 
    def test_much_larger_interferent_returns_1(self):
        # ratio = 1000 / (1 * 10) = 100 -> min(1, 100) = 1
        s2 = compute_s2_selectivity(ki_target=1.0, ki_min_interferent=1000.0, tau=10.0)
        assert s2 == pytest.approx(1.0)
 
    def test_interferent_equal_to_target_tau10_returns_01(self):
        # ratio = 5 / (5 * 10) = 0.1
        s2 = compute_s2_selectivity(ki_target=5.0, ki_min_interferent=5.0, tau=10.0)
        assert s2 == pytest.approx(0.1)
 
    def test_value_clipped_to_1(self):
        s2 = compute_s2_selectivity(ki_target=1.0, ki_min_interferent=999.0, tau=1.0)
        assert s2 <= 1.0
 
    def test_custom_tau(self):
        # tau=1 → ratio = 10 / (5 * 1) = 2 → min(1, 2) = 1
        s2 = compute_s2_selectivity(ki_target=5.0, ki_min_interferent=10.0, tau=1.0)
        assert s2 == pytest.approx(1.0)
 
    def test_interferent_better_than_target_is_penalized(self):
        # Ki_interf < Ki_target -> the interferent binds more strongly -> penalized
        # ratio = 1 / (10 * 10) = 0.01
        s2 = compute_s2_selectivity(ki_target=10.0, ki_min_interferent=1.0, tau=10.0)
        assert s2 == pytest.approx(0.01)
 
    @pytest.mark.parametrize("ki_d,ki_i,tau,expected", [
        (5.0,  50.0, 10.0, 1.0),   # interferent 10x weaker with tau=10 -> s2=1
        (5.0,   5.0, 10.0, 0.1),   # interferent igual amb tau=10 → s2=0.1
        (1.0, 100.0,  1.0, 1.0),   # interferent 100x weaker -> s2=1
        (10.0,  1.0,  1.0, 0.1),   # interferent 10x stronger -> s2=0.1
    ])
    def test_parametrized_selectivity(self, ki_d, ki_i, tau, expected):
        s2 = compute_s2_selectivity(ki_target=ki_d, ki_min_interferent=ki_i, tau=tau)
        assert s2 == pytest.approx(expected, abs=1e-6)
 
 
# --------------------------------------------
# 4b. compute_s4_stability
# --------------------------------------------
 
class TestComputeS4Stability:
 
    def test_classic_obp_returns_max_score(self):
        s4 = compute_s4_stability("Classic OBP")
        assert s4 == pytest.approx(1.00)
 
    def test_pbp_returns_085(self):
        s4 = compute_s4_stability("PBP")
        assert s4 == pytest.approx(0.85)
 
    def test_gobp1_returns_085(self):
        s4 = compute_s4_stability("GOBP1")
        assert s4 == pytest.approx(0.85)
 
    def test_gobp2_returns_085(self):
        s4 = compute_s4_stability("GOBP2")
        assert s4 == pytest.approx(0.85)
 
    def test_minus_c_returns_065(self):
        s4 = compute_s4_stability("Minus-C OBP")
        assert s4 == pytest.approx(0.65)
 
    def test_plus_c_returns_065(self):
        s4 = compute_s4_stability("Plus-C OBP")
        assert s4 == pytest.approx(0.65)
 
    def test_atypical_returns_045(self):
        s4 = compute_s4_stability("Atypical OBP")
        assert s4 == pytest.approx(0.45)
 
    def test_csp_returns_040(self):
        s4 = compute_s4_stability("CSP")
        assert s4 == pytest.approx(0.40)
 
    def test_unknown_type_returns_nan(self):
        # Tipus no reconegut → sense dada fiable → NaN (es descarta i renormalitza)
        s4 = compute_s4_stability("Unknown OBP Type")
        assert math.isnan(s4)

    def test_nan_returns_nan(self):
        s4 = compute_s4_stability(np.nan)
        assert math.isnan(s4)

    def test_type_with_surrounding_spaces_works(self):
        # El codi fa .strip() abans de buscar al diccionari
        s4 = compute_s4_stability("  Classic OBP  ")
        assert s4 == pytest.approx(1.00)

    def test_empty_string_returns_nan(self):
        s4 = compute_s4_stability("")
        assert math.isnan(s4)
 
    def test_result_always_between_0_and_1(self):
        for obp_type in TYPE_SCORES.keys():
            s4 = compute_s4_stability(obp_type)
            assert 0.0 <= s4 <= 1.0, f"{obp_type} gives {s4}, outside [0,1]"
 
    def test_classic_ranks_above_csp(self):
        """A Classic OBP must score higher than a CSP."""
        s4_classic = compute_s4_stability("Classic OBP")
        s4_csp     = compute_s4_stability("CSP")
        assert s4_classic > s4_csp
 
    def test_classic_ranks_above_atypical(self):
        """A Classic OBP must score higher than an Atypical."""
        s4_classic  = compute_s4_stability("Classic OBP")
        s4_atypical = compute_s4_stability("Atypical OBP")
        assert s4_classic > s4_atypical
 
    def test_minus_c_equals_plus_c(self):
        """Minus-C and Plus-C have the same score."""
        s4_minus = compute_s4_stability("Minus-C OBP")
        s4_plus  = compute_s4_stability("Plus-C OBP")
        assert s4_minus == s4_plus
 
    def test_full_hierarchy(self):
        """Check the hierarchy: Classic > PBP/GOBP > Minus/Plus-C > Atypical > CSP."""
        scores = {
            "Classic":  compute_s4_stability("Classic OBP"),
            "PBP":      compute_s4_stability("PBP"),
            "Minus_C":  compute_s4_stability("Minus-C OBP"),
            "Atypical": compute_s4_stability("Atypical OBP"),
            "CSP":      compute_s4_stability("CSP"),
        }
        assert scores["Classic"] > scores["PBP"]
        assert scores["PBP"]     > scores["Minus_C"]
        assert scores["Minus_C"] > scores["Atypical"]
        assert scores["Atypical"] > scores["CSP"]
        # Tipus no reconegut → NaN, no es pot ordenar contra la resta
        assert math.isnan(compute_s4_stability("Foo"))
 
    @pytest.mark.parametrize("obp_type,expected", [
        ("Classic OBP",  1.00),
        ("PBP",          0.85),
        ("GOBP1",        0.85),
        ("GOBP2",        0.85),
        ("Minus-C OBP",  0.65),
        ("Plus-C OBP",   0.65),
        ("Atypical OBP", 0.45),
        ("CSP",          0.40),
    ])
    def test_parametrized_all_types(self, obp_type, expected):
        assert compute_s4_stability(obp_type) == pytest.approx(expected)
 
 
# --------------------------------------------
# 5. compute_s5_promiscuity (formula C, logarithmic)
# --------------------------------------------
#
# Formula:
#   s5 = 1 - (1/N) * sum min(1, [log10(Ki_target / Ki_competitor)]+)
#
# Penalizes competitors that bind MORE STRONGLY than the target (Ki_c < Ki_target).
# Weak or equal competitors (Ki_c >= Ki_target) -> penalty 0.
# Invalid cases (no competitors, or an invalid target) -> returns 0.0.
# (has_s5_data, computed separately in build_obp_ranking, marks these cases
# as "no data" so the criterion is discarded and the weights renormalized,
# instead of counting this 0.0 as a genuinely known promiscuity.)

class TestComputeS5Promiscuity:

    # -- Edge cases: they return 0.0 (no reliable data) -----------------

    def test_no_competitors_returns_zero(self):
        # Empty Series -> no reliable data -> 0.0
        ki_competitors = pd.Series([], dtype=float)
        s5 = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)

    def test_all_competitors_nan_returns_zero(self):
        # All NaN -> no valid measurement -> 0.0
        ki_competitors = pd.Series([np.nan, np.nan, np.nan])
        s5 = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)

    def test_target_ki_nan_returns_zero(self):
        # Invalid target -> 0.0
        ki_competitors = pd.Series([1.0, 2.0, 3.0])
        s5 = compute_s5_promiscuity(ki_target=np.nan, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)

    def test_target_ki_zero_returns_zero(self):
        # Target = 0 -> 0.0
        ki_competitors = pd.Series([1.0, 2.0])
        s5 = compute_s5_promiscuity(ki_target=0.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)
 
    # -- Competitors weaker than the target -> s5 = 1 (no penalty) ------
 
    def test_weaker_competitor_is_not_penalized(self):
        # Ki_c = 100 > Ki_target = 10 -> ratio = 10/100 = 0.1 -> log < 0 -> clipped to 0
        # mean = 0 -> s5 = 1
        ki_competitors = pd.Series([100.0])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    def test_competitor_equal_to_target_is_not_penalized(self):
        # Ki_c = Ki_target -> ratio = 1 -> log10(1) = 0 -> no penalty
        ki_competitors = pd.Series([10.0])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    def test_all_weaker_competitors_returns_1(self):
        # All Ki_c > Ki_target -> no penalty -> s5 = 1
        ki_competitors = pd.Series([50.0, 100.0, 500.0])
        s5 = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    # -- Stronger competitors: penalty active ---------------------------
 
    def test_competitor_10x_stronger_gives_s5_zero(self):
        # Ki_c=1, Ki_target=10 -> ratio=10 -> log10(10)=1 -> min(1,1)=1
        # mean = 1 -> s5 = 0
        ki_competitors = pd.Series([1.0])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)
 
    def test_competitor_100x_stronger_is_capped(self):
        # Ki_c=0.1, Ki_target=10 -> ratio=100 -> log10(100)=2 -> min(1,2)=1
        # The cap cuts it to 1, same as the 10x case -> s5 = 0
        ki_competitors = pd.Series([0.1])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)
 
    def test_cap_makes_1000x_same_as_10x(self):
        # min(1,...) guarantees that extremes do not dominate the computation
        s5_10x   = compute_s5_promiscuity(ki_target=10.0, ki_competitors=pd.Series([1.0]))
        s5_1000x = compute_s5_promiscuity(ki_target=10.0, ki_competitors=pd.Series([0.01]))
        assert s5_10x == pytest.approx(s5_1000x)
 
    def test_competitor_sqrt10_gives_penalty_05(self):
        # Ki_c = Ki_target / sqrt(10) -> ratio = sqrt(10) -> log10 = 0.5
        # min(1, 0.5) = 0.5 -> mean = 0.5 -> s5 = 0.5
        ki_target = 10.0
        ki_c = ki_target / (10 ** 0.5)  # ≈ 3.162
        ki_competitors = pd.Series([ki_c])
        s5 = compute_s5_promiscuity(ki_target=ki_target, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5, abs=1e-6)
 
    # -- Mean over several competitors ----------------------------------
 
    def test_mean_over_mix_of_strong_and_weak(self):
        # C1: Ki_c=1 (10x stronger) -> penalty = 1
        # C2: Ki_c=100 (10x weaker) -> penalty = 0
        # mean = (1 + 0) / 2 = 0.5 → s5 = 0.5
        ki_competitors = pd.Series([1.0, 100.0])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5, abs=1e-6)
 
    def test_mean_of_three_competitors_concrete_values(self):
        # Ki_target = 10
        # C1: Ki_c=1   -> ratio=10  -> log=1   -> min(1,1) = 1
        # C2: Ki_c=10  -> ratio=1   -> log=0   -> no penalty
        # C3: Ki_c=100 -> ratio=0.1 -> log=-1  -> clipped to 0
        # mean = 1/3 -> s5 = 2/3
        ki_competitors = pd.Series([1.0, 10.0, 100.0])
        s5 = compute_s5_promiscuity(ki_target=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(2/3, abs=1e-6)
 
    def test_nan_does_not_count_in_the_mean(self):
        # [NaN, 1.0]: NaN is ignored -> the only valid penalty = 1
        # Mateix resultat que [1.0] sol
        s5_with_nan   = compute_s5_promiscuity(
            ki_target=10.0, ki_competitors=pd.Series([np.nan, 1.0])
        )
        s5_without_nan = compute_s5_promiscuity(
            ki_target=10.0, ki_competitors=pd.Series([1.0])
        )
        assert s5_with_nan == pytest.approx(s5_without_nan)
 
    # -- General properties ---------------------------------------------
 
    def test_result_always_between_0_and_1(self):
        """The score must always be in [0, 1] regardless of the input."""
        cases = [
            pd.Series([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
            pd.Series([10000.0]),
            pd.Series([0.0001]),
            pd.Series([5.0, 5.0, 5.0, 5.0]),
        ]
        for ki_competitors in cases:
            s5 = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors)
            assert 0.0 <= s5 <= 1.0, f"Case {ki_competitors.tolist()} gives {s5}"
 
    def test_independent_of_number_of_studies(self):
        """
        One OBP with 5 competitors and another with 50 competitors, with the
        SAME distribution (same ratios) -> the same s5.
        Checks independence from N (a mean, not a sum).
        """
        # 5 competidors, tots iguals
        ki_competitors_A = pd.Series([1.0] * 5)
        s5_A = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors_A)
 
        # 50 competidors, tots iguals
        ki_competitors_B = pd.Series([1.0] * 50)
        s5_B = compute_s5_promiscuity(ki_target=5.0, ki_competitors=ki_competitors_B)
 
        assert s5_A == pytest.approx(s5_B, abs=1e-6)
 
    def test_independent_of_absolute_scale(self):
        # Same relative proportions -> same s5
        s5_a = compute_s5_promiscuity(
            ki_target=10.0, ki_competitors=pd.Series([1.0, 100.0])
        )
        s5_b = compute_s5_promiscuity(
            ki_target=100.0, ki_competitors=pd.Series([10.0, 1000.0])
        )
        assert s5_a == pytest.approx(s5_b, abs=1e-6)
 
    def test_more_strong_competitors_lowers_s5(self):
        # Adding a strong competitor must lower (or keep) s5
        s5_one  = compute_s5_promiscuity(
            ki_target=10.0, ki_competitors=pd.Series([1.0])
        )
        s5_two = compute_s5_promiscuity(
            ki_target=10.0, ki_competitors=pd.Series([1.0, 0.5])
        )
        assert s5_two <= s5_one
 
 
# --------------------------------------------
# 6. compute_final_score
# --------------------------------------------
 
class TestComputeFinalScore:
 
    def test_score_with_default_weights(self):
        # All subscores at 1 -> score = sum of the weights = 1
        score = compute_final_score(
            s1=1.0, s2=1.0, s4=1.0, s5=1.0,
            weights=DEFAULT_WEIGHTS
        )
        assert score == pytest.approx(1.0, abs=1e-6)
 
    def test_score_zeros(self):
        score = compute_final_score(
            s1=0.0, s2=0.0, s4=0.0, s5=0.0,
            weights=DEFAULT_WEIGHTS
        )
        assert score == pytest.approx(0.0)
 
    def test_partial_score(self):
        weights = {'w_affinity': 0.5, 'w_selectivity': 0.3,
                   'w_stability': 0.1, 'w_promiscuity': 0.1}
        score = compute_final_score(s1=1.0, s2=0.0, s4=0.0, s5=0.0, weights=weights)
        assert score == pytest.approx(0.5)
 
    def test_partial_score_s4(self):
        # Only s4=1, the rest at 0 -> score = w_stability
        weights = {'w_affinity': 0.4, 'w_selectivity': 0.3,
                   'w_stability': 0.2, 'w_promiscuity': 0.1}
        score = compute_final_score(s1=0.0, s2=0.0, s4=1.0, s5=0.0, weights=weights)
        assert score == pytest.approx(0.2)
 
    def test_score_returns_float(self):
        score = compute_final_score(s1=0.8, s2=0.6, s4=0.7, s5=0.4,
                                     weights=DEFAULT_WEIGHTS)
        assert isinstance(score, float)
 
    def test_score_between_0_and_1_with_valid_subscores(self):
        score = compute_final_score(s1=0.7, s2=0.5, s4=0.8, s5=0.9,
                                     weights=DEFAULT_WEIGHTS)
        assert 0.0 <= score <= 1.0
 
 
# --------------------------------------------
# 7. build_obp_ranking  (integration test)
# --------------------------------------------
 
@pytest.fixture
def sample_data():
    """Minimal DataFrames for exercising build_obp_ranking."""
    binding = pd.DataFrame({
        'CAS':  ['64-17-5', '141-78-6', '100-52-7'],
        'Name': ['Ethanol', 'Ethyl acetate', 'Benzaldehyde'],
        'OBP_A': [5.0,  np.nan, 20.0],
        'OBP_B': [10.0, 8.0,   np.nan],
        'OBP_C': [2.0,  15.0,  3.0],
    })
 
    obp_info = pd.DataFrame({
        'Binding Protein Name': ['OBP_A', 'OBP_B', 'OBP_C'],
        'Binding Protein Type': ['Classic OBP', 'CSP', 'Classic OBP'],
        'Cystine count':        [6, 4, 6],
        'Species':              ['Apis mellifera', 'Bombyx mori', 'Apis mellifera'],
        'UniProtID':            ['P12345', '-', 'P67890'],
        'Alphafold':            ['-', '-', '-'],
    })
 
    ki_target = pd.Series({'OBP_A': 5.0, 'OBP_B': 10.0, 'OBP_C': 2.0})
    weights  = {'w_affinity': 0.45, 'w_selectivity': 0.25,
                'w_stability': 0.15, 'w_promiscuity': 0.15}
 
    return binding, obp_info, ki_target, weights
 
 
class TestBuildObpRanking:
 
    def test_returns_dataframe(self, sample_data):
        binding, obp_info, ki_target, weights = sample_data
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        assert isinstance(result, pd.DataFrame)
 
    def test_expected_columns(self, sample_data):
        binding, obp_info, ki_target, weights = sample_data
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        for col in ('OBP', 'Ki_target_uM', 's1_affinity', 's2_selectivity',
                    's4_stability', 's5_promiscuity', 'Score', 'Preferred'):
            assert col in result.columns, f"Falta la columna: {col}"
 
    def test_sorted_by_descending_score(self, sample_data):
        binding, obp_info, ki_target, weights = sample_data
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        scores = result['Score'].tolist()
        assert scores == sorted(scores, reverse=True)
 
    def test_preferred_classic_obp_6cys(self, sample_data):
        binding, obp_info, ki_target, weights = sample_data
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        # OBP_B has Cystine count=4 -> preferred=False
        obp_b_row = result[result['OBP'] == 'OBP_B']
        assert not obp_b_row.empty
        assert obp_b_row.iloc[0]['Preferred'] == False
 
        # OBP_A i OBP_C: Classic OBP + 6 Cys → preferred=True
        for obp in ('OBP_A', 'OBP_C'):
            row = result[result['OBP'] == obp]
            assert row.iloc[0]['Preferred'] == True
 
    def test_no_data_returns_empty(self, sample_data):
        binding, obp_info, _, weights = sample_data
        # Target Ki all NaN -> no OBP should come out
        ki_target_nan = pd.Series({'OBP_A': np.nan, 'OBP_B': np.nan, 'OBP_C': np.nan})
        result = build_obp_ranking(
            ki_values_target=ki_target_nan,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        assert result.empty
 
    def test_scores_between_0_and_1(self, sample_data):
        binding, obp_info, ki_target, weights = sample_data
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
            weights=weights,
            ki_min_matrix=2.0,
            ki_max_matrix=20.0,
        )
        assert (result['Score'] >= 0.0).all()
        assert (result['Score'] <= 1.0).all()
 
    def test_better_target_ki_gives_better_s1(self):
        """The OBP with the lowest target Ki must have the best s1."""
        binding = pd.DataFrame({
            'CAS':  ['64-17-5'],
            'Name': ['Ethanol'],
            'OBP_GOOD': [1.0],    # Ki molt baix → s1 alt
            'OBP_BAD':  [500.0],  # Ki molt alt → s1 baix
        })
        obp_info = pd.DataFrame({
            'Binding Protein Name': ['OBP_GOOD', 'OBP_BAD'],
            'Binding Protein Type': ['Classic OBP', 'Classic OBP'],
            'Cystine count':        [6, 6],
            'Species':              ['Apis mellifera', 'Apis mellifera'],
            'UniProtID':            ['-', '-'],
            'Alphafold':            ['-', '-'],
        })
        ki_target = pd.Series({'OBP_GOOD': 1.0, 'OBP_BAD': 500.0})
        weights  = {'w_affinity': 1.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 0.0}
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_GOOD', 'OBP_BAD'],
            weights=weights,
            ki_min_matrix=1.0,
            ki_max_matrix=500.0,
        )
        assert result.iloc[0]['OBP'] == 'OBP_GOOD'
 
    def test_weights_affect_the_ranking(self):
        """Changing the weights must change the ranking order."""
        binding = pd.DataFrame({
            'CAS':  ['64-17-5', '141-78-6'],
            'Name': ['Ethanol', 'Acetone'],
            'OBP_A': [1.0,  50.0],
            'OBP_B': [50.0, 1.0],
        })
        obp_info = pd.DataFrame({
            'Binding Protein Name': ['OBP_A', 'OBP_B'],
            'Binding Protein Type': ['Classic OBP', 'Classic OBP'],
            'Cystine count':        [6, 6],
            'Species':              ['Apis mellifera', 'Apis mellifera'],
            'UniProtID':            ['-', '-'],
            'Alphafold':            ['-', '-'],
        })
        ki_target = pd.Series({'OBP_A': 1.0, 'OBP_B': 50.0})
        weights  = {'w_affinity': 1.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 0.0}
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_A', 'OBP_B'],
            weights=weights,
            ki_min_matrix=1.0,
            ki_max_matrix=50.0,
        )
        assert result.iloc[0]['OBP'] == 'OBP_A'
 
    def test_n_vocs_bound_is_correct(self):
        """N_VOCs_bound must count the non-NaN values correctly."""
        binding = pd.DataFrame({
            'CAS':  ['A', 'B', 'C', 'D'],
            'Name': ['V1', 'V2', 'V3', 'V4'],
            'OBP_X': [1.0, np.nan, 3.0, np.nan],  # 2 valors no-NaN
        })
        obp_info = pd.DataFrame({
            'Binding Protein Name': ['OBP_X'],
            'Binding Protein Type': ['Classic OBP'],
            'Cystine count':        [6],
            'Species':              ['Apis mellifera'],
            'UniProtID':            ['-'],
            'Alphafold':            ['-'],
        })
        ki_target = pd.Series({'OBP_X': 1.0})
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_X'],
            weights=DEFAULT_WEIGHTS,
            ki_min_matrix=1.0,
            ki_max_matrix=3.0,
        )
        assert result.iloc[0]['N_VOCs_bound'] == 2
 
    def test_s5_drops_with_strong_competitors(self):
        """
        OBP_SPECIFIC: every competitor weaker than the target -> high s5.
        OBP_PROMISCUOUS: competitors stronger than the target -> low s5.
        """
        binding = pd.DataFrame({
            'CAS':  ['DIANA', 'B', 'C', 'D'],
            'Name': ['Diana', 'V2', 'V3', 'V4'],
            # OBP_SPECIFIC: Ki diana=5, competidors molt fluixos (Ki>>5) → s5 ≈ 1
            'OBP_SPECIFIC':    [5.0, 500.0, 1000.0, 800.0],
            # OBP_PROMISCUOUS: Ki diana=5, competidors molt forts (Ki<<5) → s5 ≈ 0
            'OBP_PROMISCUOUS': [5.0, 0.5,   0.1,    1.0],
        })
        obp_info = pd.DataFrame({
            'Binding Protein Name': ['OBP_SPECIFIC', 'OBP_PROMISCUOUS'],
            'Binding Protein Type': ['Classic OBP', 'Classic OBP'],
            'Cystine count':        [6, 6],
            'Species':              ['Apis mellifera', 'Apis mellifera'],
            'UniProtID':            ['-', '-'],
            'Alphafold':            ['-', '-'],
        })
        ki_target = pd.Series({'OBP_SPECIFIC': 5.0, 'OBP_PROMISCUOUS': 5.0})
        weights  = {'w_affinity': 0.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 1.0}
        result = build_obp_ranking(
            ki_values_target=ki_target,
            obp_info_table=obp_info,
            binding_table=binding,
            cas_col='CAS',
            name_col='Name',
            interferent_list=[],
            obp_name_list=['OBP_SPECIFIC', 'OBP_PROMISCUOUS'],
            weights=weights,
            ki_min_matrix=0.1,
            ki_max_matrix=1000.0,
        )
        specific_s5    = result[result['OBP'] == 'OBP_SPECIFIC']['s5_promiscuity'].values[0]
        promiscuous_s5 = result[result['OBP'] == 'OBP_PROMISCUOUS']['s5_promiscuity'].values[0]
        # The selective one must have a higher s5 than the promiscuous one
        assert specific_s5 > promiscuous_s5
        assert specific_s5 > 0.9       # All competitors weak -> s5 ~ 1.0
        assert promiscuous_s5 < 0.15   # All competitors strong -> s5 ~ 0.10


# --------------------------------------------
# 8. Interferent data scenarios for s2
# --------------------------------------------
#
# Three real scenarios that an interferent file can produce:
#   A) No interferents             -> no OBP can have a reliable s2
#   B) Mixed interferents          -> some OBPs have ALL the interferents
#                                     measured, others only some, others none
#   C) Complete interferents       -> ALL OBPs have ALL the interferents
#
# OBP_A: Interferent1 and Interferent2 measured   -> full coverage
# OBP_B: only Interferent1 measured               -> partial coverage
# OBP_C: no interferent measured                  -> no coverage

def _binding_table_with_interferents(all_complete=False):
    """Table with one target VOC and 2 interferents.
    all_complete=True  -> scenario C: ALL OBPs with both interferents measured.
    all_complete=False -> scenario B: OBP_A complete, OBP_B partial (missing
                           Interferent2), OBP_C with no interferent measured.
    """
    # OBP_A: Interferent1=20 gives s2=0.4 (neither 0 nor 1) - needed so we can
    # detect whether the selectivity weight affects the final Score or not.
    return pd.DataFrame({
        'CAS':  ['T-1', 'I-1', 'I-2'],
        'Name': ['TargetVOC', 'Interferent1', 'Interferent2'],
        'OBP_A': [5.0, 20.0, 80.0],
        'OBP_B': [5.0, 60.0, (70.0 if all_complete else np.nan)],
        'OBP_C': [5.0, (40.0 if all_complete else np.nan), (45.0 if all_complete else np.nan)],
    })


@pytest.fixture
def interferent_data():
    obp_info = pd.DataFrame({
        'Binding Protein Name': ['OBP_A', 'OBP_B', 'OBP_C'],
        'Binding Protein Type': ['Classic OBP', 'Classic OBP', 'Classic OBP'],
        'Cystine count':        [6, 6, 6],
        'Species':              ['Apis mellifera'] * 3,
        'UniProtID':            ['-', '-', '-'],
        'Alphafold':            ['-', '-', '-'],
    })
    ki_target = pd.Series({'OBP_A': 5.0, 'OBP_B': 5.0, 'OBP_C': 5.0})
    weights  = {'w_affinity': 0.45, 'w_selectivity': 0.25,
                'w_stability': 0.15, 'w_promiscuity': 0.15}
    return obp_info, ki_target, weights


def _build(binding, obp_info, ki_target, weights, interferent_list):
    return build_obp_ranking(
        ki_values_target=ki_target,
        obp_info_table=obp_info,
        binding_table=binding,
        cas_col='CAS',
        name_col='Name',
        interferent_list=interferent_list,
        obp_name_list=['OBP_A', 'OBP_B', 'OBP_C'],
        weights=weights,
        ki_min_matrix=5.0,
        ki_max_matrix=80.0,
    )


class TestScenarioNoInterferents:
    """A) No interferent defined (empty file / not provided)."""

    def test_no_obp_has_reliable_s2(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents()
        result = _build(binding, obp_info, ki_target, weights, interferent_list=[])

        assert not result['has_s2_data'].any()
        assert not result['has_all_interferents'].any()

    def test_s2_is_neutral_for_all(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents()
        result = _build(binding, obp_info, ki_target, weights, interferent_list=[])

        assert (result['s2_selectivity'] == 0.5).all()

    def test_score_does_not_depend_on_selectivity_weight(self, interferent_data):
        """Without s2 data the selectivity weight is discarded and renormalized
        for ALL OBPs - the result must be identical whether or not a high
        selectivity weight is included, because it is never used."""
        obp_info, ki_target, _ = interferent_data
        binding = _binding_table_with_interferents()

        weights_low = {'w_affinity': 0.45, 'w_selectivity': 0.0,
                          'w_stability': 0.15, 'w_promiscuity': 0.15}
        weights_high  = {'w_affinity': 0.45, 'w_selectivity': 0.90,
                          'w_stability': 0.15, 'w_promiscuity': 0.15}

        result_low = _build(binding, obp_info, ki_target, weights_low, interferent_list=[])
        result_high  = _build(binding, obp_info, ki_target, weights_high,  interferent_list=[])

        pd.testing.assert_series_equal(
            result_low.set_index('OBP')['Score'],
            result_high.set_index('OBP')['Score'],
        )


class TestScenarioMixedInterferents:
    """B) Some OBPs with all interferents measured, others with only some,
    others with none - the typical case for incomplete real data."""

    def test_has_all_interferents_only_for_the_complete_obp(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=False)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')

        assert result.loc['OBP_A', 'has_all_interferents'] == True
        assert result.loc['OBP_B', 'has_all_interferents'] == False
        assert result.loc['OBP_C', 'has_all_interferents'] == False

    def test_has_s2_data_with_partial_coverage(self, interferent_data):
        # OBP_B has only 1 of the 2 interferents measured -> has_s2_data=True
        # (it is reliable that no hidden interferent binds LESS, but not that
        # none hides a worse one - hence Table 1 requires has_all_interferents).
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=False)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')

        assert result.loc['OBP_A', 'has_s2_data'] == True
        assert result.loc['OBP_B', 'has_s2_data'] == True
        assert result.loc['OBP_C', 'has_s2_data'] == False

    def test_n_interferents_missing_is_correct(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=False)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')

        assert result.loc['OBP_A', 'N_interferents_missing'] == 0
        assert result.loc['OBP_B', 'N_interferents_missing'] == 1
        assert result.loc['OBP_C', 'N_interferents_missing'] == 2

    def test_obp_c_score_does_not_depend_on_selectivity(self, interferent_data):
        """OBP_C has no s2 data at all -> its Score cannot change if only the
        selectivity weight moves (it is discarded and renormalized)."""
        obp_info, ki_target, _ = interferent_data
        binding = _binding_table_with_interferents(all_complete=False)

        weights_low = {'w_affinity': 0.45, 'w_selectivity': 0.0,
                          'w_stability': 0.15, 'w_promiscuity': 0.15}
        weights_high  = {'w_affinity': 0.45, 'w_selectivity': 0.90,
                          'w_stability': 0.15, 'w_promiscuity': 0.15}

        result_low = _build(
            binding, obp_info, ki_target, weights_low,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')
        result_high = _build(
            binding, obp_info, ki_target, weights_high,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')

        assert result_low.loc['OBP_C', 'Score'] == pytest.approx(result_high.loc['OBP_C', 'Score'])
        # OBP_A, which does have a reliable s2, must change instead
        assert result_low.loc['OBP_A', 'Score'] != pytest.approx(result_high.loc['OBP_A', 'Score'])


class TestScenarioCompleteInterferents:
    """C) ALL OBPs have ALL interferents measured -> s2 reliable for
    everyone, Table 1 includes them all."""

    def test_has_all_interferents_for_all(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=True)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        )

        assert result['has_all_interferents'].all()

    def test_has_s2_data_for_all(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=True)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        )

        assert result['has_s2_data'].all()

    def test_no_interferent_is_missing(self, interferent_data):
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=True)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        )

        assert (result['N_interferents_missing'] == 0).all()

    def test_min_interferent_ki_is_correct(self, interferent_data):
        # OBP_A: min(20, 80) = 20 ; OBP_C: min(40, 45) = 40
        obp_info, ki_target, weights = interferent_data
        binding = _binding_table_with_interferents(all_complete=True)
        result = _build(
            binding, obp_info, ki_target, weights,
            interferent_list=['Interferent1', 'Interferent2'],
        ).set_index('OBP')

        assert result.loc['OBP_A', 'Min_Ki_interferent_uM'] == pytest.approx(20.0)
        assert result.loc['OBP_C', 'Min_Ki_interferent_uM'] == pytest.approx(40.0)
 