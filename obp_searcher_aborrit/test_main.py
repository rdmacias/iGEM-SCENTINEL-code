"""
Tests unitaris per a main.py — OBP Finder (iGEM URV 2025)
Executa amb: python -m pytest test_main.py -v
"""
 
import math
import numpy as np
import pandas as pd
import pytest
 
# Importem les funcions que volem testar
from main import (
    BIG_KI_VALUE,
    DEFAULT_WEIGHTS,
    SELECTIVITY_TAU,
    TYPE_SCORES,
    TYPE_SCORE_UNKNOWN,
    convert_ki_to_float,
    # find_voc_rows — eliminada, substituïda per validar_voc
    read_interferent_file,
    compute_s1_affinity,
    compute_s2_selectivity,
    compute_s4_stability,
    compute_s5_promiscuity,
    compute_final_score,
    build_obp_ranking,
)
 
 
# ──────────────────────────────────────────────
# 1. convert_ki_to_float
# ──────────────────────────────────────────────
 
class TestConvertKiToFloat:
 
    def test_valor_numeric_simple(self):
        assert convert_ki_to_float("3.5") == pytest.approx(3.5)
 
    def test_valor_numeric_enter(self):
        assert convert_ki_to_float("100") == pytest.approx(100.0)
 
    def test_valor_amb_signe_major(self):
        # ">10" → 10 * 1.1 = 11.0
        result = convert_ki_to_float(">10")
        assert result == pytest.approx(11.0)
 
    def test_valor_amb_signe_major_gran(self):
        # ">500" → 550.0
        result = convert_ki_to_float(">500")
        assert result == pytest.approx(550.0)
 
    def test_nan_retorna_nan(self):
        assert math.isnan(convert_ki_to_float(np.nan))
 
    def test_text_invalid_retorna_nan(self):
        assert math.isnan(convert_ki_to_float("no_es_un_numero"))
 
    def test_signe_major_invalit_retorna_big_ki(self):
        # ">abc" → no es pot extreure nombre → BIG_KI_VALUE
        result = convert_ki_to_float(">abc")
        assert result == pytest.approx(BIG_KI_VALUE)
 
    def test_espais_blancs(self):
        assert convert_ki_to_float("  7.2  ") == pytest.approx(7.2)
 
    def test_espai_no_separable(self):
        # '\xa0' és un espai no separable
        assert convert_ki_to_float("\xa0 5.0 \xa0") == pytest.approx(5.0)
 
    def test_float_ja_es_float(self):
        assert convert_ki_to_float(42.0) == pytest.approx(42.0)
 
    def test_string_buit_retorna_nan(self):
        assert math.isnan(convert_ki_to_float(""))
 
    def test_valor_zero_retorna_zero(self):
        # 0.0 és un valor vàlid, no NaN
        assert convert_ki_to_float("0.0") == pytest.approx(0.0)
 
    def test_nan_tipus_none(self):
        # None és tractat com NaN per pd.isna
        assert math.isnan(convert_ki_to_float(None))
 
    @pytest.mark.parametrize("val,expected", [
        ("1.0",   1.0),
        ("50",    50.0),
        (">100",  110.0),
        (">1000", 1100.0),
        ("0.5",   0.5),
    ])
    def test_parametritzat_conversions(self, val, expected):
        assert convert_ki_to_float(val) == pytest.approx(expected)
 
 
# ──────────────────────────────────────────────
# 2. read_interferent_file
# ──────────────────────────────────────────────
 
class TestReadInterferentFile:
 
    def test_llegeix_linies_valides(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("Ethanol\nAcetone\nBenzaldehyde\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone', 'Benzaldehyde']
 
    def test_ignora_comentaris(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("# comentari\nEthanol\n# altre\nAcetone\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone']
 
    def test_ignora_linies_buides(self, tmp_path):
        f = tmp_path / "interf.txt"
        f.write_text("\nEthanol\n\nAcetone\n\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == ['Ethanol', 'Acetone']
 
    def test_fitxer_buit(self, tmp_path):
        f = tmp_path / "buit.txt"
        f.write_text("", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == []
 
    def test_nomes_comentaris(self, tmp_path):
        f = tmp_path / "c.txt"
        f.write_text("# línia 1\n# línia 2\n", encoding='utf-8')
        result = read_interferent_file(str(f))
        assert result == []
 
 
# ──────────────────────────────────────────────
# 3. compute_s1_affinity
# ──────────────────────────────────────────────
 
class TestComputeS1Affinity:
 
    def test_ki_igual_al_minim_retorna_1(self):
        # ki_diana == ki_min → log_diana == log_best → s1 = 1
        s1 = compute_s1_affinity(ki_diana=1.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(1.0)
 
    def test_ki_igual_al_maxim_retorna_0(self):
        s1 = compute_s1_affinity(ki_diana=100.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(0.0)
 
    def test_ki_nan_retorna_0(self):
        s1 = compute_s1_affinity(ki_diana=np.nan, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_zero_retorna_0(self):
        s1 = compute_s1_affinity(ki_diana=0.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_min_zero_retorna_0(self):
        s1 = compute_s1_affinity(ki_diana=5.0, ki_min_matrix=0.0, ki_max_matrix=100.0)
        assert s1 == 0.0
 
    def test_ki_min_igual_max_retorna_0(self):
        # denominator = 0 → retorna 0
        s1 = compute_s1_affinity(ki_diana=5.0, ki_min_matrix=10.0, ki_max_matrix=10.0)
        assert s1 == 0.0
 
    def test_valor_intermedi_entre_0_i_1(self):
        s1 = compute_s1_affinity(ki_diana=10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert 0.0 < s1 < 1.0
 
    def test_resultat_sempre_clipat_entre_0_i_1(self):
        # Ki molt gran (pitjor que el màxim de la matriu)
        s1 = compute_s1_affinity(ki_diana=999999.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert 0.0 <= s1 <= 1.0
 
    def test_monotonia_ki_decreixent_s1_creixent(self):
        # Com més petit el Ki, millor l'afinitat → s1 més gran
        ki_vals = [100.0, 50.0, 10.0, 1.0]
        s1_vals = [
            compute_s1_affinity(ki, ki_min_matrix=1.0, ki_max_matrix=100.0)
            for ki in ki_vals
        ]
        assert s1_vals == sorted(s1_vals)  # creixent quan Ki decreix
 
    def test_simetria_logaritmica(self):
        # ki_min=1, ki_max=100 → centre geomètric = sqrt(1*100) = 10 → s1 = 0.5
        s1 = compute_s1_affinity(ki_diana=10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        assert s1 == pytest.approx(0.5, abs=1e-6)
 
    def test_consistencia_amb_rang_diferent(self):
        # Mateix Ki relatiu → mateix s1 independentment del rang absolut
        s1_a = compute_s1_affinity(10.0, ki_min_matrix=1.0, ki_max_matrix=100.0)
        s1_b = compute_s1_affinity(100.0, ki_min_matrix=10.0, ki_max_matrix=1000.0)
        assert s1_a == pytest.approx(s1_b, abs=1e-6)
 
 
# ──────────────────────────────────────────────
# 4. compute_s2_selectivity
# ──────────────────────────────────────────────
 
class TestComputeS2Selectivity:
 
    def test_sense_interferents_retorna_neutral(self):
        # Sense interferent mesurat → score neutral (0.5), no optimista
        s2 = compute_s2_selectivity(ki_diana=5.0, ki_min_interferent=np.nan)
        assert s2 == 0.5
 
    def test_ki_diana_nan_retorna_neutral(self):
        # Ki diana invàlida → score neutral (0.5)
        s2 = compute_s2_selectivity(ki_diana=np.nan, ki_min_interferent=50.0)
        assert s2 == 0.5
 
    def test_interferent_molt_mes_gran_retorna_1(self):
        # ratio = 1000 / (1 * 10) = 100 → min(1, 100) = 1
        s2 = compute_s2_selectivity(ki_diana=1.0, ki_min_interferent=1000.0, tau=10.0)
        assert s2 == pytest.approx(1.0)
 
    def test_interferent_igual_diana_tau10_retorna_01(self):
        # ratio = 5 / (5 * 10) = 0.1
        s2 = compute_s2_selectivity(ki_diana=5.0, ki_min_interferent=5.0, tau=10.0)
        assert s2 == pytest.approx(0.1)
 
    def test_valor_clipat_a_1(self):
        s2 = compute_s2_selectivity(ki_diana=1.0, ki_min_interferent=999.0, tau=1.0)
        assert s2 <= 1.0
 
    def test_tau_personalitzada(self):
        # tau=1 → ratio = 10 / (5 * 1) = 2 → min(1, 2) = 1
        s2 = compute_s2_selectivity(ki_diana=5.0, ki_min_interferent=10.0, tau=1.0)
        assert s2 == pytest.approx(1.0)
 
    def test_interferent_millor_que_diana_penalitza(self):
        # Ki_interf < Ki_diana → interferent s'uneix més fort → penalitza
        # ratio = 1 / (10 * 10) = 0.01
        s2 = compute_s2_selectivity(ki_diana=10.0, ki_min_interferent=1.0, tau=10.0)
        assert s2 == pytest.approx(0.01)
 
    @pytest.mark.parametrize("ki_d,ki_i,tau,expected", [
        (5.0,  50.0, 10.0, 1.0),   # interferent 10x més fluix amb tau=10 → s2=1
        (5.0,   5.0, 10.0, 0.1),   # interferent igual amb tau=10 → s2=0.1
        (1.0, 100.0,  1.0, 1.0),   # interferent 100x més fluix → s2=1
        (10.0,  1.0,  1.0, 0.1),   # interferent 10x més fort → s2=0.1
    ])
    def test_parametritzat_selectivitat(self, ki_d, ki_i, tau, expected):
        s2 = compute_s2_selectivity(ki_diana=ki_d, ki_min_interferent=ki_i, tau=tau)
        assert s2 == pytest.approx(expected, abs=1e-6)
 
 
# ──────────────────────────────────────────────
# 4b. compute_s4_stability
# ──────────────────────────────────────────────
 
class TestComputeS4Stability:
 
    def test_classic_obp_retorna_score_maxim(self):
        s4 = compute_s4_stability("Classic OBP")
        assert s4 == pytest.approx(1.00)
 
    def test_pbp_retorna_085(self):
        s4 = compute_s4_stability("PBP")
        assert s4 == pytest.approx(0.85)
 
    def test_gobp1_retorna_085(self):
        s4 = compute_s4_stability("GOBP1")
        assert s4 == pytest.approx(0.85)
 
    def test_gobp2_retorna_085(self):
        s4 = compute_s4_stability("GOBP2")
        assert s4 == pytest.approx(0.85)
 
    def test_minus_c_retorna_065(self):
        s4 = compute_s4_stability("Minus-C OBP")
        assert s4 == pytest.approx(0.65)
 
    def test_plus_c_retorna_065(self):
        s4 = compute_s4_stability("Plus-C OBP")
        assert s4 == pytest.approx(0.65)
 
    def test_atypical_retorna_045(self):
        s4 = compute_s4_stability("Atypical OBP")
        assert s4 == pytest.approx(0.45)
 
    def test_csp_retorna_040(self):
        s4 = compute_s4_stability("CSP")
        assert s4 == pytest.approx(0.40)
 
    def test_tipus_desconegut_retorna_default(self):
        s4 = compute_s4_stability("Unknown OBP Type")
        assert s4 == pytest.approx(TYPE_SCORE_UNKNOWN)
 
    def test_nan_retorna_default(self):
        s4 = compute_s4_stability(np.nan)
        assert s4 == pytest.approx(TYPE_SCORE_UNKNOWN)
 
    def test_tipus_amb_espais_funciona(self):
        # El codi fa .strip() abans de buscar al diccionari
        s4 = compute_s4_stability("  Classic OBP  ")
        assert s4 == pytest.approx(1.00)
 
    def test_string_buit_retorna_default(self):
        s4 = compute_s4_stability("")
        assert s4 == pytest.approx(TYPE_SCORE_UNKNOWN)
 
    def test_resultat_sempre_entre_0_i_1(self):
        for obp_type in TYPE_SCORES.keys():
            s4 = compute_s4_stability(obp_type)
            assert 0.0 <= s4 <= 1.0, f"{obp_type} dóna {s4}, fora de [0,1]"
 
    def test_ordre_classic_millor_que_csp(self):
        """Una Classic OBP ha de tenir score més alt que una CSP."""
        s4_classic = compute_s4_stability("Classic OBP")
        s4_csp     = compute_s4_stability("CSP")
        assert s4_classic > s4_csp
 
    def test_ordre_classic_millor_que_atypical(self):
        """Una Classic OBP ha de tenir score més alt que una Atypical."""
        s4_classic  = compute_s4_stability("Classic OBP")
        s4_atypical = compute_s4_stability("Atypical OBP")
        assert s4_classic > s4_atypical
 
    def test_ordre_minus_c_igual_que_plus_c(self):
        """Minus-C i Plus-C tenen el mateix score."""
        s4_minus = compute_s4_stability("Minus-C OBP")
        s4_plus  = compute_s4_stability("Plus-C OBP")
        assert s4_minus == s4_plus
 
    def test_jerarquia_completa(self):
        """Verificar la jerarquia: Classic > PBP/GOBP > Minus/Plus-C > Atypical > CSP > Unknown."""
        scores = {
            "Classic":  compute_s4_stability("Classic OBP"),
            "PBP":      compute_s4_stability("PBP"),
            "Minus_C":  compute_s4_stability("Minus-C OBP"),
            "Atypical": compute_s4_stability("Atypical OBP"),
            "CSP":      compute_s4_stability("CSP"),
            "Unknown":  compute_s4_stability("Foo"),
        }
        assert scores["Classic"] > scores["PBP"]
        assert scores["PBP"]     > scores["Minus_C"]
        assert scores["Minus_C"] > scores["Atypical"]
        assert scores["Atypical"] > scores["CSP"]
        assert scores["CSP"]     > scores["Unknown"]
 
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
    def test_parametritzat_tots_tipus(self, obp_type, expected):
        assert compute_s4_stability(obp_type) == pytest.approx(expected)
 
 
# ──────────────────────────────────────────────
# 5. compute_s5_promiscuity (fórmula C, logarítmica)
# ──────────────────────────────────────────────
#
# Fórmula:
#   s5 = 1 - (1/N) * Σ min(1, [log10(Ki_diana / Ki_competidor)]+)
#
# Penalitza competidors que s'uneixen MÉS FORT que el diana (Ki_c < Ki_diana).
# Competidors febles o iguals (Ki_c >= Ki_diana) → penalització 0.
# Casos invàlids (sense competidors o diana invàlida) → retorna 0.5 (neutral).
 
class TestComputeS5Promiscuity:
 
    # ── Casos límit: retornen valor neutral (0.5) ──────────────────────
 
    def test_sense_competidors_retorna_neutral(self):
        # Series buida → score neutral (0.5)
        ki_competitors = pd.Series([], dtype=float)
        s5 = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5)
 
    def test_tots_competidors_nan_retorna_neutral(self):
        # Tots NaN → cap mesura vàlida → score neutral (0.5)
        ki_competitors = pd.Series([np.nan, np.nan, np.nan])
        s5 = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5)
 
    def test_ki_diana_nan_retorna_neutral(self):
        # Diana invàlida → score neutral (0.5)
        ki_competitors = pd.Series([1.0, 2.0, 3.0])
        s5 = compute_s5_promiscuity(ki_diana=np.nan, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5)
 
    def test_ki_diana_zero_retorna_neutral(self):
        # Diana = 0 → score neutral (0.5)
        ki_competitors = pd.Series([1.0, 2.0])
        s5 = compute_s5_promiscuity(ki_diana=0.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5)
 
    # ── Competidors més fluixos que el diana → s5 = 1 (cap penalització) ──
 
    def test_competidor_mes_fluix_no_penalitza(self):
        # Ki_c = 100 > Ki_diana = 10 → ratio = 10/100 = 0.1 → log < 0 → clip a 0
        # mean = 0 → s5 = 1
        ki_competitors = pd.Series([100.0])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    def test_competidor_igual_diana_no_penalitza(self):
        # Ki_c = Ki_diana → ratio = 1 → log10(1) = 0 → cap penalització
        ki_competitors = pd.Series([10.0])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    def test_tots_competidors_mes_fluixos_retorna_1(self):
        # Tots Ki_c > Ki_diana → cap penalització → s5 = 1
        ki_competitors = pd.Series([50.0, 100.0, 500.0])
        s5 = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(1.0)
 
    # ── Competidors més forts: penalització activa ─────────────────────
 
    def test_competidor_10x_mes_fort_dona_s5_zero(self):
        # Ki_c=1, Ki_diana=10 → ratio=10 → log10(10)=1 → min(1,1)=1
        # mean = 1 → s5 = 0
        ki_competitors = pd.Series([1.0])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)
 
    def test_competidor_100x_mes_fort_retallat_pel_sostre(self):
        # Ki_c=0.1, Ki_diana=10 → ratio=100 → log10(100)=2 → min(1,2)=1
        # El sostre talla a 1, igual que el cas 10x → s5 = 0
        ki_competitors = pd.Series([0.1])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.0)
 
    def test_sostre_fa_que_1000x_igual_que_10x(self):
        # min(1,...) garanteix que extrems no dominen el càlcul
        s5_10x   = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=pd.Series([1.0]))
        s5_1000x = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=pd.Series([0.01]))
        assert s5_10x == pytest.approx(s5_1000x)
 
    def test_competidor_sqrt10_dona_penalitzacio_05(self):
        # Ki_c = Ki_diana / sqrt(10) → ratio = sqrt(10) → log10 = 0.5
        # min(1, 0.5) = 0.5 → mean = 0.5 → s5 = 0.5
        ki_diana = 10.0
        ki_c = ki_diana / (10 ** 0.5)  # ≈ 3.162
        ki_competitors = pd.Series([ki_c])
        s5 = compute_s5_promiscuity(ki_diana=ki_diana, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5, abs=1e-6)
 
    # ── Mitjana sobre múltiples competidors ───────────────────────────
 
    def test_mitja_sobre_mix_fort_i_fluix(self):
        # C1: Ki_c=1 (10x fort) → penalització = 1
        # C2: Ki_c=100 (10x fluix) → penalització = 0
        # mean = (1 + 0) / 2 = 0.5 → s5 = 0.5
        ki_competitors = pd.Series([1.0, 100.0])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(0.5, abs=1e-6)
 
    def test_mitja_tres_competidors_valors_concrets(self):
        # Ki_diana = 10
        # C1: Ki_c=1   → ratio=10  → log=1   → min(1,1) = 1
        # C2: Ki_c=10  → ratio=1   → log=0   → cap penalització
        # C3: Ki_c=100 → ratio=0.1 → log=-1  → clip a 0
        # mean = 1/3 → s5 = 2/3
        ki_competitors = pd.Series([1.0, 10.0, 100.0])
        s5 = compute_s5_promiscuity(ki_diana=10.0, ki_competitors=ki_competitors)
        assert s5 == pytest.approx(2/3, abs=1e-6)
 
    def test_nan_no_compta_en_la_mitja(self):
        # [NaN, 1.0]: NaN s'ignora → única penalització vàlida = 1
        # Mateix resultat que [1.0] sol
        s5_amb_nan   = compute_s5_promiscuity(
            ki_diana=10.0, ki_competitors=pd.Series([np.nan, 1.0])
        )
        s5_sense_nan = compute_s5_promiscuity(
            ki_diana=10.0, ki_competitors=pd.Series([1.0])
        )
        assert s5_amb_nan == pytest.approx(s5_sense_nan)
 
    # ── Propietats generals ────────────────────────────────────────────
 
    def test_resultat_sempre_entre_0_i_1(self):
        """El score sempre ha d'estar a [0, 1] independentment de l'entrada."""
        casos = [
            pd.Series([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
            pd.Series([10000.0]),
            pd.Series([0.0001]),
            pd.Series([5.0, 5.0, 5.0, 5.0]),
        ]
        for ki_competitors in casos:
            s5 = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors)
            assert 0.0 <= s5 <= 1.0, f"Cas {ki_competitors.tolist()} dóna {s5}"
 
    def test_independent_nombre_estudis(self):
        """
        Una OBP amb 5 competidors i una altra amb 50 competidors,
        amb la MATEIXA distribució (mateixes ràtios) → mateix s5.
        Verifica la independència del N (mitjana, no suma).
        """
        # 5 competidors, tots iguals
        ki_competitors_A = pd.Series([1.0] * 5)
        s5_A = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors_A)
 
        # 50 competidors, tots iguals
        ki_competitors_B = pd.Series([1.0] * 50)
        s5_B = compute_s5_promiscuity(ki_diana=5.0, ki_competitors=ki_competitors_B)
 
        assert s5_A == pytest.approx(s5_B, abs=1e-6)
 
    def test_independent_de_lescala_absoluta(self):
        # Mateixes proporcions relatives → mateix s5
        s5_a = compute_s5_promiscuity(
            ki_diana=10.0, ki_competitors=pd.Series([1.0, 100.0])
        )
        s5_b = compute_s5_promiscuity(
            ki_diana=100.0, ki_competitors=pd.Series([10.0, 1000.0])
        )
        assert s5_a == pytest.approx(s5_b, abs=1e-6)
 
    def test_mes_competidors_forts_redueix_s5(self):
        # Afegir un competidor fort ha de reduir (o mantenir) s5
        s5_un  = compute_s5_promiscuity(
            ki_diana=10.0, ki_competitors=pd.Series([1.0])
        )
        s5_dos = compute_s5_promiscuity(
            ki_diana=10.0, ki_competitors=pd.Series([1.0, 0.5])
        )
        assert s5_dos <= s5_un
 
 
# ──────────────────────────────────────────────
# 6. compute_final_score
# ──────────────────────────────────────────────
 
class TestComputeFinalScore:
 
    def test_score_amb_pesos_defecte(self):
        # Tots els subscores a 1 → score = suma dels pesos = 1
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
 
    def test_score_parcial(self):
        weights = {'w_affinity': 0.5, 'w_selectivity': 0.3,
                   'w_stability': 0.1, 'w_promiscuity': 0.1}
        score = compute_final_score(s1=1.0, s2=0.0, s4=0.0, s5=0.0, weights=weights)
        assert score == pytest.approx(0.5)
 
    def test_score_parcial_s4(self):
        # Només s4=1, la resta a 0 → score = w_stability
        weights = {'w_affinity': 0.4, 'w_selectivity': 0.3,
                   'w_stability': 0.2, 'w_promiscuity': 0.1}
        score = compute_final_score(s1=0.0, s2=0.0, s4=1.0, s5=0.0, weights=weights)
        assert score == pytest.approx(0.2)
 
    def test_score_retorna_float(self):
        score = compute_final_score(s1=0.8, s2=0.6, s4=0.7, s5=0.4,
                                     weights=DEFAULT_WEIGHTS)
        assert isinstance(score, float)
 
    def test_score_entre_0_i_1_amb_subscores_valids(self):
        score = compute_final_score(s1=0.7, s2=0.5, s4=0.8, s5=0.9,
                                     weights=DEFAULT_WEIGHTS)
        assert 0.0 <= score <= 1.0
 
 
# ──────────────────────────────────────────────
# 7. build_obp_ranking  (test d'integració)
# ──────────────────────────────────────────────
 
@pytest.fixture
def dades_exemple():
    """DataFrames mínims per provar build_obp_ranking."""
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
 
    ki_diana = pd.Series({'OBP_A': 5.0, 'OBP_B': 10.0, 'OBP_C': 2.0})
    weights  = {'w_affinity': 0.45, 'w_selectivity': 0.25,
                'w_stability': 0.15, 'w_promiscuity': 0.15}
 
    return binding, obp_info, ki_diana, weights
 
 
class TestBuildObpRanking:
 
    def test_retorna_dataframe(self, dades_exemple):
        binding, obp_info, ki_diana, weights = dades_exemple
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_columnes_esperades(self, dades_exemple):
        binding, obp_info, ki_diana, weights = dades_exemple
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
        for col in ('OBP', 'Ki_diana_uM', 's1_affinity', 's2_selectivity',
                    's4_stability', 's5_promiscuity', 'Score', 'Preferred'):
            assert col in result.columns, f"Falta la columna: {col}"
 
    def test_ordenat_per_score_descendent(self, dades_exemple):
        binding, obp_info, ki_diana, weights = dades_exemple
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_preferred_classic_obp_6cys(self, dades_exemple):
        binding, obp_info, ki_diana, weights = dades_exemple
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
        # OBP_B té Cystine count=4 → preferred=False
        obp_b_row = result[result['OBP'] == 'OBP_B']
        assert not obp_b_row.empty
        assert obp_b_row.iloc[0]['Preferred'] == False
 
        # OBP_A i OBP_C: Classic OBP + 6 Cys → preferred=True
        for obp in ('OBP_A', 'OBP_C'):
            row = result[result['OBP'] == obp]
            assert row.iloc[0]['Preferred'] == True
 
    def test_sense_dades_retorna_buit(self, dades_exemple):
        binding, obp_info, _, weights = dades_exemple
        # Ki diana tot NaN → cap OBP ha de sortir
        ki_diana_nan = pd.Series({'OBP_A': np.nan, 'OBP_B': np.nan, 'OBP_C': np.nan})
        result = build_obp_ranking(
            ki_values_diana=ki_diana_nan,
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
 
    def test_scores_entre_0_i_1(self, dades_exemple):
        binding, obp_info, ki_diana, weights = dades_exemple
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_millor_ki_diana_dona_millor_s1(self):
        """L'OBP amb menor Ki diana ha de tenir millor s1."""
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
        ki_diana = pd.Series({'OBP_GOOD': 1.0, 'OBP_BAD': 500.0})
        weights  = {'w_affinity': 1.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 0.0}
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_pesos_afecten_el_ranking(self):
        """Canviar els pesos ha de canviar l'ordre del ranking."""
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
        ki_diana = pd.Series({'OBP_A': 1.0, 'OBP_B': 50.0})
        weights  = {'w_affinity': 1.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 0.0}
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_n_vocs_bound_correcte(self):
        """N_VOCs_bound ha de comptar correctament els valors no-NaN."""
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
        ki_diana = pd.Series({'OBP_X': 1.0})
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
 
    def test_s5_baixa_amb_competidors_forts(self):
        """
        OBP_SPECIFIC: tots els competidors més febles que el diana → s5 alt.
        OBP_PROMISCUOUS: competidors més forts que el diana → s5 baix.
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
        ki_diana = pd.Series({'OBP_SPECIFIC': 5.0, 'OBP_PROMISCUOUS': 5.0})
        weights  = {'w_affinity': 0.0, 'w_selectivity': 0.0,
                    'w_stability': 0.0, 'w_promiscuity': 1.0}
        result = build_obp_ranking(
            ki_values_diana=ki_diana,
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
        # La selectiva ha de tenir s5 més alt que la promíscua
        assert specific_s5 > promiscuous_s5
        assert specific_s5 > 0.9       # Tots els competidors fluixos → s5 ≈ 1.0
        assert promiscuous_s5 < 0.15   # Tots els competidors forts → s5 ≈ 0.10
 