# Final imputation step: uses the best (t_min, k, p) parameters found by
# tanimoto_calibrar.py to fill in every missing Ki cell in the binding table
# via tanimoto_impute.fill_gaps(), then writes:
#   1. the "central prediction" CSV      (Ki_pred)
#   2. the "conservative lower bound" CSV (Ki_lower)
#   3. the diagnostic table              (T_max, sigma_log, decision, c1)
#
# (3) es imprescindible: sense ell tens els valors imputats pero cap rastre de
# com de fiable es cadascun. D'aqui surt la confianca de l'afinitat:
#       c1 = T_max * max(0, 1 - sigma_log / SIGMA_MAX)

import re
from pathlib import Path

import numpy as np
import pandas as pd

import tanimoto_impute as ti


BASE_DIR = Path(__file__).resolve().parent
BINDING_FILE = BASE_DIR / "Compound_OBP_binding.csv"
MATRIX_FILE = BASE_DIR / "tanimoto_data" / "tanimoto_matrix.npy"
INDEX_FILE = BASE_DIR / "tanimoto_data" / "voc_index.json"

OBP_SEARCHER_DIR = BASE_DIR.parent / "obp_searcher_aborrit"

# Both the central-prediction and conservative-lower-bound CSVs are written
# to both the tanimoto/ and obp_searcher_aborrit/ folders, so each folder
# always has a complete, matching pair.
OUTPUT_PRED_DIRS = [BASE_DIR, OBP_SEARCHER_DIR]
OUTPUT_LOWER_DIRS = [BASE_DIR, OBP_SEARCHER_DIR]
PRED_FILENAME = "Compound_OBP_binding_imputed_pred.csv"
LOWER_FILENAME = "Compound_OBP_binding_imputed_lower.csv"

CALIBRATION_SUMMARY = BASE_DIR / "results" / "loo_resum_combinacions.csv"
OUTPUT_DIAG_DIRS = [BASE_DIR / "results", OBP_SEARCHER_DIR / "results"]
DIAG_FILENAME = "tanimoto_diag_table.csv"


# ─────────────────────── CONSTANTS DE DECISIO ───────────────────────
# Palanca de conservadorisme del fitxer _lower: quantes sigma ens desplacem
# cap a la unio forta. z=0 -> Ki_lower == Ki_pred. z=1 -> un pas d'incertesa.
# NO es calibra mai (el LOO nomes valida Ki_pred): es una decisio teva.
Z = 1.0

# Normalitzacio de sigma per a la confianca c1. sigma_log=1.0 vol dir que els
# donants discrepen en un factor 10x en Ki -> confianca zero.
SIGMA_MAX = 1.0

# Fraccio minima de caselles BUIDES que volem poder omplir (terra dur).
# Nota: el fill rate depen NOMES de t_min (k i p no hi influeixen gens: k
# retalla la llista de donants despres del filtre, p nomes pondera).
MIN_FILL = 0.30

# Tolerancia de RMSE: dins d'aquest marge, la diferencia d'error es
# irrellevant i preferim la combinacio que omple MES matriu.
# 0.05 en log10 = un 12% de precisio relativa en Ki: imperceptible per
# decidir si un OBP es bon candidat. Posa 0.0 per tornar a "RMSE mana".
RMSE_TOL = 0.05

# Parametres de reserva si no hi ha calibratge disponible
FALLBACK = (0.35, 5, 2.0)

# Format de sortida. NO facis servir "%.2f": una Ki de 0.004 esdevindria 0.00,
# i convert_ki + impute_ki_tanimoto descarten els valors <= 0.
FLOAT_FMT = "%.4g"
# ────────────────────────────────────────────────────────────────────


def convert_ki(raw):
    # Parse a raw Ki cell into a float. Censored values like ">X" (Ki known
    # only to exceed X) are approximated as X*1.1.
    if pd.isna(raw):
        return np.nan
    text = str(raw).strip().replace("\xa0", "").replace(" ", "")
    if text.startswith(">"):
        try:
            return float(re.sub(r"[^\d.]", "", text)) * 1.1
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def fill_rate_for_t_min(binding_table, obp_name_list, T_matrix, idx_to_pos, t_min):
    """
    Quina fraccio de les caselles BUIDES es podria imputar amb aquest t_min?

    No cal predir res: nomes cal saber si existeix algun donant (un VOC amb Ki
    mesurada per aquell OBP) amb similitud >= t_min. Es una operacio de matriu,
    molt mes barata que executar fill_gaps().
    """
    rows_ok = np.array(sorted(idx_to_pos.keys()))
    pos_ok = np.array([idx_to_pos[r] for r in rows_ok])

    n_buides = n_imputables = 0
    for obp in obp_name_list:
        col = binding_table[obp]
        mesurats = [r for r in rows_ok if pd.notna(col[r]) and col[r] > 0]
        buides = binding_table.index[col.isna()]
        n_buides += len(buides)

        if not mesurats:
            continue  # cap donant possible en aquesta columna

        pos_donants = np.array([idx_to_pos[r] for r in mesurats])
        # files buides que a mes son a la matriu Tanimoto
        buides_ok = [r for r in buides if r in idx_to_pos]
        if not buides_ok:
            continue
        pos_buides = np.array([idx_to_pos[r] for r in buides_ok])

        sub = T_matrix[np.ix_(pos_buides, pos_donants)]
        with np.errstate(invalid="ignore"):
            millor = np.nanmax(np.where(np.isnan(sub), -np.inf, sub), axis=1)
        n_imputables += int((millor >= t_min).sum())

    return (n_imputables / n_buides) if n_buides else 0.0


def get_best_params(binding_table, obp_name_list, T_matrix, idx_to_pos):
    """
    Tria (t_min, k, p) del calibratge, pero amb dues restriccions separades:

      - t_min  -> controla QUANTES caselles s'omplen (fill rate)
      - k, p   -> controlen COM DE BE s'omplen (RMSE)

    El calibratge original agafava el RMSE minim sense mirar la cobertura, i
    aixo pot triar un t_min alt que deixa mitja matriu sense imputar a canvi
    d'una millora d'error imperceptible (0.30 -> 0.34 de RMSE = factor 2.00x
    -> 2.19x en Ki).
    """
    if not CALIBRATION_SUMMARY.is_file():
        print(f"  ! No trobo {CALIBRATION_SUMMARY.name} -> parametres de reserva {FALLBACK}")
        return FALLBACK

    try:
        df = pd.read_csv(CALIBRATION_SUMMARY)
    except Exception as e:
        print(f"  ! No puc llegir el calibratge ({e}) -> reserva {FALLBACK}")
        return FALLBACK

    if df.empty:
        print(f"  ! Calibratge buit -> reserva {FALLBACK}")
        return FALLBACK

    # Fill rate real per a cada t_min unic (nomes 3-4 valors: barat)
    print("\n  Fill rate real sobre les caselles BUIDES (depen nomes de t_min):")
    fill = {}
    for t in sorted(df["t_min"].unique()):
        fill[t] = fill_rate_for_t_min(binding_table, obp_name_list, T_matrix, idx_to_pos, float(t))
        marca = "OK " if fill[t] >= MIN_FILL else "descartat"
        print(f"     t_min = {t:<5} -> {fill[t]:6.1%}   {marca}")

    df = df.copy()
    df["fill_rate"] = df["t_min"].map(fill)

    valids = df[df["fill_rate"] >= MIN_FILL]
    if valids.empty:
        millor = df.sort_values("RMSE").iloc[0]
        print(f"\n  ! Cap t_min arriba a fill rate >= {MIN_FILL:.0%}. "
              f"Agafo el de RMSE minim (fill {millor['fill_rate']:.1%}).")
        rmse_ref = None
    else:
        # 1) el millor RMSE possible entre els que superen el terra de fill
        rmse_ref = valids["RMSE"].min()
        # 2) entre els que hi son a prop (dins de RMSE_TOL), agafa el que
        #    ompli MES matriu. Empat de fill -> el de menor RMSE.
        empatats = valids[valids["RMSE"] <= rmse_ref + RMSE_TOL]
        millor = empatats.sort_values(["fill_rate", "RMSE"],
                                      ascending=[False, True]).iloc[0]

    print(f"\n  Combinacio triada:  t_min={millor['t_min']}  "
          f"k={int(millor['k'])}  p={millor['p']}")
    print(f"     RMSE = {millor['RMSE']:.3f}  (factor d'error ~{10**millor['RMSE']:.2f}x en Ki)")
    print(f"     fill rate = {millor['fill_rate']:.1%} de les caselles buides")

    if rmse_ref is not None and millor["RMSE"] > rmse_ref + 1e-9:
        f_ref = 10 ** rmse_ref
        f_tri = 10 ** millor["RMSE"]
        alt = valids[valids["RMSE"] <= rmse_ref + 1e-9].iloc[0]
        print(f"     (renuncio a RMSE {rmse_ref:.3f} / fill {alt['fill_rate']:.1%}: "
              f"el factor d'error passa de {f_ref:.2f}x a {f_tri:.2f}x, "
              f"pero guanyo {millor['fill_rate'] - alt['fill_rate']:+.1%} de matriu)")

    return float(millor["t_min"]), int(millor["k"]), float(millor["p"])


def resum_diag(diag_table):
    """Afegeix la columna c1 (confianca de l'afinitat) i imprimeix un resum."""
    if diag_table.empty:
        print("  ! diag_table buit: cap cel.la imputada.")
        return diag_table

    d = diag_table.copy()
    d["c1"] = (d["T_max"] * (1.0 - d["sigma_log"] / SIGMA_MAX).clip(lower=0.0)).round(3)

    print("\n  Repartiment de decisions:")
    for dec, n in d["decision"].value_counts().items():
        print(f"     {dec:<14} {n:>8,}  ({n/len(d):.1%})")

    print(f"\n  T_max      -> mediana {d['T_max'].median():.3f}  ·  min {d['T_max'].min():.3f}")
    print(f"  sigma_log  -> mediana {d['sigma_log'].median():.3f}  ·  max {d['sigma_log'].max():.3f}")
    print(f"  c1         -> mediana {d['c1'].median():.3f}  ·  >0.7 en {(d['c1'] > 0.7).sum():,} cel.les")
    return d


def main():
    print("=" * 70)
    print("  GENERANT CSVS D'IMPUTACIO TANIMOTO")
    print("=" * 70)

    if not BINDING_FILE.is_file():
        raise FileNotFoundError(f"No s'ha trobat el fitxer de binding: {BINDING_FILE}")

    print(f"\n  Llegint {BINDING_FILE.name}...")
    df = pd.read_csv(BINDING_FILE)
    cas_col = df.columns[0]
    obp_name_list = list(df.columns[2:])

    binding_table = df.copy()
    for col in obp_name_list:
        binding_table[col] = binding_table[col].apply(convert_ki)

    n_cel = len(binding_table) * len(obp_name_list)
    n_known = int(binding_table[obp_name_list].notna().sum().sum())
    print(f"  -> {len(binding_table)} VOCs x {len(obp_name_list)} OBPs = {n_cel:,} caselles")
    print(f"  -> {n_known:,} mesurades ({n_known/n_cel:.1%})")

    print(f"\n  Carregant matriu Tanimoto...")
    T_matrix, cas_to_pos = ti.load_tanimoto_data(
        matrix_path=str(MATRIX_FILE),
        index_path=str(INDEX_FILE),
    )
    idx_to_pos = ti.build_binding_idx_to_matrix_pos(binding_table, cas_col, cas_to_pos)
    n_fora = len(binding_table) - len(idx_to_pos)
    print(f"  -> {len(idx_to_pos)}/{len(binding_table)} VOCs amb posicio valida a la matriu")
    if n_fora:
        print(f"     ({n_fora} VOCs sense SMILES resolt: no rebran cap imputacio "
              f"ni podran ser donants)")

    # Tria de parametres: t_min per fill rate, k/p per RMSE
    t_min, k, p = get_best_params(binding_table, obp_name_list, T_matrix, idx_to_pos)

    print(f"\n  Fent fill_gaps() amb: t_min={t_min}, k={k}, p={p}, z={Z}")
    binding_imputed, binding_pred, diag_table = ti.fill_gaps(
        binding_table, T_matrix, idx_to_pos, obp_name_list,
        t_min=t_min, k=k, p=p, z=Z,
    )

    # Confianca de l'afinitat per a cada cel.la imputada
    diag_table = resum_diag(diag_table)

    # Avis si l'arrodoniment pot matar valors petits
    ki_min = float(binding_pred[obp_name_list].min().min())
    if ki_min < 0.005:
        print(f"\n  ! Ki minima = {ki_min:.5f} uM. Amb '%.2f' esdevindria 0.00 i es "
              f"descartaria com a invalida. Usant {FLOAT_FMT}.")

    # binding_pred: central prediction values; binding_imputed: conservative
    # lower-bound values (safer for downstream selectivity decisions).
    print()
    for out_dir in OUTPUT_PRED_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / PRED_FILENAME
        binding_pred.to_csv(out_path, index=False, float_format=FLOAT_FMT)
        print(f"  CSV central guardat a: {out_path}")

    for out_dir in OUTPUT_LOWER_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / LOWER_FILENAME
        binding_imputed.to_csv(out_path, index=False, float_format=FLOAT_FMT)
        print(f"  CSV conservador guardat a: {out_path}")

    # LA LINIA QUE FALTAVA: sense aixo, perds T_max/sigma/c1 de cada cel.la
    # Escrit a tots dos directoris (com PRED/LOWER): obp_searcher_aborrit/main.py
    # el llegeix des del seu propi results/, no des del de tanimoto/.
    for out_dir in OUTPUT_DIAG_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / DIAG_FILENAME
        diag_table.to_csv(out_path, index=False)
        print(f"  Diagnostic guardat a:  {out_path}")

    n_plenes = int(binding_pred[obp_name_list].notna().sum().sum())
    print("\n" + "=" * 70)
    print("  RESUM")
    print("=" * 70)
    print(f"  Caselles plenes: {n_known:,} -> {n_plenes:,}  "
          f"({n_known/n_cel:.1%} -> {n_plenes/n_cel:.1%})")
    print(f"  Imputades: {len(diag_table):,}  ·  encara buides: {n_cel - n_plenes:,}")


if __name__ == "__main__":
    main()
