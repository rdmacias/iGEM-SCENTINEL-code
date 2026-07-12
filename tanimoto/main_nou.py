import os
import sys
import re
import pandas as pd
import numpy as np

import common

_cfg = common.load_config()

#  CONFIGURACIÓ (llegida de config.yaml)

BINDING_FILE  = _cfg["fitxers"]["binding_csv"]
INFO_FILE     = _cfg["fitxers"]["info_csv"]

# Fitxers generats per tanimoto_omplir.py (si existeixen, s'usen per omplir buits)
BINDING_IMPUTED_PRED  = "Compound_OBP_binding_imputed_pred.csv"   # Ki central (per mostrar)
BINDING_IMPUTED_LOWER = "Compound_OBP_binding_imputed_lower.csv"  # Ki conservador (per s2)
IMPUTATION_DIAG       = "results/tanimoto_imputation_diagnostic.csv"
USE_IMPUTED           = True   # posa False per desactivar la imputació sense esborrar fitxers

BEST_OBP_TYPE = _cfg["ranking"]["best_obp_type"]
BEST_CYS_NUM  = _cfg["ranking"]["best_cys_num"]
BIG_KI_VALUE  = _cfg["ranking"]["big_ki_value_um"]   
SELECTIVITY_TAU = _cfg["ranking"]["selectivity_tau"]

DEFAULT_WEIGHTS = dict(_cfg["ranking"]["pesos_defecte"])

TYPE_SCORES = {
    "Classic OBP": 1.00,
    "PBP":         0.85,
    "GOBP1":       0.85,
    "GOBP2":       0.85,
    "Minus-C OBP": 0.65,
    "Plus-C OBP":  0.65,
    "Atypical OBP":0.45,
    "CSP":         0.40,
}
TYPE_SCORE_UNKNOWN = 0.20

common.ensure_dirs()



#  Funcions de ranking experimental (de l'original main.py)


def convert_ki_to_float(raw_value):
    if pd.isna(raw_value):
        return np.nan
    text = str(raw_value).strip().replace('\xa0', '').replace(' ', '')
    if text.startswith('>'):
        try:
            return float(re.sub(r'[^\d.]', '', text)) * 1.1
        except ValueError:
            return BIG_KI_VALUE
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_csv_files(binding_file_path, info_file_path):
    print(f"Llegint matriu de binding: {binding_file_path}")
    raw_binding = pd.read_csv(binding_file_path)
    print(f"  → {raw_binding.shape[0]} VOCs i {raw_binding.shape[1] - 2} OBPs")

    print(f"Llegint informació d'OBPs: {info_file_path}")
    obp_info_table = pd.read_csv(info_file_path)
    print(f"  → {len(obp_info_table)} OBPs amb metadades\n")

    cas_col       = raw_binding.columns[0]
    name_col      = raw_binding.columns[1]
    obp_name_list = list(raw_binding.columns[2:])

    binding_table = raw_binding.copy()
    for col_name in obp_name_list:
        binding_table[col_name] = binding_table[col_name].apply(convert_ki_to_float)

    # Carregar taules imputades (vNN) si existeixen i USE_IMPUTED és True
    imputed_pred  = None
    imputed_lower = None
    imputed_diag  = None

    if USE_IMPUTED:
        if all(os.path.isfile(f) for f in [BINDING_IMPUTED_PRED, BINDING_IMPUTED_LOWER]):
            print("  Carregant taules imputades (vNN Tanimoto)...")
            imputed_pred  = pd.read_csv(BINDING_IMPUTED_PRED)
            imputed_lower = pd.read_csv(BINDING_IMPUTED_LOWER)
            for col_name in obp_name_list:
                imputed_pred[col_name]  = pd.to_numeric(imputed_pred[col_name],  errors='coerce')
                imputed_lower[col_name] = pd.to_numeric(imputed_lower[col_name], errors='coerce')
            n_imp = int(imputed_pred[obp_name_list].notna().sum().sum()) - int(binding_table[obp_name_list].notna().sum().sum())
            print(f"  → {n_imp} cel·les addicionals cobertes per imputació vNN")
        else:
            print("  [Info] Fitxers imputats no trobats — executa tanimoto_omplir.py per generar-los.")

        if os.path.isfile(IMPUTATION_DIAG):
            imputed_diag = pd.read_csv(IMPUTATION_DIAG)

    # Rang Ki NOMÉS sobre dades experimentals (no distorsionar l'escala amb imputats)
    all_ki_values = binding_table[obp_name_list].values.flatten()
    all_ki_values = all_ki_values[~np.isnan(all_ki_values)]
    all_ki_values = all_ki_values[all_ki_values > 0]
    ki_min_matrix = float(all_ki_values.min())
    ki_max_matrix = float(all_ki_values.max())
    print(f"  → Rang Ki experimental: {ki_min_matrix:.3f} – {ki_max_matrix:.1f} μM")

    return (binding_table, obp_info_table, cas_col, name_col, obp_name_list,
            ki_min_matrix, ki_max_matrix, imputed_pred, imputed_lower, imputed_diag)


def read_interferent_file(file_path):
    interferent_list = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'):
                continue
            interferent_list.append(clean_line)
    return interferent_list


def validar_voc(binding_table, name_col, cas_col, search_text, file_path=None):
    search_text = search_text.strip()
    is_cas = bool(re.match(r'^\d+-\d+-\d+$', search_text))

    cas_exact = binding_table[cas_col].astype(str).str.strip() == search_text
    if cas_exact.any():
        return binding_table[cas_exact].iloc[0]

    if not is_cas:
        name_exact = binding_table[name_col].str.strip().str.lower() == search_text.lower()
        if name_exact.any():
            return binding_table[name_exact].iloc[0]

    if not is_cas:
        name_partial = binding_table[name_col].str.contains(
            search_text, case=False, na=False, regex=False)
        found_partial = binding_table[name_partial]
    else:
        found_partial = pd.DataFrame()

    if found_partial.empty:
        if is_cas:
            print(f"\n  ERROR: El CAS '{search_text}' no existeix a la base de dades.")
        else:
            suggestions = set()
            for word in [w for w in search_text.split() if len(w) >= 3]:
                hits = binding_table[
                    binding_table[name_col].str.contains(word, case=False, na=False, regex=False)
                ][name_col].tolist()
                suggestions.update(hits)
            print(f"\n  ERROR: '{search_text}' no s'ha trobat a la base de dades.")
            if suggestions:
                print(f"    Potser et referies a algun d'aquests VOCs:")
                for s in sorted(suggestions)[:10]:
                    cas_val = binding_table.loc[binding_table[name_col] == s, cas_col].values
                    cas_str = cas_val[0] if len(cas_val) > 0 else "?"
                    print(f"      · {s}  (CAS: {cas_str})")
        if file_path:
            print(f"    Corregeix-ho al fitxer: {file_path}")
        return None

    if len(found_partial) == 1:
        return found_partial.iloc[0]

    print(f"\n  '{search_text}': {len(found_partial)} coincidències parcials trobades:")
    for i, (_, row) in enumerate(found_partial.iterrows()):
        print(f"    [{i+1}]  {row[name_col][:70]}  (CAS: {row[cas_col]})")

    if file_path:
        print(f"    Especifica el nom exacte o el CAS al fitxer: {file_path}")
        return None

    while True:
        choice = input(f"  Tria un número [1-{len(found_partial)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(found_partial):
            chosen = found_partial.iloc[int(choice) - 1]
            print(f"  → Seleccionat: {chosen[name_col]}  (CAS: {chosen[cas_col]})")
            return chosen
        print("  Número invàlid. Torna-ho a intentar.")


def validar_llista_interferents(interferent_list, binding_table, name_col, cas_col, file_path):
    errors_found = False
    for entry in interferent_list:
        result = validar_voc(binding_table, name_col, cas_col, entry, file_path=file_path)
        if result is None:
            errors_found = True
    if errors_found:
        print(f"\n  ══ Corregeix els errors al fitxer d'interferents i torna a executar. ══")
        sys.exit(1)
    return interferent_list


def ask_user_for_weights():
    print(f"  Pesos per defecte:")
    print(f"    Afinitat (s1)     : {DEFAULT_WEIGHTS['w_affinity']:.2f}")
    print(f"    Selectivitat (s2) : {DEFAULT_WEIGHTS['w_selectivity']:.2f}")
    print(f"    Estabilitat (s4)  : {DEFAULT_WEIGHTS['w_stability']:.2f}")
    print(f"    Promiscuïtat (s5) : {DEFAULT_WEIGHTS['w_promiscuity']:.2f}")

    use_default = input("Usar pesos per defecte? [S/n]: ").strip().lower()
    if use_default not in ('n', 'no'):
        return dict(DEFAULT_WEIGHTS)

    custom_weights = {}
    weight_names = {
        "w_affinity":    "Afinitat (s1)",
        "w_selectivity": "Selectivitat (s2)",
        "w_stability":   "Estabilitat (s4)",
        "w_promiscuity": "Promiscuïtat (s5)",
    }
    print("\n  Introdueix els pesos (números entre 0 i 1).")
    print("  No cal que sumin 1: el programa els normalitzarà automàticament.\n")
    for key, label in weight_names.items():
        while True:
            raw = input(f"  Pes per a {label} [per defecte {DEFAULT_WEIGHTS[key]:.2f}]: ").strip()
            if raw == "":
                custom_weights[key] = DEFAULT_WEIGHTS[key]
                break
            try:
                value = float(raw)
                if value < 0:
                    print("    El pes no pot ser negatiu.")
                    continue
                custom_weights[key] = value
                break
            except ValueError:
                print("    Introdueix un número vàlid (ex: 0.4).")

    total = sum(custom_weights.values())
    if total == 0:
        print("  Tots els pesos són 0. S'usaran els pesos per defecte.")
        return dict(DEFAULT_WEIGHTS)

    normalized = {k: v / total for k, v in custom_weights.items()}
    print(f"\n  Pesos normalitzats (suma = 1.0):")
    for key, label in weight_names.items():
        print(f"    {label:<25}: {normalized[key]:.3f}")
    return normalized


def compute_s1_affinity(ki_diana, ki_min_matrix, ki_max_matrix):
    if pd.isna(ki_diana) or ki_diana <= 0:
        return 0.0
    log_diana = -np.log10(ki_diana)
    log_best  = -np.log10(ki_min_matrix)
    log_worst = -np.log10(ki_max_matrix)
    denominator = log_best - log_worst
    if denominator == 0:
        return 0.0
    return float(np.clip((log_diana - log_worst) / denominator, 0.0, 1.0))


def compute_s2_selectivity(ki_diana, ki_min_interferent, tau=SELECTIVITY_TAU):
    if pd.isna(ki_min_interferent) or pd.isna(ki_diana) or ki_diana <= 0:
        return 0.5
    return float(min(1.0, ki_min_interferent / (ki_diana * tau)))


def compute_s5_promiscuity(ki_diana, ki_competitors):
    if pd.isna(ki_diana) or ki_diana <= 0:
        return 0.5
    valid = pd.Series(ki_competitors).dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return 0.5
    penalties = np.clip(np.log10(ki_diana / valid), 0.0, 1.0)
    return float(np.clip(1.0 - penalties.mean(), 0.0, 1.0))


def compute_s4_stability(obp_type):
    if pd.isna(obp_type):
        return TYPE_SCORE_UNKNOWN
    return TYPE_SCORES.get(str(obp_type).strip(), TYPE_SCORE_UNKNOWN)


def compute_final_score(s1, s2, s4, s5, weights):
    return float(
        weights['w_affinity']    * s1 +
        weights['w_selectivity'] * s2 +
        weights['w_promiscuity'] * s5 +
        weights['w_stability']   * s4
    )


def build_obp_ranking(ki_values_diana, obp_info_table, binding_table,
                      cas_col, name_col, interferent_list, obp_name_list, weights,
                      ki_min_matrix, ki_max_matrix,
                      ki_values_diana_lower=None, ki_source=None,
                      imputed_lower=None, voc_row_idx=None):
    """
    ki_values_diana        → Ki per mostrar (experimental o Ki_pred imputat)
    ki_values_diana_lower  → Ki conservador (experimental o Ki_lower imputat),
                             usat ÚNICAMENT per calcular s2 i s5 dels interferents
    ki_source              → dict {obp_name: 'experimental'/'imputed_vNN'}
    imputed_lower          → taula sencera Ki_lower (per buscar interferents imputats)
    voc_row_idx            → índex de fila del VOC diana a binding_table/imputed
    """
    if ki_values_diana_lower is None:
        ki_values_diana_lower = ki_values_diana
    if ki_source is None:
        ki_source = {}

    info_by_name = obp_info_table.set_index('Binding Protein Name')
    n_vocs_total = len(binding_table)

    obp_rows = []
    for obp_name in obp_name_list:
        ki_diana = ki_values_diana.get(obp_name, np.nan)
        if pd.isna(ki_diana):
            continue

        if obp_name in info_by_name.index:
            obp_row    = info_by_name.loc[obp_name]
            obp_type   = obp_row['Binding Protein Type']
            raw_cys    = obp_row['Cystine count']
            cys_count  = int(raw_cys) if str(raw_cys).strip() not in ('-', '', 'nan') else np.nan
            species    = obp_row['Species']
            uniprot_id = obp_row['UniProtID']
            alphafold  = obp_row['Alphafold']
        else:
            obp_type, cys_count, species = "?", np.nan, "?"
            uniprot_id, alphafold = "-", "-"

        is_preferred = (obp_type == BEST_OBP_TYPE and cys_count == BEST_CYS_NUM)
        n_vocs_bound = int(binding_table[obp_name].notna().sum())
        source = ki_source.get(obp_name, 'experimental')

        obp_rows.append({
            'OBP':          obp_name,
            'Ki_diana_uM':  ki_diana,
            'Ki_source':    source,
            'Type':         obp_type,
            'Cystines':     cys_count,
            'Preferred':    is_preferred,
            'Species':      species,
            'UniProtID':    uniprot_id,
            'Alphafold':    alphafold,
            'N_VOCs_bound': n_vocs_bound,
        })

    if not obp_rows:
        return pd.DataFrame()

    result_table = pd.DataFrame(obp_rows)

    # Interferents: busca la Ki a binding_table (experimental) i, si no hi és,
    # a imputed_lower (conservador) — exactament com diu el protocol per a s2
    ki_per_interferent       = {}
    ki_per_interferent_lower = {}
    if interferent_list:
        print(f"\n[·] Aplicant filtre de selectivitat ({len(interferent_list)} interferents)...")
        for interferent_name in interferent_list:
            chosen_row = validar_voc(binding_table, name_col, cas_col, interferent_name)
            if chosen_row is None:
                print(f"  ✗ '{interferent_name}' no trobat — ignorat.")
                continue
            print(f"  ✓ '{interferent_name}' → {chosen_row[name_col]}")
            ki_series_exp = pd.Series({col: chosen_row[col] for col in obp_name_list})
            ki_per_interferent[interferent_name] = ki_series_exp

            # Ki conservador per a s2: usa Ki_lower imputat quan no hi ha experimental
            if imputed_lower is not None and voc_row_idx is not None:
                interf_idx = chosen_row.name
                ki_lower_row = imputed_lower.iloc[interf_idx]
                ki_series_lower = ki_series_exp.copy()
                for col in obp_name_list:
                    if pd.isna(ki_series_exp.get(col)):
                        ki_series_lower[col] = pd.to_numeric(ki_lower_row.get(col, np.nan), errors='coerce')
                ki_per_interferent_lower[interferent_name] = ki_series_lower
            else:
                ki_per_interferent_lower[interferent_name] = ki_series_exp

    min_interferent_ki_list, worst_interferent_list = [], []
    for _, obp_row in result_table.iterrows():
        min_ki_found, worst_name = np.nan, "-"
        # Per a s2 usem ki_lower dels interferents (conservador)
        for interf_name, ki_series in ki_per_interferent_lower.items():
            ki_i = ki_series.get(obp_row['OBP'], np.nan)
            if pd.isna(ki_i):
                continue
            if pd.isna(min_ki_found) or ki_i < min_ki_found:
                min_ki_found, worst_name = ki_i, interf_name
        min_interferent_ki_list.append(min_ki_found)
        worst_interferent_list.append(worst_name)

    result_table['Min_Ki_interferent_uM'] = min_interferent_ki_list
    result_table['Worst_interferent']     = worst_interferent_list

    s1_list, s2_list, s4_list, s5_list = [], [], [], []
    for _, obp_row in result_table.iterrows():
        # s1: usa Ki_diana (pred, central) per normalitzar
        s1 = compute_s1_affinity(obp_row['Ki_diana_uM'], ki_min_matrix, ki_max_matrix)
        # s2: usa Min_Ki_interferent ja calculat amb lower
        s2 = compute_s2_selectivity(obp_row['Ki_diana_uM'], obp_row['Min_Ki_interferent_uM'])
        s4 = compute_s4_stability(obp_row['Type'])
        # s5: usa la columna experimental (no la imputada) per no amplificar soroll
        ki_column = binding_table[obp_row['OBP']]
        ki_competitors = ki_column[ki_column != obp_row['Ki_diana_uM']]
        s5 = compute_s5_promiscuity(obp_row['Ki_diana_uM'], ki_competitors)
        s1_list.append(s1); s2_list.append(s2)
        s4_list.append(s4); s5_list.append(s5)

    result_table['s1_affinity']    = s1_list
    result_table['s2_selectivity'] = s2_list
    result_table['s4_stability']   = s4_list
    result_table['s5_promiscuity'] = s5_list

    final_scores = []
    for _, obp_row in result_table.iterrows():
        final_scores.append(compute_final_score(
            obp_row['s1_affinity'], obp_row['s2_selectivity'],
            obp_row['s4_stability'], obp_row['s5_promiscuity'], weights))
    result_table['Score'] = final_scores

    return result_table.sort_values('Score', ascending=False).reset_index(drop=True)


def show_results(result_table, voc_name, how_many, weights):
    separator = "═" * 82
    print(f"\n{separator}")
    print(f"  RANKING OBP per a: {voc_name}")
    print(f"  Pesos: af={weights['w_affinity']:.2f}  "
          f"sel={weights['w_selectivity']:.2f}  "
          f"est={weights['w_stability']:.2f}  "
          f"pro={weights['w_promiscuity']:.2f}")
    print(separator)

    best_obp  = result_table.iloc[0]
    cys_display = int(best_obp['Cystines']) if not pd.isna(best_obp['Cystines']) else "?"

    print(f"\n  MILLOR CANDIDAT: {best_obp['OBP']}")
    print(f"    Score final  : {best_obp['Score']:.4f}  "
          f"(s1={best_obp['s1_affinity']:.3f}  "
          f"s2={best_obp['s2_selectivity']:.3f}  "
          f"s4={best_obp['s4_stability']:.3f}  "
          f"s5={best_obp['s5_promiscuity']:.3f})")
    print(f"    Ki diana     : {best_obp['Ki_diana_uM']:.2f} μM")
    if best_obp.get('Ki_source') == 'imputed_vNN':
        print(f"    ⚠ FONT Ki    : ESTIMAT per similitud estructural (vNN Tanimoto)")
        print(f"                   No és un valor experimental — validar amb docking GNINA")
    print(f"    Tipus OBP    : {best_obp['Type']}")
    print(f"    Cisteïnes    : {cys_display}")
    print(f"    Espècie      : {best_obp['Species']}")
    print(f"    Preferit     : {'Sí (Classic OBP + 6 Cys)' if best_obp['Preferred'] else 'No'}")
    print(f"    VOCs units   : {best_obp['N_VOCs_bound']}")
    if not pd.isna(best_obp['Min_Ki_interferent_uM']):
        print(f"    Pitjor interf: {best_obp['Worst_interferent']} "
              f"(Ki={best_obp['Min_Ki_interferent_uM']:.2f} μM)")
    if str(best_obp['UniProtID']) not in (None, '-', 'nan', ''):
        print(f"    UniProt ID   : {best_obp['UniProtID']}")
    if str(best_obp['Alphafold']) not in (None, '-', 'nan', ''):
        print(f"    AlphaFold    : {best_obp['Alphafold']}")

    print(f"\n{'─'*82}")
    print(f"  TOP {how_many} OBPs — Score alt = millor candidat")
    print(f"{'─'*82}")
    print(f"  {'#':>3}  {'OBP':<20} {'Score':>7}  {'s1':>5}  {'s2':>5}  {'s4':>5}  {'s5':>5}  "
          f"{'Ki(μM)':>8}  {'Cys':>4}  {'Pref':>4}  {'Ki_interf':>9}")
    print(f"{'─'*82}")

    for position, (_, row) in enumerate(result_table.head(how_many).iterrows()):
        pref_mark = "*" if row['Preferred'] else " "
        cys_val   = int(row['Cystines']) if not pd.isna(row['Cystines']) else "?"
        ki_interf = (f"{row['Min_Ki_interferent_uM']:.2f}"
                     if not pd.isna(row['Min_Ki_interferent_uM']) else "   —  ")
        print(f"  {position+1:>3}. {row['OBP']:<20} {row['Score']:>7.4f}  "
              f"{row['s1_affinity']:>5.3f}  {row['s2_selectivity']:>5.3f}  "
              f"{row['s4_stability']:>5.3f}  {row['s5_promiscuity']:>5.3f}  "
              f"{row['Ki_diana_uM']:>8.2f}  {str(cys_val):>4}  {pref_mark:>4}  {ki_interf:>9}")

    print(f"{'─'*82}")
    print(f"\n  Total OBPs amb dades : {len(result_table)}")
    print(f"  Classic OBP + 6 Cys  : {result_table['Preferred'].sum()}")
    print(f"\n  Columnes sub-score:")
    print(f"    s1 = afinitat [-log10(Ki) normalitzat a [0,1]]")
    print(f"    s2 = selectivitat [min(1, Ki_interf / (Ki_diana × τ={SELECTIVITY_TAU:.0f}))]")
    print(f"    s4 = estabilitat estructural (tipus OBP)")
    print(f"    s5 = selectivitat global [1 − mitjana(min(1, log10(Ki_diana/Ki_VOC))+)]")
    print(f"    *  = Classic OBP amb 6 cisteïnes")
    print(separator)



# BLOC 2 — Docking GNINA per OBPs sense dades experimentals



def resolve_smiles(voc_name, cache=None):
    return common.resolve_smiles(voc_name, cache=cache)


def run_docking_for_obps(obp_list, voc_name, voc_smiles, smiles_cache=None):

    print("\n" + "=" * 55)
    print(f"  DOCKING GNINA — {len(obp_list)} OBPs sense dades experimentals")
    print("=" * 55)

    if not common.confirmar_docking_massiu(len(obp_list)):
        print("  Docking cancel·lat per l'usuari.")
        return {}

    scores = {}
    for i, obp_name in enumerate(obp_list, start=1):
        common.print_progress(i, len(obp_list), label=obp_name)
        ja_fet = common.docking_already_done(obp_name, voc_name, mode="vina")
        if ja_fet is not None:
            print(f"  ↺ Docking previ reutilitzat per {obp_name}")
            try:
                scores[obp_name] = float(ja_fet["vina_score"])
            except Exception:
                pass
            continue
        result = common.dock_pair(obp_name, voc_name, smiles_cache=smiles_cache, mode="vina")
        if result is not None:
            scores[obp_name] = result["vina_score"]
    return scores


def main():
    print("OBP FINDER — iGEM URV 2025")
    print("Selecció del millor candidat OBP per a un VOC diana\n")

    smiles_cache = common.load_smiles_cache()

    for csv_path in (BINDING_FILE, INFO_FILE):
        if not os.path.isfile(csv_path):
            print(f"ERROR: No es troba el fitxer '{csv_path}'.")
            print("Posa els CSV a la mateixa carpeta que aquest script.")
            sys.exit(1)

    (binding_table, obp_info_table, cas_col, name_col,
     obp_name_list, ki_min_matrix, ki_max_matrix,
     imputed_pred, imputed_lower, imputed_diag) = load_csv_files(BINDING_FILE, INFO_FILE)

    # ── 1) VOC diana ──────────────────────────────────────────────────────────
    print("Pots cercar el VOC diana pel seu nom o número CAS (ex: 3391-86-4)")
    while True:
        user_query = input("Nom o CAS del VOC diana: ").strip()
        if not user_query:
            print("  Cal escriure alguna cosa.")
            continue
        chosen_voc = validar_voc(binding_table, name_col, cas_col, user_query)
        if chosen_voc is None:
            print(f"  Torna-ho a intentar.")
            continue
        print(f"\n  VOC seleccionat: {chosen_voc[name_col]}  (CAS: {chosen_voc[cas_col]})")
        break

    voc_display   = chosen_voc[name_col]
    voc_name_safe = re.sub(r'[^\w]+', '_', voc_display)[:40]
    voc_row_idx   = chosen_voc.name   # índex de fila a binding_table

    # Ki experimentals del VOC diana
    ki_values_diana = pd.Series({col: chosen_voc[col] for col in obp_name_list})

    # Combinar amb Ki imputats (vNN): pred per mostrar, lower per calcular s2
    ki_values_diana_pred  = ki_values_diana.copy()
    ki_values_diana_lower = ki_values_diana.copy()
    ki_source = {}   # {obp_name: 'experimental' / 'imputed_vNN'}

    if imputed_pred is not None and imputed_lower is not None:
        voc_pred_row  = imputed_pred.iloc[voc_row_idx]
        voc_lower_row = imputed_lower.iloc[voc_row_idx]
        for col in obp_name_list:
            if pd.isna(ki_values_diana.get(col)):
                pred_val  = pd.to_numeric(voc_pred_row.get(col,  np.nan), errors='coerce')
                lower_val = pd.to_numeric(voc_lower_row.get(col, np.nan), errors='coerce')
                if not pd.isna(pred_val):
                    ki_values_diana_pred[col]  = pred_val
                    ki_values_diana_lower[col] = lower_val if not pd.isna(lower_val) else pred_val
                    ki_source[col] = 'imputed_vNN'
        n_imp_diana = sum(1 for v in ki_source.values() if v == 'imputed_vNN')
        print(f"  → {n_imp_diana} OBPs addicionals coberts per imputació vNN")
    else:
        ki_values_diana_pred  = ki_values_diana
        ki_values_diana_lower = ki_values_diana

    #  2) Identificar OBPs sense dades (ni experimentals ni imputades)
    obps_sense_dades = [
        name for name in obp_name_list
        if pd.isna(ki_values_diana_pred.get(name, np.nan))
    ]
    obps_amb_dades = [
        name for name in obp_name_list
        if not pd.isna(ki_values_diana_pred.get(name, np.nan))
    ]
    n_exp  = sum(1 for n in obps_amb_dades if ki_source.get(n, 'experimental') == 'experimental')
    n_vnn  = sum(1 for n in obps_amb_dades if ki_source.get(n) == 'imputed_vNN')
    print(f"\n  OBPs amb Ki experimental : {n_exp}")
    print(f"  OBPs amb Ki imputat (vNN): {n_vnn}")
    print(f"  OBPs sense dades (docking possible): {len(obps_sense_dades)}")

    # 3) Docking opcional per OBPs sense dades 
    docking_scores = {}
    if obps_sense_dades:
        fer_docking = input(
            f"\nVols fer docking GNINA per als {len(obps_sense_dades)} OBPs sense "
            f"dades experimentals? [s/N]: "
        ).strip().lower()

        if fer_docking in ('s', 'si', 'sí', 'y', 'yes'):
            top_n_str = input(
                f"  Quants (top per alfabètic, màx {len(obps_sense_dades)})? [10]: "
            ).strip()
            top_n = int(top_n_str) if top_n_str.isdigit() and int(top_n_str) > 0 else 10
            top_n = min(top_n, len(obps_sense_dades))

            print(f"\n  Obtenint SMILES per al VOC...")
            voc_smiles = resolve_smiles(voc_display, cache=smiles_cache)
            print(f"  SMILES: {voc_smiles}")

            obps_a_dockar = obps_sense_dades[:top_n]
            docking_scores = run_docking_for_obps(obps_a_dockar, voc_display, voc_smiles, smiles_cache=smiles_cache)

            # Guardem CSV de docking per a main4
            if docking_scores:
                slug = voc_display.replace(" ", "_").replace("-", "_")
                df_dock = pd.DataFrame([
                    {"obp_name": k, "vina_score": v, "kd_nm_exp": None}
                    for k, v in docking_scores.items()
                ])
                dock_csv = f"results/docking_complet_{slug}.csv"
                df_dock.to_csv(dock_csv, index=False)
                print(f"\n  ✓ CSV docking guardat a {dock_csv}")

    #  4) Interferents 
    print("\nEl fitxer d'interferents pot contenir noms o números CAS, un per línia.")
    interferent_list = []
    while True:
        interf_path = input(
            "Fitxer d'interferents (deixa buit per saltar): "
        ).strip().strip('"').strip("'")
        if not interf_path:
            print("  → Continuem sense interferents.")
            break
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents carregats.")
            print(f"\n  Validant interferents contra la base de dades...")
            interferent_list = validar_llista_interferents(
                interferent_list, binding_table, name_col, cas_col, interf_path)
            print(f"\n   {len(interferent_list)} interferents vàlids")
            break
        print(f"   Fitxer '{interf_path}' no trobat. Torna-ho a intentar o deixa buit.")


    weights = ask_user_for_weights()
    top_input = input("\nQuants candidats vols veure? [per defecte: 10]: ").strip()
    how_many  = int(top_input) if top_input.isdigit() and int(top_input) > 0 else 10

    #  6) Ranking experimental 
    print("\nCalculant ranking experimental...")
    result_table = build_obp_ranking(
        ki_values_diana=ki_values_diana_pred,
        obp_info_table=obp_info_table,
        binding_table=binding_table,
        cas_col=cas_col,
        name_col=name_col,
        interferent_list=interferent_list,
        obp_name_list=obp_name_list,
        weights=weights,
        ki_min_matrix=ki_min_matrix,
        ki_max_matrix=ki_max_matrix,
        ki_values_diana_lower=ki_values_diana_lower,
        ki_source=ki_source,
        imputed_lower=imputed_lower,
        voc_row_idx=voc_row_idx,
    )

    if result_table.empty:
        print("  Cap OBP té dades experimentals per a aquest VOC.")
    else:
        show_results(result_table, voc_display, how_many, weights)
        out_csv = f"results/ranking_final_{voc_name_safe}.csv"
        result_table.to_csv(out_csv, index=False)
        print(f"\n  ✓ Ranking guardat a {out_csv}")


    if docking_scores:
        print("\n" + "═"*55)
        print("  RESUM DOCKING GNINA (OBPs sense dades experimentals)")
        print("═"*55)
        df_d = pd.DataFrame(
            sorted(docking_scores.items(), key=lambda x: x[1]),
            columns=["OBP", "Vina (kcal/mol)"]
        )
        for i, (_, row) in enumerate(df_d.iterrows(), 1):
            print(f"  {i:>3}. {row['OBP']:<25}  Vina={row['Vina (kcal/mol)']:.3f}")
        print(f"\n  → Millor docking: {df_d.iloc[0]['OBP']}  "
              f"({df_d.iloc[0]['Vina (kcal/mol)']:.3f} kcal/mol)")


if __name__ == "__main__":
    main()