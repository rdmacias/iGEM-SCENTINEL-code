import os
import sys
import re
import pandas as pd
import numpy as np
 
 
 
 
BINDING_FILE  = "Compound_OBP_binding.csv"
INFO_FILE     = "OBP_info_new.csv"
 
BEST_OBP_TYPE = "Classic OBP"
BEST_CYS_NUM  = 6
 
BIG_KI_VALUE  = 1000.0   # μM — valor quan el paper diu ">XX"
 
# Tau (τ) per al criteri de selectivitat: s2 = min(1, Ki_interf / (Ki_diana × τ))
# Com més gran τ, més exigent és el criteri de selectivitat.
SELECTIVITY_TAU = 10.0
 
 
 
DEFAULT_WEIGHTS = {
    "w_affinity":     0.45,   # s1: Ki diana (afinitat pel VOC diana)
    "w_selectivity":  0.25,   # s2: selectivitat vs interferents
    "w_stability":   0.15,   # s4: estabilitat estructural 
    "w_promiscuity":  0.15,   # s5: promiscuïtat (penalització per unió a molts VOCs)
}
 
 
# Score d'estabilitat estructural per tipus d'OBP (s4)
# Basat en abundància de protocols d'expressió, estabilitat tèrmica
# (nombre de ponts disulfur) i similitud al fold Classic ben caracteritzat.
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
 
# Score per defecte quan el tipus no és reconegut o falta
TYPE_SCORE_UNKNOWN = 0.20
 
# ── LLEGIR I NETEJAR ELS CSV ──────────────────────────────────────────────────
 
def convert_ki_to_float(raw_value):
    """Converteix un valor de Ki (afinitat) a float en μM."""
    if pd.isna(raw_value):
        return np.nan
 
    text = str(raw_value).strip().replace('\xa0', '').replace(' ', '')
 
    if text.startswith('>'):
        try:
            number = float(re.sub(r'[^\d.]', '', text))
            return number * 1.1
        except ValueError:
            return BIG_KI_VALUE
 
    try:
        return float(text)
    except ValueError:
        return np.nan
 
 
def load_csv_files(binding_file_path, info_file_path):
    """
    Llegeix els dos CSV i retorna:
      - binding_table   : matriu VOCs × OBPs amb Kis en float
      - obp_info_table  : taula de metadades de cada OBP
      - cas_col         : nom de la columna CAS
      - name_col        : nom de la columna de noms de VOC
      - obp_name_list   : llista dels noms de les columnes d'OBP
    """
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
 
    # Rang real de Ki a tota la matriu — s'usa per normalitzar s1
    all_ki_values = binding_table[obp_name_list].values.flatten()
    all_ki_values = all_ki_values[~np.isnan(all_ki_values)]
    all_ki_values = all_ki_values[all_ki_values > 0]
    ki_min_matrix = float(all_ki_values.min())
    ki_max_matrix = float(all_ki_values.max())
    print(f"  → Rang Ki a la matriu: {ki_min_matrix:.3f} – {ki_max_matrix:.1f} μM")
 
    return binding_table, obp_info_table, cas_col, name_col, obp_name_list, ki_min_matrix, ki_max_matrix
 
 
# ── LLEGIR EL FITXER D'INTERFERENTS ──────────────────────────────────────────
 
def read_interferent_file(file_path):
    interferent_list = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'):
                continue
            interferent_list.append(clean_line)
    return interferent_list
 
# ── VALIDAR I SELECCIONAR UN VOC ──────────────────────────────────────────────
 
def validar_voc(binding_table, name_col, cas_col, search_text, file_path=None):
    """
    Cerca i valida un VOC per nom o CAS, amb tres nivells de prioritat:
      1) Coincidència exacta de CAS  (ex: "100-52-7")
      2) Coincidència exacta de nom  (case-insensitive)  → "1-hexanol" != "2-ethyl-1-hexanol"
      3) Coincidència parcial de nom (fallback interactiu)
 
    Modes:
      - file_path=None : mode interactiu → demana triar si hi ha múltiples parcials
      - file_path=ruta : mode fitxer    → mostra error i retorna None si no és exacte
    """
    search_text = search_text.strip()
    is_cas = bool(re.match(r'^\d+-\d+-\d+$', search_text))
 
    # ── 1) Coincidència exacta de CAS ─────────────────────────────────────────
    cas_exact = binding_table[cas_col].astype(str).str.strip() == search_text
    if cas_exact.any():
        return binding_table[cas_exact].iloc[0]
 
    # ── 2) Coincidència exacta de nom (case-insensitive) ──────────────────────
    if not is_cas:
        name_exact = binding_table[name_col].str.strip().str.lower() == search_text.lower()
        if name_exact.any():
            return binding_table[name_exact].iloc[0]
 
    # ── 3) Cerca parcial de nom (fallback) ────────────────────────────────────
    if not is_cas:
        name_partial = binding_table[name_col].str.contains(
            search_text, case=False, na=False, regex=False
        )
        found_partial = binding_table[name_partial]
    else:
        found_partial = pd.DataFrame()
 
    # ── No trobat en cap nivell ───────────────────────────────────────────────
    if found_partial.empty:
        if is_cas:
            print(f"\n  ✗ ERROR: El CAS '{search_text}' no existeix a la base de dades.")
            print(f"    Comprova que el número CAS sigui correcte.")
        else:
            # Suggeriments per paraula clau
            suggestions = set()
            for word in [w for w in search_text.split() if len(w) >= 3]:
                hits = binding_table[
                    binding_table[name_col].str.contains(word, case=False, na=False, regex=False)
                ][name_col].tolist()
                suggestions.update(hits)
 
            print(f"\n  ✗ ERROR: '{search_text}' no s'ha trobat a la base de dades.")
            if suggestions:
                print(f"    Potser et referies a algun d'aquests VOCs:")
                for s in sorted(suggestions)[:10]:
                    cas_val = binding_table.loc[binding_table[name_col] == s, cas_col].values
                    cas_str = cas_val[0] if len(cas_val) > 0 else "?"
                    print(f"      · {s}  (CAS: {cas_str})")
            else:
                print(f"    No s'han trobat VOCs similars a la base de dades.")
 
        if file_path:
            print(f"    Corregeix-ho al fitxer: {file_path}")
        return None
 
    # ── Una sola coincidència parcial → retornem directament ──────────────────
    if len(found_partial) == 1:
        return found_partial.iloc[0]
 
    # ── Múltiples coincidències parcials ──────────────────────────────────────
    print(f"\n  '{search_text}': {len(found_partial)} coincidències parcials trobades:")
    for i, (_, row) in enumerate(found_partial.iterrows()):
        print(f"    [{i+1}]  {row[name_col][:70]}  (CAS: {row[cas_col]})")
 
    if file_path:
        # Mode fitxer: no demanem input, cal que l'usuari especifiqui el nom exacte o CAS
        print(f"    Especifica el nom exacte o el CAS al fitxer: {file_path}")
        return None
 
    # Mode interactiu: l'usuari tria
    while True:
        choice = input(f"  Tria un número [1-{len(found_partial)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(found_partial):
            chosen = found_partial.iloc[int(choice) - 1]
            print(f"  → Seleccionat: {chosen[name_col]}  (CAS: {chosen[cas_col]})")
            return chosen
        print("  Número invàlid. Torna-ho a intentar.")
 
 
def validar_llista_interferents(interferent_list, binding_table, name_col, cas_col, file_path):
    """
    Valida tots els interferents del fitxer usant validar_voc.
    Mostra TOTS els errors i atura el programa si n'hi ha algun.
    L'usuari ha de corregir el fitxer .txt manualment i tornar a executar.
    """
    errors_found = False
    for entry in interferent_list:
        result = validar_voc(binding_table, name_col, cas_col, entry, file_path=file_path)
        if result is None:
            errors_found = True
 
    if errors_found:
        print(f"\n  ══ Corregeix els errors al fitxer d'interferents i torna a executar. ══")
        sys.exit(1)
 
    return interferent_list
 
 
# ── DEMANAR PESOS A L'USUARI ──────────────────────────────────────────────────
 
def ask_user_for_weights():
 
    print(f"  Pesos per defecte:")
    print(f"    Afinitat (s1)     : {DEFAULT_WEIGHTS['w_affinity']:.2f}")
    print(f"    Selectivitat (s2) : {DEFAULT_WEIGHTS['w_selectivity']:.2f}")
    print(f"    Estabilitat (s4)  : {DEFAULT_WEIGHTS['w_stability']:.2f}")
    print(f"    Promiscuïtat (s5) : {DEFAULT_WEIGHTS['w_promiscuity']:.2f}")
    print(f"  Suma: {sum(DEFAULT_WEIGHTS.values()):.2f}")
 
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
 
 
# ── CÀLCUL DELS SUB-SCORES ────────────────────────────────────────────────────
 
def compute_s1_affinity(ki_diana, ki_min_matrix, ki_max_matrix):
    if pd.isna(ki_diana) or ki_diana <= 0:
        return 0.0
    if ki_min_matrix <= 0 or pd.isna(ki_min_matrix):
        return 0.0
    if ki_max_matrix <= 0 or pd.isna(ki_max_matrix):
        return 0.0
 
    log_diana = -np.log10(ki_diana)
    log_best  = -np.log10(ki_min_matrix)
    log_worst = -np.log10(ki_max_matrix)
 
    denominator = log_best - log_worst
    if denominator == 0:
        return 0.0
 
    s1 = (log_diana - log_worst) / denominator
    return float(np.clip(s1, 0.0, 1.0))
 
 
def compute_s2_selectivity(ki_diana, ki_min_interferent, tau=SELECTIVITY_TAU):
    if pd.isna(ki_min_interferent) or pd.isna(ki_diana) or ki_diana <= 0:
        return 0.5  # Valor neutre quan no tenim informació fiable de les Kis
 
    ratio = ki_min_interferent / (ki_diana * tau)
    return float(min(1.0, ratio))
 
 
def compute_s5_promiscuity(ki_diana, ki_competitors):
    if pd.isna(ki_diana) or ki_diana <= 0:
        return 0.5  # Valor neutre quan no tenim informació fiable de la Ki diana

    valid = pd.Series(ki_competitors).dropna()
    valid = valid[valid > 0]

    if valid.empty:
        return 0.5  # Valor neutre quan no tenim informació fiable de les Ki dels competidors

    ratios = ki_diana / valid

    log_ratios = np.log10(ratios)

    penalties = np.clip(log_ratios, 0.0, 1.0)

    s5 = 1.0 - penalties.mean()

    return float(np.clip(s5, 0.0, 1.0))
 
 
def compute_s4_stability(obp_type):
   
 
    if pd.isna(obp_type):
        return TYPE_SCORE_UNKNOWN
   
    return TYPE_SCORES.get(str(obp_type).strip(), TYPE_SCORE_UNKNOWN)
 
 
def compute_final_score(s1, s2, s4, s5, weights):
    score = (
        weights['w_affinity']    * s1 +
        weights['w_selectivity'] * s2 +
        weights['w_promiscuity'] * s5 +
        weights['w_stability']   * s4
    )
    return float(score)
 
 
# ── CALCULAR EL RANKING ───────────────────────────────────────────────────────
 
def build_obp_ranking(ki_values_diana, obp_info_table, binding_table,
                      cas_col, name_col, interferent_list, obp_name_list, weights,
                      ki_min_matrix, ki_max_matrix):
 
    info_by_name  = obp_info_table.set_index('Binding Protein Name')
    n_vocs_total  = len(binding_table)
 
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
            obp_type   = "?"
            cys_count  = np.nan
            species    = "?"
            uniprot_id = "-"
            alphafold  = "-"
 
        is_preferred = (obp_type == BEST_OBP_TYPE and cys_count == BEST_CYS_NUM)
        n_vocs_bound = int(binding_table[obp_name].notna().sum())
 
        obp_rows.append({
            'OBP':          obp_name,
            'Ki_diana_uM':  ki_diana,
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
 
    # ── Selectivitat: Ki mínima dels interferents per cada OBP ────────────────
    ki_per_interferent = {}
 
    if interferent_list:
        print(f"\n[·] Aplicant filtre de selectivitat ({len(interferent_list)} interferents)...")
        for interferent_name in interferent_list:
            chosen_row = validar_voc(binding_table, name_col, cas_col, interferent_name)
 
            if chosen_row is None:
                print(f"  ✗ '{interferent_name}' no trobat — ignorat.")
                continue
 
            print(f"  ✓ '{interferent_name}': trobat → {chosen_row[name_col]}  (CAS: {chosen_row[cas_col]})")
 
            ki_series = pd.Series({col: chosen_row[col] for col in obp_name_list})
            ki_per_interferent[interferent_name] = ki_series
 
    # ── Per cada OBP calculem la Ki mínima dels interferents ─────────────────
    min_interferent_ki_list = []
    worst_interferent_list  = []
 
    for _, obp_row in result_table.iterrows():
        current_obp  = obp_row['OBP']
        min_ki_found = np.nan
        worst_name   = "-"
 
        for interf_name, ki_series in ki_per_interferent.items():
            ki_i = ki_series.get(current_obp, np.nan)
            if pd.isna(ki_i):
                continue
            if pd.isna(min_ki_found) or ki_i < min_ki_found:
                min_ki_found = ki_i
                worst_name   = interf_name
 
        min_interferent_ki_list.append(min_ki_found)
        worst_interferent_list.append(worst_name)
 
    result_table['Min_Ki_interferent_uM'] = min_interferent_ki_list
    result_table['Worst_interferent']     = worst_interferent_list
 
    # ── Calculem els sub-scores per a cada OBP ────────────────────────────────
    s1_list = []
    s2_list = []
    s4_list = []
    s5_list = []
 
    for _, obp_row in result_table.iterrows():
        s1 = compute_s1_affinity(
            ki_diana=obp_row['Ki_diana_uM'],
            ki_min_matrix=ki_min_matrix,
            ki_max_matrix=ki_max_matrix
        )
        s2 = compute_s2_selectivity(
            ki_diana=obp_row['Ki_diana_uM'],
            ki_min_interferent=obp_row['Min_Ki_interferent_uM'],
            tau=SELECTIVITY_TAU
        )
        s4 = compute_s4_stability(obp_type=obp_row['Type'])
 
        # ── Promiscuïtat (fórmula C, logarítmica) ──────────────────────────
        # Extraiem la columna sencera de Kis per aquesta OBP, excloent el diana
        ki_column      = binding_table[obp_row['OBP']]
        ki_competitors = ki_column[ki_column != obp_row['Ki_diana_uM']]
 
        s5 = compute_s5_promiscuity(
            ki_diana=obp_row['Ki_diana_uM'],
            ki_competitors=ki_competitors
        )
 
        s1_list.append(s1)
        s2_list.append(s2)
        s4_list.append(s4)
        s5_list.append(s5)
 
    result_table['s1_affinity']    = s1_list
    result_table['s2_selectivity'] = s2_list
    result_table['s4_stability']   = s4_list
    result_table['s5_promiscuity'] = s5_list
 
    # ── Score final ───────────────────────────────────────────────────────────
    final_scores = []
    for _, obp_row in result_table.iterrows():
        score = compute_final_score(
            s1=obp_row['s1_affinity'],
            s2=obp_row['s2_selectivity'],
            s4=obp_row['s4_stability'],
            s5=obp_row['s5_promiscuity'],
            weights=weights
        )
        final_scores.append(score)
 
    result_table['Score'] = final_scores
 
    result_table = result_table.sort_values(
        by='Score', ascending=False
    ).reset_index(drop=True)
 
    return result_table
 
 
# ── MOSTRAR ELS RESULTATS ─────────────────────────────────────────────────────
 
def show_results(result_table, voc_name, how_many, weights):
    """Imprimeix el ranking per pantalla."""
 
    separator = "═" * 82
    print(f"\n{separator}")
    print(f"  RANKING OBP per a: {voc_name}")
    print(f"  Pesos: af={weights['w_affinity']:.2f}  "
          f"sel={weights['w_selectivity']:.2f}  "
          f"est={weights['w_stability']:.2f}  "
          f"pro={weights['w_promiscuity']:.2f}")
    print(separator)
 
    best_obp = result_table.iloc[0]
    cys_display = int(best_obp['Cystines']) if not pd.isna(best_obp['Cystines']) else "?"
 
    print(f"\n  MILLOR CANDIDAT: {best_obp['OBP']}")
    print(f"    Score final  : {best_obp['Score']:.4f}  "
          f"(s1={best_obp['s1_affinity']:.3f}  "
          f"s2={best_obp['s2_selectivity']:.3f}  "
          f"s4={best_obp['s4_stability']:.3f}  "
          f"s5={best_obp['s5_promiscuity']:.3f})")
    print(f"    Ki diana     : {best_obp['Ki_diana_uM']:.2f} μM")
    print(f"    Tipus OBP    : {best_obp['Type']}")
    print(f"    Cisteïnes    : {cys_display}")
    print(f"    Espècie      : {best_obp['Species']}")
    print(f"    Preferit     : {'Sí (Classic OBP + 6 Cys)' if best_obp['Preferred'] else 'No'}")
    print(f"    VOCs units   : {best_obp['N_VOCs_bound']} de {len(result_table)} mesurats")
    if not pd.isna(best_obp['Min_Ki_interferent_uM']):
        print(f"    Pitjor interf: {best_obp['Worst_interferent']} "
              f"(Ki={best_obp['Min_Ki_interferent_uM']:.2f} μM)")
    if best_obp['UniProtID'] not in (None, '-', 'nan', ''):
        print(f"    UniProt ID   : {best_obp['UniProtID']}")
    if best_obp['Alphafold'] not in (None, '-', 'nan', ''):
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
              f"{row['s4_stability']:>5.3f}  {row['s5_promiscuity']:>5.3f}  {row['Ki_diana_uM']:>8.2f}  "
              f"{str(cys_val):>4}  {pref_mark:>4}  {ki_interf:>9}")
 
    print(f"{'─'*82}")
 
    total       = len(result_table)
    n_preferred = result_table['Preferred'].sum()
    print(f"\n  Total OBPs amb dades : {total}")
    print(f"  Classic OBP + 6 Cys  : {n_preferred}")
    print(f"\n  Columnes sub-score:")
    print(f"    s1 = afinitat [-log10(Ki) normalitzat a [0,1]]")
    print(f"    s2 = selectivitat [min(1, Ki_interf / (Ki_diana × τ={SELECTIVITY_TAU:.0f}))]")
    print(f"    s4 = estabilitat estructural (segons tipus d'OBP, basat en dades d'estabilitat i expressió)")
    print(f"    s5 = selectivitat global [1 − mitjana(min(1, log10(Ki_diana/Ki_VOC))+)]")
    print(f"    *  = Classic OBP amb 6 cisteïnes")
    print(separator)
 
 
# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
 
    print("OBP FINDER — iGEM URV 2025")
    print("Selecció del millor candidat OBP per a un VOC diana\n")
 
    for csv_path in (BINDING_FILE, INFO_FILE):
        if not os.path.isfile(csv_path):
            print(f"ERROR: No es troba el fitxer '{csv_path}'.")
            print("Posa els CSV a la mateixa carpeta que aquest script.")
            sys.exit(1)
 
    binding_table, obp_info_table, cas_col, name_col, obp_name_list, ki_min_matrix, ki_max_matrix = load_csv_files(
        BINDING_FILE, INFO_FILE
    )
 
    # ── Cercar el VOC diana (per nom o CAS) ───────────────────────────────────
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
 
    ki_values_diana = pd.Series({col: chosen_voc[col] for col in obp_name_list})
 
    # ── Fitxer d'interferents ─────────────────────────────────────────────────
    
    print("\nEl fitxer d'interferents pot contenir noms o números CAS, un per línia.")
    
    interferent_list = []
    while True:
        interf_path = input(
            "Fitxer d'interferents (deixa buit per saltar): "
        ).strip().strip('"').strip("'")
        
        # Cas 1: l'usuari ha deixat buit → saltar sense interferents
        if not interf_path:
            print("  → Continuem sense interferents.")
            break
        
        # Cas 2: el fitxer existeix → carregar i sortir del bucle
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents carregats: {interferent_list}")
            print(f"\n  Validant interferents contra la base de dades...")
            interferent_list = validar_llista_interferents(
                interferent_list, binding_table, name_col, cas_col, interf_path
            )
            print(f"\n  ✓ {len(interferent_list)} interferents vàlids: {interferent_list}")
            break
        
        # Cas 3: el fitxer NO existeix → demanar de nou
        print(f"  ✗ Fitxer '{interf_path}' no trobat.")
        print(f"    Torna-ho a intentar o deixa buit per continuar sense interferents.")
 
    weights = ask_user_for_weights()
 
    top_input = input("\nQuants candidats vols veure? [per defecte: 10]: ").strip()
    how_many  = int(top_input) if top_input.isdigit() and int(top_input) > 0 else 10
 
    print("\nCalculant ranking...")
    result_table = build_obp_ranking(
        ki_values_diana=ki_values_diana,
        obp_info_table=obp_info_table,
        binding_table=binding_table,
        cas_col=cas_col,
        name_col=name_col,
        interferent_list=interferent_list,
        obp_name_list=obp_name_list,
        weights=weights,
        ki_min_matrix=ki_min_matrix,
        ki_max_matrix=ki_max_matrix,
    )
 
    if result_table.empty:
        print("  Cap OBP té dades per a aquest VOC.")
        sys.exit(0)
 
    show_results(result_table, voc_display, how_many, weights)
 
 
if __name__ == "__main__":
    main()