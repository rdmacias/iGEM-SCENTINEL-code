import os
import sys
import re
import pandas as pd
import numpy as np
from SMAAA import smaa_complet 
 
 
 
BINDING_FILE  = "Compound_OBP_binding.csv"
INFO_FILE     = "OBP_info_new.csv"
 
BEST_OBP_TYPE = "Classic OBP"
BEST_CYS_NUM  = 6
 
BIG_KI_VALUE  = 1000.0   # μM — valor quan el paper diu ">XX"
 
 
 
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
 
    # Ancoratges fixos de la literatura per normalitzar s1 (Cui et al.):
    # ceiling 1 μM (afinitat excel·lent) / floor 40 μM (afinitat feble).
    LITERATURE_MIN = 1.0
    LITERATURE_MAX = 40.0
    print(f"  → Ancoratges Ki (literatura): {LITERATURE_MIN:.1f} – {LITERATURE_MAX:.1f} μM")
 
    return binding_table, obp_info_table, cas_col, name_col, obp_name_list, LITERATURE_MIN, LITERATURE_MAX
 
 
# ── LLEGIR EL FITXER D'INTERFERENTS ──────────────────────────────────────────
 
def read_interferent_file(file_path): # ara el fitxer de interferents te la importancia associada 
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
 
def ask_user_for_tau():
    """Demana a l'usuari el valor de τ per al criteri de selectivitat."""
    print(f"\n  Tau (τ) per al criteri de selectivitat:")
    print(f"    s2 = min(1,  Ki_interferent / (Ki_diana × τ))")
    print(f"     1 = l'interferent simplement ha de lligar menys que el diana (lax)")
    print(f"    10 = l'interferent ha de lligar 10× menys (recomanat)")
    print(f"    50 = criteri exigent, per a dianes i interferents molt similars")
    print(f"  Rang vàlid: [1, 100]")
    while True:
        raw = input(f"  Valor de τ [per defecte 10]: ").strip()
        if raw == "":
            return 10.0
        try:
            tau = float(raw)
        except ValueError:
            print(f"  ✗ '{raw}' no és un número vàlid. Introdueix un valor entre 1 i 100.")
            continue
        if tau <= 0:
            print(f"  ✗ τ ≤ 0 causa divisió per zero o scores negatius. Ha de ser ≥ 1.")
        elif tau < 1:
            print(f"  ✗ τ < 1 no té sentit biològic: premiarà OBPs que uneixen l'interferent")
            print(f"    igual o més fort que el diana. Usa un valor entre 1 i 100.")
        elif tau > 100:
            print(f"  ✗ τ > 100 és massa restrictiu: quasi cap OBP passarà el criteri.")
            print(f"    Usa un valor entre 1 i 100.")
        else:
            return tau


def ask_user_for_weights(lock_selectivity_zero=False):
    """
    lock_selectivity_zero=True: la selectivitat (s2) no es pot avaluar per a
    cap OBP (cap té tots els interferents mesurats) → es bloqueja a 0 i el seu
    pes per defecte es redistribueix proporcionalment entre afinitat, estabilitat
    i promiscuïtat. No es demana el pes de selectivitat a l'usuari.
    """

    if lock_selectivity_zero:
        remaining = (DEFAULT_WEIGHTS['w_affinity'] +
                     DEFAULT_WEIGHTS['w_stability'] +
                     DEFAULT_WEIGHTS['w_promiscuity'])
        default_weights = {
            'w_affinity':    DEFAULT_WEIGHTS['w_affinity']    / remaining,
            'w_selectivity': 0.0,
            'w_stability':   DEFAULT_WEIGHTS['w_stability']   / remaining,
            'w_promiscuity': DEFAULT_WEIGHTS['w_promiscuity'] / remaining,
        }
        print(f"\n  Selectivitat (s2) bloquejada a 0: no es pot avaluar de manera fiable per a cap candidat.")
        print(f"  Nous pesos per defecte (pes de selectivitat redistribuït entre la resta):")
    else:
        default_weights = dict(DEFAULT_WEIGHTS)
        print(f"  Pesos per defecte:")

    print(f"    Afinitat (s1)     : {default_weights['w_affinity']:.2f}")
    print(f"    Selectivitat (s2) : {default_weights['w_selectivity']:.2f}"
          + ("  (bloquejat)" if lock_selectivity_zero else ""))
    print(f"    Estabilitat (s4)  : {default_weights['w_stability']:.2f}")
    print(f"    Promiscuïtat (s5) : {default_weights['w_promiscuity']:.2f}")
    print(f"  Suma: {sum(default_weights.values()):.2f}")

    use_default = input("Usar pesos per defecte? [S/n]: ").strip().lower()
    if use_default not in ('n', 'no'):
        return dict(default_weights)

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
        if lock_selectivity_zero and key == 'w_selectivity':
            custom_weights[key] = 0.0
            print(f"  Pes per a {label}: 0.00  (bloquejat — sense dades fiables de selectivitat)")
            continue
        while True:
            raw = input(f"  Pes per a {label} [per defecte {default_weights[key]:.2f}]: ").strip()
            if raw == "":
                custom_weights[key] = default_weights[key]
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
        return dict(default_weights)

    normalized = {k: v / total for k, v in custom_weights.items()}

    # Restricció: estabilitat = promiscuïtat (sempre iguals)
    mean_sp = (normalized['w_stability'] + normalized['w_promiscuity']) / 2.0
    normalized['w_stability']   = mean_sp
    normalized['w_promiscuity'] = mean_sp

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
 
 
def compute_s2_selectivity(ki_diana, ki_min_interferent, tau=10.0):
    if pd.isna(ki_min_interferent) or pd.isna(ki_diana) or ki_diana <= 0:
        return 0.5  # neutral: sense dades de selectivitat
 
    ratio = ki_min_interferent / (ki_diana * tau)
    return float(min(1.0, ratio))
 
 
def compute_s5_promiscuity(ki_diana, ki_competitors):
    if pd.isna(ki_diana) or ki_diana <= 0:
        return 0.0  # Valor 0 quan no tenim informació fiable de la Ki diana

    valid = pd.Series(ki_competitors).dropna()
    valid = valid[valid > 0]

    if valid.empty:
        return 0.0  # Valor 0 quan no hi ha competidors vàlids amb Ki positiva
    
    ratios = ki_diana / valid

    log_ratios = np.log10(ratios)

    penalties = np.clip(log_ratios, 0.0, 1.0)

    s5 = 1.0 - penalties.mean()

    return float(np.clip(s5, 0.0, 1.0))
 
 
def compute_s4_stability(obp_type):
    if pd.isna(obp_type) or str(obp_type).strip() in ("", "?", "nan"):
        return np.nan
    return TYPE_SCORES.get(str(obp_type).strip(), np.nan)
 
 
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
                      ki_min_matrix, ki_max_matrix, tau=10.0):
 
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
    # i quins interferents NO s'han pogut mesurar (falten) per a aquest OBP.
    min_interferent_ki_list  = []
    worst_interferent_list   = []
    missing_interferent_list = []

    n_interferents_total = len(ki_per_interferent)

    for _, obp_row in result_table.iterrows():
        current_obp  = obp_row['OBP']
        min_ki_found = np.nan
        worst_name   = "-"
        missing      = []

        for interf_name, ki_series in ki_per_interferent.items():
            ki_i = ki_series.get(current_obp, np.nan)
            if pd.isna(ki_i):
                missing.append(interf_name)
                continue
            if pd.isna(min_ki_found) or ki_i < min_ki_found:
                min_ki_found = ki_i
                worst_name   = interf_name

        min_interferent_ki_list.append(min_ki_found)
        worst_interferent_list.append(worst_name)
        missing_interferent_list.append(missing)

    result_table['Min_Ki_interferent_uM'] = min_interferent_ki_list
    result_table['Worst_interferent']     = worst_interferent_list
    result_table['Missing_interferents']  = missing_interferent_list
    result_table['N_interferents_total']   = n_interferents_total
    result_table['N_interferents_missing'] = result_table['Missing_interferents'].apply(len)
    # has_all_interferents: TOTS els interferents s'han mesurat per aquest OBP
    # → la s2 és fiable, perquè el que falta no pot amagar un mínim més baix.
    result_table['has_all_interferents'] = (
        (n_interferents_total > 0) & (result_table['N_interferents_missing'] == 0)
    )
 
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
            tau=tau
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

    # ── Flags de disponibilitat de dades ─────────────────────────────────────
    # has_s2_data: True si es va mesurar almenys un interferent per aquest OBP
    result_table['has_s2_data'] = ~result_table['Min_Ki_interferent_uM'].isna()
    # has_s4_data: True si el tipus d'OBP és reconegut al diccionari TYPE_SCORES
    result_table['has_s4_data'] = ~result_table['s4_stability'].isna()
    # has_s5_data: True si l'OBP té mesures de més d'un VOC (hi ha competidors)
    result_table['has_s5_data'] = result_table['N_VOCs_bound'] > 1

    # ── Score final: pesos renormalitzats per criteris disponibles ────────────
    # Si un criteri no té dades es descarta i els pesos restants es renormalitzen.
    final_scores = []
    for _, obp_row in result_table.iterrows():
        w = {'s1': weights['w_affinity']}
        if obp_row['has_s2_data']:
            w['s2'] = weights['w_selectivity']
        if obp_row['has_s4_data']:
            w['s4'] = weights['w_stability']
        if obp_row['has_s5_data']:
            w['s5'] = weights['w_promiscuity']

        total_w = sum(w.values())
        if total_w == 0:
            final_scores.append(0.0)
            continue

        score = (
            w.get('s1', 0.0) / total_w * obp_row['s1_affinity'] +
            w.get('s2', 0.0) / total_w * obp_row['s2_selectivity'] +
            w.get('s4', 0.0) / total_w * obp_row['s4_stability'] +
            w.get('s5', 0.0) / total_w * obp_row['s5_promiscuity']
        )
        final_scores.append(float(score))

    result_table['Score'] = final_scores

    # ── Score sense s2: descarta SEMPRE la selectivitat (s1+s4+s5 renormalitzats) ──
    # S'usa per a OBPs amb interferents incomplets, on s2 no és fiable
    # (el que falta podria amagar el mínim real).
    scores_no_s2 = []
    for _, obp_row in result_table.iterrows():
        w = {'s1': weights['w_affinity']}
        if obp_row['has_s4_data']:
            w['s4'] = weights['w_stability']
        if obp_row['has_s5_data']:
            w['s5'] = weights['w_promiscuity']

        total_w = sum(w.values())
        if total_w == 0:
            scores_no_s2.append(0.0)
            continue

        score = (
            w.get('s1', 0.0) / total_w * obp_row['s1_affinity'] +
            w.get('s4', 0.0) / total_w * obp_row['s4_stability'] +
            w.get('s5', 0.0) / total_w * obp_row['s5_promiscuity']
        )
        scores_no_s2.append(float(score))

    result_table['Score_sense_s2'] = scores_no_s2

    result_table = result_table.sort_values(
        by='Score', ascending=False
    ).reset_index(drop=True)

    return result_table
 
 
# ── TAULES 1/2/3 — selectivitat fiable / prometedors / llista de feina (Mode 1) ──

def show_table1_honest(result_table, how_many):
    """Taula 1 — OBPs amb TOTS els interferents mesurats: s2 fiable,
    comparació honesta amb els 4 criteris (s1+s2+s4+s5).

    Si cap OBP té totes les dades, NO es mostra una taula buida o amb
    valors inventats: s'avisa i es salta directament a la Taula 2.
    """
    df = result_table[result_table['has_all_interferents']].reset_index(drop=True)
    if df.empty:
        sep = "─" * 88
        print(f"\n{sep}")
        print(f"  TAULA 1 — Selectivitat NO FIABLE")
        n_interferents_total = (result_table['N_interferents_total'].iloc[0]
                                 if 'N_interferents_total' in result_table.columns
                                 and not result_table.empty else 0)
        if n_interferents_total == 0:
            print(f"  ✗ No hi ha dades d'interferents — la selectivitat no es pot avaluar.")
        else:
            print(f"  ✗ Cap OBP té TOTS els interferents mesurats — la selectivitat no es coneix")
            print(f"    de manera fiable per a cap candidat.")
        print(f"    Saltem directament a la Taula 2.")
        print(sep)
        return
    sep = "─" * 88
    print(f"\n{sep}")
    print(f"  TAULA 1 — Selectivitat FIABLE (tots els interferents mesurats) — top {how_many}")
    print(f"  Comparació honesta amb els 4 criteris (pesos renormalitzats)")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<22} {'Score':>7}  {'s1':>5}  {'s2':>5}  "
          f"{'s4':>5}  {'s5':>5}  {'Ki(μM)':>8}  {'Ki_interf':>9}  {'Tipus':<12}")
    print(sep)
    for pos, (_, row) in enumerate(df.head(how_many).iterrows()):
        ki_i = (f"{row['Min_Ki_interferent_uM']:.2f}"
                if not pd.isna(row['Min_Ki_interferent_uM']) else "   —  ")
        print(f"  {pos+1:>3}. {row['OBP']:<22} {row['Score']:>7.4f}  "
              f"{row['s1_affinity']:>5.3f}  {row['s2_selectivity']:>5.3f}  "
              f"{row['s4_stability']:>5.3f}  {row['s5_promiscuity']:>5.3f}  "
              f"{row['Ki_diana_uM']:>8.2f}  {ki_i:>9}  {str(row['Type']):<12}")
    print(sep)
    print(f"  Total amb tots els interferents mesurats: {len(df)}")


def show_table2_promising(result_table, how_many):
    """Taula 2 — OBPs als quals falta ≥1 interferent: s2 descartada
    (el que falta podria ser el mínim) → s1+s4+s5 renormalitzats.
    Retorna el subconjunt ordenat, per construir la Taula 3 amb la mateixa prioritat."""
    df = result_table[~result_table['has_all_interferents']].copy()
    if df.empty:
        return df

    df = df.sort_values('Score_sense_s2', ascending=False).reset_index(drop=True)

    sep = "─" * 88
    print(f"\n{sep}")
    print(f"  TAULA 2 — Prometedors IGNORANT la selectivitat (falta ≥1 interferent) — top {how_many}")
    print(f"  s2 descartada perquè és desconeguda → score amb s1+s4+s5 renormalitzats")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<22} {'Score':>7}  {'s1':>5}  {'s4':>5}  {'s5':>5}  "
          f"{'Ki(μM)':>8}  {'Falten':>6}  {'Tipus':<12}")
    print(sep)
    for pos, (_, row) in enumerate(df.head(how_many).iterrows()):
        s4_disp = f"{row['s4_stability']:.3f}" if not pd.isna(row['s4_stability']) else "  ? "
        print(f"  {pos+1:>3}. {row['OBP']:<22} {row['Score_sense_s2']:>7.4f}  "
              f"{row['s1_affinity']:>5.3f}  {s4_disp:>5}  {row['s5_promiscuity']:>5.3f}  "
              f"{row['Ki_diana_uM']:>8.2f}  {row['N_interferents_missing']:>6}  {str(row['Type']):<12}")
    print(sep)
    print(f"  Total amb interferents incomplets: {len(df)}")
    return df


def show_table3_boltz_worklist(df_table2_sorted):
    """Taula 3 — llista de feina Boltz: parelles OBP/interferent que falten i
    s'han de calcular. Ordenada segons la Taula 2: primer les dels OBPs
    més prometedors (val la pena calcular-les primer)."""
    if df_table2_sorted is None or df_table2_sorted.empty:
        return

    pairs = []
    for _, row in df_table2_sorted.iterrows():
        for interf in sorted(row['Missing_interferents']):
            pairs.append((row['OBP'], row['Score_sense_s2'], interf))

    if not pairs:
        sep = "─" * 70
        print(f"\n{sep}")
        print(f"  TAULA 3 — Llista de feina Boltz (parelles OBP/interferent que falten)")
        print(f"  ✗ No s'han definit interferents — no hi ha cap parella OBP/interferent a calcular.")
        print(sep)
        return

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  TAULA 3 — Llista de feina Boltz (parelles OBP/interferent que falten)")
    print(f"  Ordenada per prioritat de la Taula 2 (calcula primer les de dalt)")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<22} {'Score T2':>9}  {'Interferent que falta':<30}")
    print(sep)
    for i, (obp, score, interf) in enumerate(pairs, start=1):
        print(f"  {i:>3}. {obp:<22} {score:>9.4f}  {interf:<30}")
    print(sep)
    print(f"  Total de parelles OBP/interferent a calcular: {len(pairs)}")


# ── TAULES 1/2/3 — comú als 3 modes ──────────────────────────────────────────

def show_tables_1_2_3(result_table, how_many):
    """Mostra Taula 1/2/3 igual per als 3 modes: si cap OBP té TOTS els
    interferents mesurats, la Taula 1 ho indica i es salta a la 2 i la 3."""
    has_incomplete_data = (~result_table['has_all_interferents']).any()
    if has_incomplete_data:
        show_table1_honest(result_table, how_many)
        df_table2 = show_table2_promising(result_table, how_many)
        show_table3_boltz_worklist(df_table2)
    else:
        print("\n  (Tots els OBPs tenien TOTS els interferents mesurats — selectivitat fiable per a tots.)")


# ── RESULTATS PROBABILÍSTICS (Modes 2 i 3) ───────────────────────────────────

def show_smaa_results(result_table, rank_accept, how_many, central_w, criteri_noms=None):
    """Mostra el rànquing probabilístic SMAA.

    criteri_noms: etiquetes dels criteris realment explorats (en l'ordre dels
    pesos a central_w). Si la selectivitat està bloquejada a 0, no s'hi inclou
    — així el perfil de pesos no l'ensenya com si fos una dimensió explorada.
    """
    m = len(result_table)
    n_rank_cols = min(m, 5)
    sep = "═" * 95

    print(f"\n{sep}")
    print("  RÀNQUING PROBABILÍSTIC SMAA")
    print(sep)

    rank_header = "  ".join(f"{'p('+str(i+1)+'r)':>7}" for i in range(n_rank_cols))
    print(f"  {'OBP':<24} {'p(millor)':>10}  {rank_header}  barra")
    print("─" * 95)

    order = np.argsort(-rank_accept[:, 0])
    for obp_idx in order[:how_many]:
        row    = result_table.iloc[obp_idx]
        p_best = rank_accept[obp_idx, 0]
        probs  = "  ".join(f"{rank_accept[obp_idx, pos]:>7.3f}"
                           for pos in range(n_rank_cols))
        bar    = "█" * int(p_best * 40)
        print(f"  {row['OBP']:<24} {p_best:>10.3f}  {probs}  {bar}")

    print(sep)

    if central_w is not None:
        noms = criteri_noms if criteri_noms is not None else ['Afinitat', 'Selectiv.', 'Estabil.', 'Promiscu.']
        width = 9 * len(noms) + 2 * (len(noms) - 1) + 26
        print(f"\n  Perfils de pesos típics (en quin món de pesos guanya cada candidat):")
        header = "  ".join(f"{nom:>9}" for nom in noms)
        print(f"  {'OBP':<24}  {header}")
        print("─" * width)
        for obp_idx in order[:how_many]:
            cw = central_w.get(obp_idx)
            if cw is None:
                continue
            name = result_table.iloc[obp_idx]['OBP']
            valors = "  ".join(f"{v:>9.3f}" for v in cw)
            print(f"  {name:<24}  {valors}")
        print("─" * width)


# ── MAIN ──────────────────────────────────────────────────────────────────────
 
def main():
 
    print("OBP FINDER — iGEM URV 2025")
    print("Selecció del millor candidat OBP per a un VOC diana\n")
 
    for csv_path in (BINDING_FILE, INFO_FILE):
        if not os.path.isfile(csv_path):
            print(f"ERROR: No es troba el fitxer '{csv_path}'.")
            print("Posa els CSV a la mateixa carpeta que aquest script.")
            sys.exit(1)
 
    binding_table, obp_info_table, cas_col, name_col, obp_name_list, LITERATURE_MIN, LITERATURE_MAX = load_csv_files(
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
        
        if not interf_path:
            print("  → Continuem sense interferents.")
            break
        
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents carregats: {interferent_list}")
            print(f"\n  Validant interferents contra la base de dades...")
            interferent_list = validar_llista_interferents(
                interferent_list, binding_table, name_col, cas_col, interf_path
            )
            print(f"\n  ✓ {len(interferent_list)} interferents vàlids: {interferent_list}")
            break
        
        print(f"  ✗ Fitxer '{interf_path}' no trobat.")
        print(f"    Torna-ho a intentar o deixa buit per continuar sense interferents.")

    # Sense interferents, la selectivitat (s2) mai es podrà avaluar → bloquegem-la
    # ja en el primer cop que demanem els pesos (evita preguntar-los dues vegades).
    has_interferents = bool(interferent_list)
    weights = ask_user_for_weights(lock_selectivity_zero=not has_interferents)
    tau     = ask_user_for_tau()

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
        ki_min_matrix=LITERATURE_MIN,
        ki_max_matrix=LITERATURE_MAX,
        tau=tau,
    )

    if result_table.empty:
        print("  Cap OBP té dades per a aquest VOC.")
        sys.exit(0)

    # ── El cas "sense interferents" ja s'ha bloquejat en demanar els pesos.
    # Aquí només falta el cas en què SÍ hi ha interferents però cap OBP en té
    # TOTS mesurats: això no es pot saber fins que no es construeix el ranking.
    has_table1_data = result_table['has_all_interferents'].any() if has_interferents else False

    if has_interferents and not has_table1_data:
        print(f"\n  ✗ Cap OBP té TOTS els interferents mesurats: la selectivitat (s2)")
        print(f"    no es pot avaluar de manera fiable per a CAP candidat amb els pesos actuals.")
        print(f"    Això afecta els 3 mètodes (pesos deterministes, Dirichlet i SMAA).")
        print(f"    Torna a introduir els pesos sense selectivitat (es bloqueja a 0).")
        weights = ask_user_for_weights(lock_selectivity_zero=True)
        print("\nRecalculant ranking amb els nous pesos...")
        result_table = build_obp_ranking(
            ki_values_diana=ki_values_diana,
            obp_info_table=obp_info_table,
            binding_table=binding_table,
            cas_col=cas_col,
            name_col=name_col,
            interferent_list=interferent_list,
            obp_name_list=obp_name_list,
            weights=weights,
            ki_min_matrix=LITERATURE_MIN,
            ki_max_matrix=LITERATURE_MAX,
            tau=tau,
        )
        if result_table.empty:
            print("  Cap OBP té dades per a aquest VOC.")
            sys.exit(0)

    # ═══════════════════════════════════════════════════════════
    # SELECCIÓ DE MODE
    # ═══════════════════════════════════════════════════════════
    sep = "═" * 82
    print(f"\n{sep}")
    print("  MODE D'ANÀLISI")
    print(sep)
    print("  [1] Pesos deterministes    — resultat únic, ràpid, transparent")
    print("  [2] Dirichlet (c%)         — probabilitat robusta al voltant dels teus pesos")
    print("  [3] SMAA pur               — exploració total de l'espai de pesos")
    mode_raw = input("\n  Tria mode [1/2/3, per defecte 1]: ").strip()
    mode_num = mode_raw if mode_raw in ("1", "2", "3") else "1"

    # Si la selectivitat està bloquejada a 0 (cap candidat la pot avaluar de
    # manera fiable), s'exclou DEL TOT com a criteri — no només del càlcul
    # determinista. Si no, els Modes 2/3 exploren pesos aleatoris per a un
    # criteri que sabem que és soroll, i el perfil de pesos típics l'ensenya
    # com si fos una dimensió real de decisió.
    selectivitat_activa = weights['w_selectivity'] > 0.0

    criteri_noms = ['Afinitat', 'Estabil.', 'Promiscu.']
    pesos_base_list = [
        weights['w_affinity'], weights['w_stability'], weights['w_promiscuity'],
    ]
    cols_subscores = ['s1_affinity', 's4_stability', 's5_promiscuity']
    cols_te_dada = [
        np.ones(len(result_table), dtype=bool),
        np.ones(len(result_table), dtype=bool),
        result_table['has_s5_data'].values,
    ]
    if selectivitat_activa:
        criteri_noms.insert(1, 'Selectiv.')
        pesos_base_list.insert(1, weights['w_selectivity'])
        cols_subscores.insert(1, 's2_selectivity')
        cols_te_dada.insert(1, result_table['has_s2_data'].values)

    subscores = result_table[cols_subscores].values
    te_dada = np.column_stack(cols_te_dada)

    # ── Mode 1: Pesos deterministes ─────────────────────────────────────────
    if mode_num == "1":
        show_tables_1_2_3(result_table, how_many)

    # ── Mode 2: Dirichlet al voltant dels pesos ──────────────────────────────
    elif mode_num == "2":
        print(f"\n  Dispersió c% — fins on poden moure's els pesos (com a % del salt al criteri adjacent):")
        print(f"    10 = molt concentrat (pesos quasi fixos, ±10% del salt mínim entre criteris)")
        print(f"    50 = meitat del salt  (recomanat)")
        print(f"   100 = quasi empat entre els dos criteris més propers")
        c_raw = input("  Valor de c [per defecte 50]: ").strip()
        try:
            c_val = float(c_raw)
            if not (0.1 <= c_val <= 100):
                raise ValueError
        except ValueError:
            c_val = 50.0
            print(f"  → Usant c={c_val:.0f}%")

        n_raw = input("  Nombre d'iteracions [per defecte 200000]: ").strip()
        n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 200_000

        print(f"\nExecutant SMAA Dirichlet ({n_iter:,} iteracions, c={c_val:.0f}%)...")
        rank_accept, _, _ = smaa_complet(
            subscores=subscores, te_dada=te_dada,
            n_iter=n_iter, pesos_base=pesos_base_list, c=c_val,
            mode="dirichlet", seed=42,
        )
        show_smaa_results(result_table, rank_accept, how_many, central_w=None,
                           criteri_noms=criteri_noms)
        show_tables_1_2_3(result_table, how_many)

    # ── Mode 3: SMAA totalment aleatori ──────────────────────────────────────
    elif mode_num == "3":
        n_raw = input("  Nombre d'iteracions [per defecte 200000]: ").strip()
        n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 200_000

        print(f"\nExecutant SMAA pur ({n_iter:,} iteracions)...")
        rank_accept, _, central_w = smaa_complet(
            subscores=subscores, te_dada=te_dada,
            n_iter=n_iter, mode="smaa", seed=42,
        )
        show_smaa_results(result_table, rank_accept, how_many, central_w=central_w,
                           criteri_noms=criteri_noms)
        show_tables_1_2_3(result_table, how_many)


if __name__ == '__main__':
    main()