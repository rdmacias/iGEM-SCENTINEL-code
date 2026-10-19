import os
import sys
import re
import pandas as pd
import numpy as np
import confidence as cf
from SMAAA import smaa_full
 
 
DIAG_FILE = os.path.join("results", "tanimoto_diag_table.csv")
MIN_CONF_FILTER = None   # set 0.8 to see only the reliable candidates
 
 
 
BINDING_FILE_EXPERIMENTAL      = "Compound_OBP_binding.csv"
# With predicted information (option [2]) we use TWO different files:
#   - BINDING_FILE_PREDICTED            -> selectivity (s2): interferent Ki values
#   - BINDING_FILE_PREDICTED_AFFINITY   -> affinity (s1): target VOC Ki
BINDING_FILE_PREDICTED          = "Compound_OBP_binding_imputed_lower.csv"
BINDING_FILE_PREDICTED_AFFINITY = "Compound_OBP_binding_imputed_pred.csv"
INFO_FILE                       = "OBP_info_new.csv"
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
 
BEST_OBP_TYPE = "Classic OBP"
BEST_CYS_NUM  = 6
 
DANGER_INTERFERENT_MARGIN = 1.33   # flag an interferent if it binds equal, stronger, or up to 33% weaker than the target
MAX_DANGER_NAMES     = 10      # how many names to show per candidate (None = all)
SUPERBINDER_KI_MAX   = 2.0     # Ki (uM) below which a VOC is a strong "superbinder" (marked *)
MAX_SUPERBINDER_NAMES = 10     # max superbinder names to show per candidate
S2_NEUTRAL           = 0.5     # s2 value when selectivity is not evaluable (neither good nor bad)
 
DEFAULT_WEIGHTS = {
    "w_affinity":     0.45,   # s1: target Ki (affinity for the target VOC)
    "w_selectivity":  0.25,   # s2: selectivity vs interferents
    "w_stability":   0.15,   # s4: structural stability
    "w_promiscuity":  0.15,   # s5: promiscuity (penalty for binding many VOCs)
}
 
 
# Structural stability score per OBP type (s4).
# Based on abundance of expression protocols, thermal stability
# (number of disulfide bridges) and similarity to the well-characterized Classic fold.
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
 
 
# -- BINDING DATA SOURCE SELECTION -------------------------------------------
 
def infer_s1_source_label(selected_value, experimental_value, binding_source_label):
    """Label whether s1 comes from experimental data or from the imputed/predicted matrix."""
    source = str(binding_source_label or "").strip().lower()
    if source == "experimental":
        return "experimental"
 
    if pd.isna(selected_value) or pd.isna(experimental_value):
        return "new (tanimoto)"
 
    if np.isclose(selected_value, experimental_value):
        return "experimental"
 
    return "new (tanimoto)"
 
 
def resolve_data_file(file_name):
    """Resolve a file path relative to the script folder."""
    if not file_name:
        return file_name
 
    if os.path.isabs(file_name):
        return file_name
 
    for candidate in (
        file_name,
        os.path.join(SCRIPT_DIR, file_name),
        os.path.join(os.getcwd(), file_name),
    ):
        if os.path.isfile(candidate):
            return candidate
 
    return os.path.join(SCRIPT_DIR, file_name)
 
 
def select_binding_source():
    """Ask whether to use only experimental information or data with imputation/prediction."""
    print("\nBinding data type:")
    print(f"  [1] Experimental information only  -> {BINDING_FILE_EXPERIMENTAL}")
    print(f"  [2] With predicted information     -> selectivity: {BINDING_FILE_PREDICTED}")
    print(f"                                       affinity:    {BINDING_FILE_PREDICTED_AFFINITY}")
 
    while True:
        raw = input("Select [1/2, default 1]: ").strip().lower()
        if raw in ("", "1", "experimental", "exp", "e"):
            return BINDING_FILE_EXPERIMENTAL, "experimental"
        if raw in ("2", "predicted", "prediction", "pred", "p"):
            return BINDING_FILE_PREDICTED, "predicted"
        print("  x Invalid option. Choose 1 or 2.")
 
 
# -- READ AND CLEAN THE CSV FILES --------------------------------------------
 
def convert_ki_to_float(raw_value):
    """Convert a Ki (affinity) value to a float in uM."""
    if pd.isna(raw_value):
        return np.nan
 
    text = str(raw_value).strip().replace('\xa0', '').replace(' ', '')
 
    if text.startswith('>'):
        try:
            number = float(re.sub(r'[^\d.]', '', text))
            return number * 1.1
        except ValueError:
            return np.nan
 
    try:
        return float(text)
    except ValueError:
        return np.nan
 
 
def load_csv_files(binding_file_path, info_file_path):
    """
    Read the two CSV files and return:
      - binding_table   : VOCs x OBPs matrix with Ki values as floats
      - obp_info_table  : metadata table for each OBP
      - cas_col         : name of the CAS column
      - name_col        : name of the VOC-name column
      - obp_name_list   : list of the OBP column names
    """
    print(f"Reading binding matrix: {binding_file_path}")
    raw_binding = pd.read_csv(binding_file_path)
    print(f"  -> {raw_binding.shape[0]} VOCs and {raw_binding.shape[1] - 2} OBPs")
 
    print(f"Reading OBP information: {info_file_path}")
    obp_info_table = pd.read_csv(info_file_path)
    print(f"  -> {len(obp_info_table)} OBPs with metadata\n")
 
    cas_col       = raw_binding.columns[0]
    name_col      = raw_binding.columns[1]
    obp_name_list = list(raw_binding.columns[2:])
 
    binding_table = raw_binding.copy()
    for col_name in obp_name_list:
        binding_table[col_name] = binding_table[col_name].apply(convert_ki_to_float)
 
    # Fixed literature anchors to normalize s1 (Cui et al.):
    # ceiling 1 uM (excellent affinity) / floor 40 uM (weak affinity).
    LITERATURE_MIN = 1.0
    LITERATURE_MAX = 40.0
    print(f"  -> Ki anchors (literature): {LITERATURE_MIN:.1f} - {LITERATURE_MAX:.1f} uM")
 
    return binding_table, obp_info_table, cas_col, name_col, obp_name_list, LITERATURE_MIN, LITERATURE_MAX
 
 
# -- READ THE INTERFERENT FILE -----------------------------------------------
 
def read_interferent_file(file_path):
    """Read the interferent file. Lines with # or empty lines are ignored."""
    interferent_list = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'):
                continue
            interferent_list.append(clean_line)
    return interferent_list
 
 
# -- VALIDATE AND SELECT A VOC -----------------------------------------------
 
def validate_voc(binding_table, name_col, cas_col, search_text, file_path=None):
    """
    Search and validate a VOC by name or CAS, with three priority levels:
      1) Exact CAS match  (e.g. "100-52-7")
      2) Exact name match (case-insensitive)  -> "1-hexanol" != "2-ethyl-1-hexanol"
      3) Partial name match (interactive fallback)
 
    Modes:
      - file_path=None : interactive mode -> ask the user to choose if there are several partial matches
      - file_path=path : file mode        -> show an error and return None if it is not an exact match
    """
    search_text = search_text.strip()
    is_cas = bool(re.match(r'^\d+-\d+-\d+$', search_text))
 
    # -- 1) Exact CAS match --------------------------------------------------
    cas_exact = binding_table[cas_col].astype(str).str.strip() == search_text
    if cas_exact.any():
        return binding_table[cas_exact].iloc[0]
 
    # -- 2) Exact name match (case-insensitive) ------------------------------
    if not is_cas:
        name_exact = binding_table[name_col].str.strip().str.lower() == search_text.lower()
        if name_exact.any():
            return binding_table[name_exact].iloc[0]
 
    # -- 3) Partial name search (fallback) -----------------------------------
    if not is_cas:
        name_partial = binding_table[name_col].str.contains(
            search_text, case=False, na=False, regex=False
        )
        found_partial = binding_table[name_partial]
    else:
        found_partial = pd.DataFrame()
 
    # -- Not found at any level ----------------------------------------------
    if found_partial.empty:
        if is_cas:
            print(f"\n  x ERROR: CAS '{search_text}' does not exist in the database.")
            print(f"    Check that the CAS number is correct.")
        else:
            # Keyword suggestions
            suggestions = set()
            for word in [w for w in search_text.split() if len(w) >= 3]:
                hits = binding_table[
                    binding_table[name_col].str.contains(word, case=False, na=False, regex=False)
                ][name_col].tolist()
                suggestions.update(hits)
 
            print(f"\n  x ERROR: '{search_text}' was not found in the database.")
            if suggestions:
                print(f"    Maybe you meant one of these VOCs:")
                for s in sorted(suggestions)[:10]:
                    cas_val = binding_table.loc[binding_table[name_col] == s, cas_col].values
                    cas_str = cas_val[0] if len(cas_val) > 0 else "?"
                    print(f"      - {s}  (CAS: {cas_str})")
            else:
                print(f"    No similar VOCs found in the database.")
 
        if file_path:
            print(f"    Fix it in the file: {file_path}")
        return None
 
    # -- A single partial match -> return it directly ------------------------
    if len(found_partial) == 1:
        return found_partial.iloc[0]
 
    # -- Multiple partial matches --------------------------------------------
    print(f"\n  '{search_text}': {len(found_partial)} partial matches found:")
    for i, (_, row) in enumerate(found_partial.iterrows()):
        print(f"    [{i+1}]  {row[name_col][:70]}  (CAS: {row[cas_col]})")
 
    if file_path:
        # File mode: we do not ask for input, the user must specify the exact name or CAS
        print(f"    Specify the exact name or CAS in the file: {file_path}")
        return None
 
    # Interactive mode: the user chooses
    while True:
        choice = input(f"  Choose a number [1-{len(found_partial)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(found_partial):
            chosen = found_partial.iloc[int(choice) - 1]
            print(f"  -> Selected: {chosen[name_col]}  (CAS: {chosen[cas_col]})")
            return chosen
        print("  Invalid number. Try again.")
 
 
def validate_interferent_list(interferent_list, binding_table, name_col, cas_col, file_path):
    """
    Validate all interferents from the file using validate_voc.
    Show ALL errors and stop the program if there is any.
    The user must fix the .txt file manually and run again.

    Returns the list with the CANONICAL NAMES of each interferent (so a CAS
    typed in the file becomes its VOC name), so the rest of the program works
    only with names and never displays raw CAS numbers. When a name holds
    several synonyms separated by '/', the shortest one is kept (usually the
    common, most readable name).
    """
    errors_found = False
    canonical_names = []
    for entry in interferent_list:
        result = validate_voc(binding_table, name_col, cas_col, entry, file_path=file_path)
        if result is None:
            errors_found = True
        else:
            parts = [p.strip() for p in str(result[name_col]).split('/')]
            name = min(parts, key=len)   # shortest synonym (usually the common name)
            canonical_names.append(name)

    if errors_found:
        print(f"\n  == Fix the errors in the interferent file and run again. ==")
        sys.exit(1)

    return canonical_names
 
# -- ASK THE USER FOR WEIGHTS ------------------------------------------------
 
def ask_user_for_tau():
    """Ask the user for the tau value for the selectivity criterion."""
    print(f"\n  Tau (t) for the selectivity criterion:")
    print(f"    s2 = min(1,  Ki_interferent / (Ki_target x t))")
    print(f"     1 = the interferent must simply bind less than the target (lax)")
    print(f"    10 = the interferent must bind 10x weaker (recommended)")
    print(f"    50 = strict criterion, for very similar targets and interferents")
    print(f"  Valid range: [1, 100]")
    while True:
        raw = input(f"  Tau value [default 5]: ").strip()
        if raw == "":
            return 5.0
        try:
            tau = float(raw)
        except ValueError:
            print(f"  x '{raw}' is not a valid number. Enter a value between 1 and 100.")
            continue
        if tau <= 0:
            print(f"  x tau <= 0 causes division by zero or negative scores. Must be >= 1.")
        elif tau < 1:
            print(f"  x tau < 1 makes no biological sense: it would reward OBPs that bind the")
            print(f"    interferent equal to or stronger than the target. Use a value between 1 and 100.")
        elif tau > 100:
            print(f"  x tau > 100 is too restrictive: almost no OBP will pass the criterion.")
            print(f"    Use a value between 1 and 100.")
        else:
            return tau
 
 
def ask_user_for_weights(lock_selectivity_zero=False):
    """
    lock_selectivity_zero=True: selectivity (s2) cannot be evaluated for any
    OBP (none has all interferents measured) -> it is locked to 0 and its
    default weight is redistributed proportionally among affinity, stability
    and promiscuity. The selectivity weight is not asked from the user.
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
        print(f"\n  Selectivity (s2) locked to 0: it cannot be reliably evaluated for any candidate.")
        print(f"  New default weights (selectivity weight redistributed among the rest):")
    else:
        default_weights = dict(DEFAULT_WEIGHTS)
        print(f"  Default weights:")
 
    print(f"    Affinity (s1)     : {default_weights['w_affinity']:.2f}")
    print(f"    Selectivity (s2)  : {default_weights['w_selectivity']:.2f}"
          + ("  (locked)" if lock_selectivity_zero else ""))
    print(f"    Stability (s4)    : {default_weights['w_stability']:.2f}")
    print(f"    Promiscuity (s5)  : {default_weights['w_promiscuity']:.2f}")
    print(f"  Sum: {sum(default_weights.values()):.2f}")
 
    use_default = input("Use default weights? [Y/n]: ").strip().lower()
    if use_default not in ('n', 'no'):
        return dict(default_weights)
 
    custom_weights = {}
    weight_names = {
        "w_affinity":    "Affinity (s1)",
        "w_selectivity": "Selectivity (s2)",
        "w_stability":   "Stability (s4)",
        "w_promiscuity": "Promiscuity (s5)",
    }
 
    print("\n  Enter the weights (numbers between 0 and 1).")
    print("  They need not sum to 1: the program will normalize them automatically.\n")
 
    for key, label in weight_names.items():
        if lock_selectivity_zero and key == 'w_selectivity':
            custom_weights[key] = 0.0
            print(f"  Weight for {label}: 0.00  (locked - no reliable selectivity data)")
            continue
        while True:
            raw = input(f"  Weight for {label} [default {default_weights[key]:.2f}]: ").strip()
            if raw == "":
                custom_weights[key] = default_weights[key]
                break
            try:
                value = float(raw)
                if value < 0:
                    print("    The weight cannot be negative.")
                    continue
                custom_weights[key] = value
                break
            except ValueError:
                print("    Enter a valid number (e.g. 0.4).")
 
    total = sum(custom_weights.values())
    if total == 0:
        print("  All weights are 0. Default weights will be used.")
        return dict(default_weights)
 
    normalized = {k: v / total for k, v in custom_weights.items()}
 
    # Constraint: stability = promiscuity (always equal)
    mean_sp = (normalized['w_stability'] + normalized['w_promiscuity']) / 2.0
    normalized['w_stability']   = mean_sp
    normalized['w_promiscuity'] = mean_sp
 
    print(f"\n  Normalized weights (sum = 1.0):")
    for key, label in weight_names.items():
        print(f"    {label:<25}: {normalized[key]:.3f}")
 
    return normalized
 
 
# -- SUB-SCORE COMPUTATION ---------------------------------------------------
 
def compute_s1_affinity(ki_target, ki_min_matrix, ki_max_matrix):
    if pd.isna(ki_target) or ki_target <= 0:
        return 0.0
    if ki_min_matrix <= 0 or pd.isna(ki_min_matrix):
        return 0.0
    if ki_max_matrix <= 0 or pd.isna(ki_max_matrix):
        return 0.0
 
    log_target = -np.log10(ki_target)
    log_best  = -np.log10(ki_min_matrix)
    log_worst = -np.log10(ki_max_matrix)
 
    denominator = log_best - log_worst
    if denominator == 0:
        return 0.0
 
    s1 = (log_target - log_worst) / denominator
    return float(np.clip(s1, 0.0, 1.0))
 
 
def compute_s2_selectivity(ki_target, ki_min_interferent, tau=10.0):
    # Not evaluable: the target Ki or the interferent Ki is missing.
    # Return the NEUTRAL value, not 0.0: 0.0 is the score of an OBP measured
    # against an interferent it binds just as well, i.e. genuinely terrible
    # selectivity. Saying that about an OBP nobody has measured is a claim the
    # data does not support. has_s2_data=False makes this value be discarded
    # and the s2 weight renormalized, so the neutral value never reaches the
    # score; it only keeps the printed table honest.
    if pd.isna(ki_min_interferent) or pd.isna(ki_target) or ki_target <= 0:
        return S2_NEUTRAL
 
    ratio = ki_min_interferent / (ki_target * tau)
    return float(min(1.0, ratio))
 
 
def compute_s5_promiscuity(ki_target, ki_competitors):
    if pd.isna(ki_target) or ki_target <= 0:
        return 0.0  # value 0 when we have no reliable target Ki
 
    valid = pd.Series(ki_competitors).dropna()
    valid = valid[valid > 0]
 
    if valid.empty:
        return 0.0  # value 0 when there are no valid competitors with positive Ki
 
    ratios = ki_target / valid
 
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
 
 
# -- BUILD THE RANKING -------------------------------------------------------
 
def _ascii_safe(text):
    """Replace typographic punctuation and drop any remaining non-ASCII, so the
    name can be printed on a cp1252 console and drawn in the PDF."""
    table = {0x2010: '-', 0x2011: '-', 0x2012: '-', 0x2013: '-', 0x2014: '-',
             0x2212: '-', 0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
             0x00A0: ' ', 0x00B7: '.', 0x03B1: 'alpha', 0x03B2: 'beta',
             0x03B3: 'gamma', 0x03B4: 'delta', 0x00B5: 'u'}
    cleaned = str(text).translate(table)
    return cleaned.encode('ascii', 'ignore').decode('ascii')


def build_obp_ranking(ki_values_target, obp_info_table, binding_table,
                      cas_col, name_col, interferent_list, obp_name_list, weights,
                      ki_min_matrix, ki_max_matrix, tau=10.0,
                      experimental_ki_values_target=None, binding_source_label="experimental",
                      ki_values_target_affinity=None, target_name=None):
    """
    ki_values_target           : target VOC Ki used for selectivity (s2),
                                 promiscuity (s5) and as the main displayed value.
    ki_values_target_affinity  : target VOC Ki used ONLY for affinity (s1).
                                 If not provided (or the OBP is missing), the
                                 ki_values_target value is used as fallback.
    target_name                : name of the target VOC, used to skip it when it
                                 appears in its own interferent danger list.
    """
 
    info_by_name  = obp_info_table.set_index('Binding Protein Name')
    n_vocs_total  = len(binding_table)
 
    obp_rows = []
    for obp_name in obp_name_list:
 
        ki_target = ki_values_target.get(obp_name, np.nan)
 
        if pd.isna(ki_target):
            continue
 
        ki_target_affinity = np.nan
        if ki_values_target_affinity is not None:
            ki_target_affinity = ki_values_target_affinity.get(obp_name, np.nan)
        if pd.isna(ki_target_affinity):
            ki_target_affinity = ki_target
 
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
 
        experimental_ki = None
        if experimental_ki_values_target is not None:
            experimental_ki = experimental_ki_values_target.get(obp_name, np.nan)
 
        s1_source_label = infer_s1_source_label(
            selected_value=ki_target_affinity,
            experimental_value=experimental_ki,
            binding_source_label=binding_source_label,
        )
 
        obp_rows.append({
            'OBP':          obp_name,
            'Ki_target_uM':  ki_target,
            'Ki_target_affinity_uM': ki_target_affinity,
            'Type':         obp_type,
            'Cystines':     cys_count,
            'Preferred':    is_preferred,
            'Species':      species,
            'UniProtID':    uniprot_id,
            'Alphafold':    alphafold,
            'N_VOCs_bound': n_vocs_bound,
            's1_source_label': s1_source_label,
        })
 
    if not obp_rows:
        return pd.DataFrame()
 
    result_table = pd.DataFrame(obp_rows)
 
    # -- Selectivity: minimum interferent Ki for each OBP --------------------
    ki_per_interferent = {}
 
    interferent_row_idx = {}
    if interferent_list:
        print(f"\n[.] Applying selectivity filter ({len(interferent_list)} interferents)...")
        for interferent_name in interferent_list:
            chosen_row = validate_voc(binding_table, name_col, cas_col, interferent_name)
 
            if chosen_row is None:
                print(f"  x '{interferent_name}' not found - ignored.")
                continue
 
            print(f"  ok '{interferent_name}': found -> {chosen_row[name_col]}  (CAS: {chosen_row[cas_col]})")
 
            ki_series = pd.Series({col: chosen_row[col] for col in obp_name_list})
            ki_per_interferent[interferent_name] = ki_series
            interferent_row_idx[interferent_name] = chosen_row.name
 
    # -- For each OBP compute the minimum interferent Ki ---------------------
    # and which interferents could NOT be measured (missing) for this OBP.
    min_interferent_ki_list  = []
    worst_interferent_list   = []
    missing_interferent_list = []
    danger_interf_list       = []          # danger list of vineyard interferents per OBP
 
    n_interferents_total = len(ki_per_interferent)
 
    for _, obp_row in result_table.iterrows():
        current_obp  = obp_row['OBP']
        ki_target_row = obp_row['Ki_target_uM']
        min_ki_found = np.nan
        worst_name   = "-"
        missing      = []
        danger       = []
 
        for interf_name, ki_series in ki_per_interferent.items():
            ki_i = ki_series.get(current_obp, np.nan)
            if pd.isna(ki_i):
                missing.append(interf_name)
                continue
            if pd.isna(min_ki_found) or ki_i < min_ki_found:
                min_ki_found = ki_i
                worst_name   = interf_name
            # Dangerous vineyard interferent: binds equal, stronger, or a bit
            # weaker than the target (within the margin). Only from MEASURED
            # data -> the alert is reliable: what is missing cannot erase it,
            # at most it would add new ones. The target itself is skipped.
            if pd.notna(ki_target_row) and ki_target_row > 0 and \
               ki_i <= ki_target_row * DANGER_INTERFERENT_MARGIN:
                if str(interf_name).strip().lower() == str(target_name).strip().lower():
                    continue   # this is the target itself, not an interferent
                danger.append((interf_name, float(ki_i)))
 
        danger.sort(key=lambda t: t[1])   # most dangerous (lowest Ki) first
        min_interferent_ki_list.append(min_ki_found)
        worst_interferent_list.append(worst_name)
        missing_interferent_list.append(missing)
        danger_interf_list.append(danger)
 
    result_table['Min_Ki_interferent_uM'] = min_interferent_ki_list
    result_table['Worst_interferent']     = worst_interferent_list
    result_table['Missing_interferents']  = missing_interferent_list
    result_table['N_interferents_total']   = n_interferents_total
    result_table['N_interferents_missing'] = result_table['Missing_interferents'].apply(len)
    result_table['Danger_interferents']         = danger_interf_list
    result_table['N_danger_interferents']       = [len(p) for p in danger_interf_list]
    # has_all_interferents: ALL interferents were measured for this OBP
    # -> s2 is reliable, because what is missing cannot hide a lower minimum.
    result_table['has_all_interferents'] = (
        (n_interferents_total > 0) & (result_table['N_interferents_missing'] == 0)
    )
 
    # -- Compute the sub-scores for each OBP ---------------------------------
    s1_list = []
    s2_list = []
    s4_list = []
    s5_list = []
    n_competitors_s5_list = []
 
    for _, obp_row in result_table.iterrows():
        s1 = compute_s1_affinity(
            ki_target=obp_row['Ki_target_affinity_uM'],
            ki_min_matrix=ki_min_matrix,
            ki_max_matrix=ki_max_matrix
        )
        s2 = compute_s2_selectivity(
            ki_target=obp_row['Ki_target_uM'],
            ki_min_interferent=obp_row['Min_Ki_interferent_uM'],
            tau=tau
        )
        s4 = compute_s4_stability(obp_type=obp_row['Type'])
 
        # -- Promiscuity (formula C, logarithmic) ----------------------------
        # Take the whole Ki column for this OBP, excluding the target.
        ki_column      = binding_table[obp_row['OBP']]
        ki_competitors = ki_column[ki_column != obp_row['Ki_target_uM']]
 
        s5 = compute_s5_promiscuity(
            ki_target=obp_row['Ki_target_uM'],
            ki_competitors=ki_competitors
        )
 
        # Number of competitors actually used in the s5 computation (same
        # filter as compute_s5_promiscuity: no NaN and Ki > 0).
        # Needed for c5: confidence in s5 cannot depend only on a binary
        # (has_s5_data) but on how much evidence there is behind it.
        valid_competitors = ki_competitors.dropna()
        n_competitors_s5 = int((valid_competitors > 0).sum())
 
        s1_list.append(s1)
        s2_list.append(s2)
        s4_list.append(s4)
        s5_list.append(s5)
        n_competitors_s5_list.append(n_competitors_s5)
 
 
    result_table['s1_affinity']    = s1_list
    result_table['s2_selectivity'] = s2_list
    result_table['s4_stability']   = s4_list
    result_table['s5_promiscuity'] = s5_list
    result_table['N_competitors_s5'] = n_competitors_s5_list

    # -- Superbinders: VOCs from the WHOLE database this OBP binds strongly --
    # Two signals per VOC: '*' if Ki <= SUPERBINDER_KI_MAX (strong in absolute
    # terms), 'v' if it binds stronger than the target. We keep any VOC that
    # triggers at least one signal. Informative only; does not touch the score.
    voc_names = binding_table[name_col].tolist()
    superbinder_list = []
    for _, obp_row in result_table.iterrows():
        current_obp   = obp_row['OBP']
        ki_col        = binding_table[current_obp]
        ki_target_row = obp_row['Ki_target_uM']
        supers = []
        for voc_name, ki_v in zip(voc_names, ki_col):
            if pd.isna(ki_v) or ki_v <= 0:
                continue
            is_strong = ki_v <= SUPERBINDER_KI_MAX
            beats_target = (pd.notna(ki_target_row) and ki_target_row > 0
                            and ki_v < ki_target_row
                            and not np.isclose(ki_v, ki_target_row))
            if not (is_strong or beats_target):
                continue
            if str(voc_name).strip().lower() == str(target_name).strip().lower():
                continue   # skip the target itself
            # A name may carry synonyms separated by '/': keep the shortest.
            # Fragments shorter than 3 chars are not names but leftovers of a
            # racemic prefix ('(+/-) 3-Carene' -> '(+', '-) 3-Carene'), so they
            # are dropped and the dangling prefix is trimmed off the winner.
            parts = [p.strip() for p in str(voc_name).split('/') if len(p.strip()) >= 3]
            short = min(parts, key=len) if parts else str(voc_name)
            short = _ascii_safe(re.sub(r'^[^A-Za-z0-9(\[]+', '', short))
            supers.append((short, float(ki_v), bool(is_strong), bool(beats_target)))
        supers.sort(key=lambda t: t[1])   # strongest (lowest Ki) first
        superbinder_list.append(supers)

    result_table['Superbinders']   = superbinder_list
    result_table['N_superbinders'] = [len(s) for s in superbinder_list]
 
    # -- Data-availability flags ---------------------------------------------
    # has_s2_data: True if at least one interferent was measured for this OBP
    # AND the target Ki is usable. Both are needed: with either one missing
    # compute_s2_selectivity returns the neutral value, which must be
    # discarded rather than scored as if it were a measured selectivity.
    result_table['has_s2_data'] = (
        ~result_table['Min_Ki_interferent_uM'].isna()
        & result_table['Ki_target_uM'].notna()
        & (result_table['Ki_target_uM'] > 0)
    )
    # has_s4_data: True if the OBP type is recognized in the TYPE_SCORES dict
    result_table['has_s4_data'] = ~result_table['s4_stability'].isna()
    # has_s5_data: True if the OBP has measurements for more than one VOC (there are competitors)
    result_table['has_s5_data'] = result_table['N_VOCs_bound'] > 1
 
    # -- Final score: weights renormalized for the available criteria --------
    # If a criterion has no data it is discarded and the remaining weights are renormalized.
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
 
    # -- Score without s2: ALWAYS discards selectivity (s1+s4+s5 renormalized) --
    # Used for OBPs with incomplete interferents, where s2 is not reliable
    # (what is missing could hide the real minimum).
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
 
    result_table['Score_without_s2'] = scores_no_s2
 
    result_table = result_table.sort_values(
        by='Score', ascending=False
    ).reset_index(drop=True)
 
    result_table.attrs['interferent_row_idx'] = interferent_row_idx
    return result_table
 
 
# -- TABLES 1/2/3 - reliable selectivity / promising / worklist (Mode 1) -----
 
def show_table1_honest(result_table, how_many, binding_source_label="experimental"):
    """Table 1 - OBPs with ALL interferents measured: reliable s2,
    honest comparison with the 4 criteria (s1+s2+s4+s5).
 
    If no OBP has all the data, an empty table (or one with made-up values)
    is NOT shown: it warns and jumps straight to Table 2.
    """
    df = result_table[result_table['has_all_interferents']].reset_index(drop=True)
    if df.empty:
        sep = "-" * 88
        print(f"\n{sep}")
        print(f"  TABLE 1 - Selectivity NOT RELIABLE")
        n_interferents_total = (result_table['N_interferents_total'].iloc[0]
                                 if 'N_interferents_total' in result_table.columns
                                 and not result_table.empty else 0)
        if n_interferents_total == 0:
            print(f"  x No interferent data - selectivity cannot be evaluated.")
        else:
            print(f"  x No OBP has ALL interferents measured - selectivity is not known")
            print(f"    reliably for any candidate.")
        print(f"    Jumping straight to Table 2.")
        print(sep)
        return
    sep = "-" * 88
    print(f"\n{sep}")
    print(f"  TABLE 1 - RELIABLE selectivity (all interferents measured) - top {how_many}")
    print(f"  Honest comparison with the 4 criteria (renormalized weights)")
    print(sep)
    show_source_col = str(binding_source_label or "").strip().lower() == "predicted"
    header = f"  {'#':>3}  {'OBP':<22} {'Score':>7}  {'s1':>5}"
    if show_source_col:
        header += f"  {'s1 src':>12}"
    header += f"  {'s2':>5}  {'s4':>5}  {'s5':>5}  {'Ki_sel(uM)':>10}"
    if show_source_col:
        header += f"  {'Ki_af(uM)':>10}"
    header += f"  {'Ki_interf':>9}  {'Type':<12}"
    print(header)
    print(sep)
    for pos, (_, row) in enumerate(df.head(how_many).iterrows()):
        ki_i = (f"{row['Min_Ki_interferent_uM']:.2f}"
                if not pd.isna(row['Min_Ki_interferent_uM']) else "   -  ")
        s1_src = row.get('s1_source_label', '-')
        line = f"  {pos+1:>3}. {row['OBP']:<22} {row['Score']:>7.4f}  {row['s1_affinity']:>5.3f}"
        if show_source_col:
            line += f"  {str(s1_src):>12}"
        line += f"  {row['s2_selectivity']:>5.3f}  {row['s4_stability']:>5.3f}  {row['s5_promiscuity']:>5.3f}  {row['Ki_target_uM']:>10.2f}"
        if show_source_col:
            ki_af = row.get('Ki_target_affinity_uM', np.nan)
            line += f"  {ki_af:>10.2f}" if not pd.isna(ki_af) else f"  {'-':>10}"
        line += f"  {ki_i:>9}  {str(row['Type']):<12}"
        print(line)
    print(sep)
    print(f"  Total with all interferents measured: {len(df)}")
 
 
def show_table2_promising(result_table, how_many):
    """Table 2 - OBPs missing at least 1 interferent: s2 discarded
    (what is missing could be the minimum) -> s1+s4+s5 renormalized.
    Returns the ordered subset, to build Table 3 with the same priority."""
    df = result_table[~result_table['has_all_interferents']].copy()
    if df.empty:
        return df
 
    df = df.sort_values('Score_without_s2', ascending=False).reset_index(drop=True)
 
    sep = "-" * 88
    print(f"\n{sep}")
    print(f"  TABLE 2 - Promising IGNORING selectivity (missing >=1 interferent) - top {how_many}")
    print(f"  s2 discarded because it is unknown -> score with s1+s4+s5 renormalized")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<22} {'Score':>7}  {'s1':>5}  {'s4':>5}  {'s5':>5}  "
          f"{'Ki(uM)':>8}  {'Missing':>7}  {'Type':<12}")
    print(sep)
    for pos, (_, row) in enumerate(df.head(how_many).iterrows()):
        s4_disp = f"{row['s4_stability']:.3f}" if not pd.isna(row['s4_stability']) else "  ? "
        print(f"  {pos+1:>3}. {row['OBP']:<22} {row['Score_without_s2']:>7.4f}  "
              f"{row['s1_affinity']:>5.3f}  {s4_disp:>5}  {row['s5_promiscuity']:>5.3f}  "
              f"{row['Ki_target_uM']:>8.2f}  {row['N_interferents_missing']:>7}  {str(row['Type']):<12}")
    print(sep)
    print(f"  Total with incomplete interferents: {len(df)}")
    return df
 
 
def show_table3_boltz_worklist(df_table2_sorted):
    """Table 3 - Boltz worklist: OBP/interferent pairs that are missing and
    must be computed. Ordered by Table 2: first those of the most promising
    OBPs (worth computing first)."""
    if df_table2_sorted is None or df_table2_sorted.empty:
        return
 
    pairs = []
    for _, row in df_table2_sorted.iterrows():
        for interf in sorted(row['Missing_interferents']):
            pairs.append((row['OBP'], row['Score_without_s2'], interf))
 
    if not pairs:
        sep = "-" * 70
        print(f"\n{sep}")
        print(f"  TABLE 3 - Boltz worklist (missing OBP/interferent pairs)")
        print(f"  x No interferents defined - there is no OBP/interferent pair to compute.")
        print(sep)
        return
 
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  TABLE 3 - Boltz worklist (missing OBP/interferent pairs)")
    print(f"  Ordered by Table 2 priority (compute the top ones first)")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<22} {'Score T2':>9}  {'Missing interferent':<30}")
    print(sep)
    for i, (obp, score, interf) in enumerate(pairs, start=1):
        print(f"  {i:>3}. {obp:<22} {score:>9.4f}  {interf:<30}")
    print(sep)
    print(f"  Total OBP/interferent pairs to compute: {len(pairs)}")
 
 
# -- TABLES 1/2/3 - shared across the 3 modes --------------------------------
 
def show_tables_1_2_3(result_table, how_many, binding_source_label="experimental"):
    """Show the unified candidate list and the worklist: Tables 1/2/3 merged."""
    is_predicted = str(binding_source_label).lower() == "predicted"
    cf.show_unified_table(result_table, how_many, is_predicted, MIN_CONF_FILTER)
    cf.show_dangerous_vineyard_interferents(result_table, how_many, DANGER_INTERFERENT_MARGIN, MAX_DANGER_NAMES)
    cf.show_superbinders(result_table, how_many, SUPERBINDER_KI_MAX, MAX_SUPERBINDER_NAMES)
    cf.show_worklist(result_table)
 
 
# -- PROBABILISTIC RESULTS (Modes 2 and 3) -----------------------------------
 
def show_smaa_results(result_table, rank_accept, how_many, central_w, criterion_names=None):
    """Show the SMAA probabilistic ranking.
 
    criterion_names: labels of the criteria actually explored (in the order of the
    weights in central_w). If selectivity is locked to 0, it is not included
    here - so the weight profile does not show it as an explored dimension.
    """
    m = len(result_table)
    n_rank_cols = min(m, 5)
    sep = "=" * 95
 
    print(f"\n{sep}")
    print("  SMAA PROBABILISTIC RANKING")
    print(sep)
 
    rank_header = "  ".join(f"{'p('+str(i+1)+'r)':>7}" for i in range(n_rank_cols))
    print(f"  {'OBP':<24} {'p(best)':>10}  {rank_header}  bar")
    print("-" * 95)
 
    order = np.argsort(-rank_accept[:, 0])
    for obp_idx in order[:how_many]:
        row    = result_table.iloc[obp_idx]
        p_best = rank_accept[obp_idx, 0]
        probs  = "  ".join(f"{rank_accept[obp_idx, pos]:>7.3f}"
                           for pos in range(n_rank_cols))
        bar    = "#" * int(p_best * 40)
        print(f"  {row['OBP']:<24} {p_best:>10.3f}  {probs}  {bar}")
 
    print(sep)
 
    if central_w is not None:
        names = criterion_names if criterion_names is not None else ['Affinity', 'Selectiv.', 'Stability', 'Promisc.']
        width = 9 * len(names) + 2 * (len(names) - 1) + 26
        print(f"\n  Typical weight profiles (in which weight world each candidate wins):")
        header = "  ".join(f"{name:>9}" for name in names)
        print(f"  {'OBP':<24}  {header}")
        print("-" * width)
        for obp_idx in order[:how_many]:
            cw = central_w.get(obp_idx)
            if cw is None:
                continue
            name = result_table.iloc[obp_idx]['OBP']
            values = "  ".join(f"{v:>9.3f}" for v in cw)
            print(f"  {name:<24}  {values}")
        print("-" * width)
 
 
# -- MAIN --------------------------------------------------------------------
 
def main():
 
    print("SCENTINEL CODE - OBP candidate selection for a target VOC")
    print("Selection of the best OBP candidate for a target VOC\n")
 
    binding_file_name, binding_source = select_binding_source()
    binding_file = resolve_data_file(binding_file_name)
    info_file = resolve_data_file(INFO_FILE)
 
    for csv_path in (binding_file, info_file):
        if not os.path.isfile(csv_path):
            print(f"ERROR: File '{os.path.basename(csv_path)}' not found.")
            print(f"Searched in: {os.path.dirname(csv_path) or '.'}")
            sys.exit(1)
 
    print(f"\nUsing the binding matrix: {binding_file} ({binding_source})")
    binding_table, obp_info_table, cas_col, name_col, obp_name_list, LITERATURE_MIN, LITERATURE_MAX = load_csv_files(
        binding_file, info_file
    )
 
    # -- Search the target VOC (by name or CAS) ------------------------------
    print("You can search the target VOC by name or CAS number (e.g. 3391-86-4)")
    while True:
        user_query = input("Name or CAS of the target VOC: ").strip()
        if not user_query:
            print("  You must type something.")
            continue
 
        chosen_voc = validate_voc(binding_table, name_col, cas_col, user_query)
 
        if chosen_voc is None:
            print(f"  Try again.")
            continue
 
        print(f"\n  Selected VOC: {chosen_voc[name_col]}  (CAS: {chosen_voc[cas_col]})")
        break
 
    voc_display   = chosen_voc[name_col]
    voc_name_safe = re.sub(r'[^\w]+', '_', voc_display)[:40]
 
    ki_values_target = pd.Series({col: chosen_voc[col] for col in obp_name_list})
    experimental_ki_values_target = None
    if binding_source == "predicted":
        experimental_binding_table, _, _, _, _, _, _ = load_csv_files(
            resolve_data_file(BINDING_FILE_EXPERIMENTAL), info_file
        )
        experimental_voc = validate_voc(
            experimental_binding_table,
            name_col,
            cas_col,
            chosen_voc[cas_col],
        )
        if experimental_voc is not None:
            experimental_ki_values_target = pd.Series(
                {col: experimental_voc[col] for col in obp_name_list}
            )
 
    # -- With predicted information: affinity (s1) is computed from a separate --
    # file (BINDING_FILE_PREDICTED_AFFINITY), while binding_table (BINDING_FILE_PREDICTED)
    # still serves for selectivity (s2) and the rest of the analysis.
    ki_values_target_affinity = None
    if binding_source == "predicted":
        affinity_file = resolve_data_file(BINDING_FILE_PREDICTED_AFFINITY)
        if not os.path.isfile(affinity_file):
            print(f"ERROR: Predicted affinity file '{os.path.basename(affinity_file)}' not found.")
            print(f"Searched in: {os.path.dirname(affinity_file) or '.'}")
            sys.exit(1)
 
        print(f"Using the predicted affinity matrix: {affinity_file}")
        affinity_binding_table, _, _, _, _, _, _ = load_csv_files(affinity_file, info_file)
        affinity_voc = validate_voc(
            affinity_binding_table,
            name_col,
            cas_col,
            chosen_voc[cas_col],
        )
        if affinity_voc is not None:
            ki_values_target_affinity = pd.Series(
                {col: affinity_voc[col] for col in obp_name_list}
            )
        else:
            print(f"  ! VOC not found in {BINDING_FILE_PREDICTED_AFFINITY}; "
                  f"the values from {BINDING_FILE_PREDICTED} will be used for affinity too.")
 
    # -- Interferent file ----------------------------------------------------
 
    print("\nThe interferent file can contain names or CAS numbers, one per line.")
 
    interferent_list = []
    while True:
        interf_path = input(
            "Interferent file (leave empty to skip): "
        ).strip().strip('"').strip("'")
 
        if not interf_path:
            print("  -> Continuing without interferents.")
            break
 
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents loaded: {interferent_list}")
            print(f"\n  Validating interferents against the database...")
            interferent_list = validate_interferent_list(
                interferent_list, binding_table, name_col, cas_col, interf_path
            )
            print(f"\n  ok {len(interferent_list)} valid interferents: {interferent_list}")
            break
 
        print(f"  x File '{interf_path}' not found.")
        print(f"    Try again or leave empty to continue without interferents.")
 
    # Without interferents, selectivity (s2) can never be evaluated -> lock it
    # the first time we ask for the weights (avoids asking twice).
    has_interferents = bool(interferent_list)
    weights = ask_user_for_weights(lock_selectivity_zero=not has_interferents)
    tau     = ask_user_for_tau()
 
    top_input = input("\nHow many candidates do you want to see? [default: 10]: ").strip()
    how_many  = int(top_input) if top_input.isdigit() and int(top_input) > 0 else 10
 
    print("\nComputing ranking...")
    result_table = build_obp_ranking(
        ki_values_target=ki_values_target,
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
        experimental_ki_values_target=experimental_ki_values_target,
        binding_source_label=binding_source,
        ki_values_target_affinity=ki_values_target_affinity,
        target_name=voc_display,
    )
 
    if result_table.empty:
        print("  No OBP has data for this VOC.")
        sys.exit(0)
 
    diag_c1 = cf.load_diag_c1(resolve_data_file(DIAG_FILE))
    if diag_c1 is None and binding_source == "predicted":
        print(f"  ! Cannot find {DIAG_FILE}: c1 of the imputed cells will be {cf.C1_FALLBACK}")
 
    result_table = cf.compute_confidences(
        result_table, weights,
        target_row_idx=chosen_voc.name,
        interferent_row_idx=result_table.attrs.get('interferent_row_idx', {}),
        diag_c1=diag_c1,
        is_predicted=(binding_source == "predicted"),
    )
 
    # -- The "no interferents" case is already locked when asking the weights.
    # Here only the case remains where there ARE interferents but no OBP has
    # ALL of them measured: this cannot be known until the ranking is built.
    has_table1_data = result_table['has_all_interferents'].any() if has_interferents else False
 
    if has_interferents and not has_table1_data:
        result_table = cf.compute_confidences(
            result_table, weights,
            target_row_idx=chosen_voc.name,
            interferent_row_idx=result_table.attrs.get('interferent_row_idx', {}),
            diag_c1=diag_c1,
            is_predicted=(binding_source == "predicted"),
        )
 
    # ===========================================================
    # MODE SELECTION
    # ===========================================================
    sep = "=" * 82
    print(f"\n{sep}")
    print("  ANALYSIS MODE")
    print(sep)
    print("  [1] Deterministic weights  - single, fast, transparent result")
    print("  [2] Perturbation (c%)      - robust probability around your weights")
    print("  [3] Pure SMAA              - full exploration of the weight space")
    mode_raw = input("\n  Choose mode [1/2/3, default 1]: ").strip()
    mode_num = mode_raw if mode_raw in ("1", "2", "3") else "1"
 
    # If selectivity is locked to 0 (no candidate can evaluate it reliably),
    # it is excluded ENTIRELY as a criterion - not only from the deterministic
    # computation. Otherwise Modes 2/3 would explore random weights for a
    # criterion we know is noise, and the typical weight profile would show it
    # as if it were a real decision dimension.
    selectivity_active = weights['w_selectivity'] > 0.0
 
    criterion_names = ['Affinity', 'Stability', 'Promisc.']
    base_weights_list = [
        weights['w_affinity'], weights['w_stability'], weights['w_promiscuity'],
    ]
    cols_subscores = ['s1_affinity', 's4_stability', 's5_promiscuity']
    cols_has_data = [
        np.ones(len(result_table), dtype=bool),
        np.ones(len(result_table), dtype=bool),
        result_table['has_s5_data'].values,
    ]
    if selectivity_active:
        criterion_names.insert(1, 'Selectiv.')
        base_weights_list.insert(1, weights['w_selectivity'])
        cols_subscores.insert(1, 's2_selectivity')
        cols_has_data.insert(1, result_table['has_s2_data'].values)
 
    subscores = result_table[cols_subscores].values
    has_data = np.column_stack(cols_has_data)
 
    # -- Mode 1: Deterministic weights ---------------------------------------
    if mode_num == "1":
        show_tables_1_2_3(result_table, how_many, binding_source_label=binding_source)
 
    # -- Mode 2: Perturbation around the weights -----------------------------
    elif mode_num == "2":
        print(f"\n  Dispersion c% - how far the weights can move (as a % of the gap to the adjacent criterion):")
        print(f"    10 = very concentrated (weights almost fixed, +-10% of the minimum gap between criteria)")
        print(f"    50 = half of the gap  (recommended)")
        print(f"   100 = almost a tie between the two closest criteria")
        c_raw = input("  Value of c [default 50]: ").strip()
        try:
            c_val = float(c_raw)
            if not (0.1 <= c_val <= 100):
                raise ValueError
        except ValueError:
            c_val = 50.0
            print(f"  -> Using c={c_val:.0f}%")
 
        n_raw = input("  Number of iterations [default 200000]: ").strip()
        n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 200_000
 
        print(f"\nRunning weight perturbation ({n_iter:,} iterations, c={c_val:.0f}%)...")
        rank_accept, _, _ = smaa_full(
            subscores=subscores, has_data=has_data,
            n_iter=n_iter, base_weights=base_weights_list, c=c_val,
            mode="perturbation", seed=42,
        )
        show_smaa_results(result_table, rank_accept, how_many, central_w=None,
                           criterion_names=criterion_names)
        show_tables_1_2_3(result_table, how_many, binding_source_label=binding_source)
 
    # -- Mode 3: Fully random SMAA -------------------------------------------
    elif mode_num == "3":
        n_raw = input("  Number of iterations [default 200000]: ").strip()
        n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 200_000
 
        print(f"\nRunning pure SMAA ({n_iter:,} iterations)...")
        rank_accept, _, central_w = smaa_full(
            subscores=subscores, has_data=has_data,
            n_iter=n_iter, mode="smaa", seed=42,
        )
        show_smaa_results(result_table, rank_accept, how_many, central_w=central_w,
                           criterion_names=criterion_names)
        show_tables_1_2_3(result_table, how_many, binding_source_label=binding_source)
 
 
if __name__ == '__main__':
    main()