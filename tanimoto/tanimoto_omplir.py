
import os
import re
import numpy as np
import pandas as pd

import tanimoto_impute as ti

BINDING_FILE = "Compound_OBP_binding.csv"

# Paràmetres triats al pas 2 (tanimoto_calibrar.py)
T_MIN = 0.35
K     = 8
P     = 2.0
Z     = 1.0   # z=1 → Ki_lower a 1 sigma (conservador estàndard del protocol)

OUT_LOWER = "Compound_OBP_binding_imputed_lower.csv"
OUT_PRED  = "Compound_OBP_binding_imputed_pred.csv"
OUT_DIAG  = "results/tanimoto_imputation_diagnostic.csv"

os.makedirs("results", exist_ok=True)


def convert_ki(raw):

    if pd.isna(raw):
        return np.nan
    text = str(raw).strip().replace('\xa0', '').replace(' ', '')
    if text.startswith('>'):
        try:
            return float(re.sub(r'[^\d.]', '', text)) * 1.1
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def main():
    print("=" * 70)
    print("  OMPLIR BUITS - PROTOCOL TANIMOTO-Ki")
    print(f"  Paràmetres: T_MIN={T_MIN}  K={K}  P={P}  Z={Z}")
    print("=" * 70)

    print(f"\n  Llegint {BINDING_FILE}...")
    df_raw = pd.read_csv(BINDING_FILE)
    cas_col, name_col = df_raw.columns[0], df_raw.columns[1]
    obp_name_list = list(df_raw.columns[2:])

    binding_table = df_raw.copy()
    for col in obp_name_list:
        binding_table[col] = binding_table[col].apply(convert_ki)

    n_total = len(binding_table) * len(obp_name_list)
    n_known = int(binding_table[obp_name_list].notna().sum().sum())
    n_empty = n_total - n_known
    print(f"  → {len(binding_table)} VOCs × {len(obp_name_list)} OBPs = {n_total} cel·les")
    print(f"  → {n_known} amb Ki conegut, {n_empty} buides per omplir")

    print(f"\n  Carregant matriu Tanimoto...")
    T_matrix, cas_to_pos = ti.load_tanimoto_data()
    idx_to_pos = ti.build_binding_idx_to_matrix_pos(binding_table, cas_col, cas_to_pos)
    print(f"  → {len(idx_to_pos)}/{len(binding_table)} VOCs amb posició vàlida a la matriu")

    print(f"\n  Omplint buits (read-across)...")
    binding_lower, binding_pred, diag_table = ti.fill_gaps(
        binding_table, T_matrix, idx_to_pos, obp_name_list,
        t_min=T_MIN, k=K, p=P, z=Z,
    )

    #  Desar les dues taules amples 
    binding_lower.to_csv(OUT_LOWER, index=False)
    binding_pred.to_csv(OUT_PRED, index=False)
    print(f"\n   Taula CONSERVADORA (Ki_lower) desada a {OUT_LOWER}")
    print(f"   Taula CENTRAL (Ki_pred) desada a {OUT_PRED}")

    #  Desar diagnòstic en format llarg, amb noms 
    if not diag_table.empty:
        cas_lookup  = df_raw[cas_col].astype(str).str.strip()
        name_lookup = df_raw[name_col]
        diag_table["CAS"] = diag_table["row_idx"].map(cas_lookup)
        diag_table["Compound_name"] = diag_table["row_idx"].map(name_lookup)
        diag_table = diag_table[
            ["row_idx", "CAS", "Compound_name", "OBP",
             "Ki_pred_uM", "Ki_lower_uM", "T_max", "sigma_log",
             "n_donors", "decision"]
        ]
        diag_table.to_csv(OUT_DIAG, index=False)
        print(f"   Diagnòstic detallat ({len(diag_table)} files) desat a {OUT_DIAG}")
    else:
        print("   Cap cel·la s'ha pogut imputar (revisa T_MIN o la matriu).")

    # Resum final 
    n_imputed = len(diag_table)
    n_unfillable = n_empty - n_imputed

    print("\n" + "=" * 70)
    print("  RESUM FINAL")
    print("=" * 70)
    print(f"  Cel·les experimentals          : {n_known}")
    print(f"  Cel·les imputades (vNN)        : {n_imputed}")
    print(f"  Cel·les encara buides          : {n_unfillable}  "
          f"(cap veí amb Tanimoto≥{T_MIN} → candidates a GNINA/Boltz-2)")

    if not diag_table.empty:
        print(f"\n  Desglossament per decisió:")
        print(diag_table["decision"].value_counts().to_string())
        print(f"\n  T_max mitjà de les imputacions : {diag_table['T_max'].mean():.3f}")
        print(f"  sigma_log mitjà                : {diag_table['sigma_log'].mean():.3f}")

    print(f"\n  Següent pas: adaptar main_nou.py perquè:")
    print(f"    - faci servir '{OUT_LOWER}' per calcular selectivitat (s2)")
    print(f"    - faci servir '{OUT_PRED}' per mostrar el Ki estimat a l'usuari")
    print(f"    - llegeixi '{OUT_DIAG}' per marcar Source=imputed_vNN al ranking")


if __name__ == "__main__":
    main()