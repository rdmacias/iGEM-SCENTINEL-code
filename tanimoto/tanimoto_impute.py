"""
tanimoto_impute.py
─
Mòdul base per al protocol Tanimoto-Ki (read-across / vNN).
Implementa exactament les funcions del Tanimoto_Ki_Protocol.docx:
  - impute_ki_tanimoto()  → predicció puntual (Ki_pred + Ki_lower conservador)
  - run_loo()             → validació Leave-One-Out (RMSE)
  - fill_gaps()            → omple tots els buits de binding_table

Connecta amb la matriu Tanimoto que ja tens generada
(tanimoto_data/tanimoto_matrix.npy + tanimoto_data/voc_index.json), fent l'enllaç
CAS ↔ posició a la matriu ↔ fila de binding_table.

--------------------------------------------------------------------
Core module for the Tanimoto-Ki protocol (read-across / weighted k-NN).
Implements the functions described in Tanimoto_Ki_Protocol.docx:
  - impute_ki_tanimoto()  -> single-cell prediction (central Ki_pred +
                              conservative Ki_lower)
  - run_loo()              -> Leave-One-Out validation (RMSE)
  - fill_gaps()             -> fills every missing cell in binding_table

Links the compound metadata (CAS + SMILES) to the Tanimoto similarity
matrix, connecting: CAS number <-> position in the matrix <-> row in
binding_table.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs
except Exception:  # pragma: no cover - rdkit may be absent in some environments
    Chem = None
    rdFingerprintGenerator = None
    DataStructs = None


def _resolve_compound_info_path(compound_info_path=None):
    # Try the explicit path first, then fall back to a few known default
    # locations for the compound metadata CSV (relative to this file).
    if compound_info_path:
        candidate = Path(compound_info_path)
        if candidate.is_file():
            return candidate

    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "Compound_info (1).csv",
        base_dir / "Compound_info.csv",
        base_dir.parent / "tanimoto" / "Compound_info (1).csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_compound_info_smiles(compound_info_path=None):
    """Load CAS -> SMILES mapping from the compound metadata CSV."""
    info_path = _resolve_compound_info_path(compound_info_path)
    if info_path is None:
        raise FileNotFoundError("No s'ha trobat el fitxer de metadades amb SMILES")

    df = pd.read_csv(info_path)
    normalized = {col.strip().lower(): col for col in df.columns}

    cas_col = normalized.get("cas-number") or normalized.get("cas number") or normalized.get("cas")
    smiles_col = (
        normalized.get("smiles")
        or normalized.get("smiles_string")
        or normalized.get("canonicalsmiles")
        or normalized.get("isomericsmiles")
    )

    if cas_col is None or smiles_col is None:
        raise ValueError(f"El fitxer {info_path} no té columnes CAS/SMILES reconegudes")

    # Build CAS -> SMILES dict, keeping the first SMILES seen for each CAS
    mapping = {}
    for _, row in df.iterrows():
        cas = str(row[cas_col]).strip() if pd.notna(row[cas_col]) else ""
        smiles = str(row[smiles_col]).strip() if pd.notna(row[smiles_col]) else ""
        if cas and smiles:
            mapping.setdefault(cas, smiles)

    return mapping, info_path


def build_tanimoto_matrix_from_smiles(smiles_values, radius=2, n_bits=2048):
    """Build a Tanimoto similarity matrix from a list of SMILES strings."""
    if Chem is None or rdFingerprintGenerator is None or DataStructs is None:
        raise ImportError("rdkit is required to build the Tanimoto matrix")

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # Convert each SMILES string to a Morgan fingerprint (None if missing/invalid)
    fingerprints = []
    for smiles in smiles_values:
        if pd.isna(smiles) or not str(smiles).strip():
            fingerprints.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            fingerprints.append(None)
        else:
            fingerprints.append(morgan_gen.GetFingerprint(mol))

    # Pairwise Tanimoto similarity matrix; entries with a missing fingerprint stay NaN
    n = len(fingerprints)
    matrix = np.full((n, n), np.nan, dtype=np.float32)
    valid_idx = [i for i, fp in enumerate(fingerprints) if fp is not None]

    for i in valid_idx:
        fp_i = fingerprints[i]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, [fingerprints[j] for j in valid_idx])
        for sim, j in zip(sims, valid_idx):
            matrix[i, j] = sim

    return matrix


def load_tanimoto_data(matrix_path=None, index_path=None, compound_info_path=None):
    """Load the Tanimoto matrix and its CAS index.

    If matrix_path and index_path both point to existing files (as written
    by tanimoto.py), load them directly — this avoids recomputing RDKit
    fingerprints on every run. Otherwise, rebuild the matrix live from the
    compound metadata CSV's SMILES column.
    """
    if matrix_path and index_path and Path(matrix_path).is_file() and Path(index_path).is_file():
        T_matrix = np.load(matrix_path)
        with open(index_path, "r", encoding="utf-8") as f:
            cas_index = json.load(f)
        cas_to_pos = {cas: i for i, cas in enumerate(cas_index)}
        return T_matrix, cas_to_pos

    compound_info, info_path = load_compound_info_smiles(compound_info_path)
    cas_index = list(compound_info.keys())
    smiles_list = [compound_info[cas] for cas in cas_index]
    T_matrix = build_tanimoto_matrix_from_smiles(smiles_list)
    cas_to_pos = {cas: i for i, cas in enumerate(cas_index)}
    return T_matrix, cas_to_pos


def build_binding_idx_to_matrix_pos(binding_table, cas_col, cas_to_pos):
    # Map each binding_table row index to its position (row/col) in T_matrix,
    # via the shared CAS number. Rows whose CAS isn't found in the matrix are
    # simply left out of the mapping.
    mapping = {}
    for row_idx, cas in binding_table[cas_col].astype(str).str.strip().items():
        if cas in cas_to_pos:
            mapping[row_idx] = cas_to_pos[cas]
    return mapping


#  POINT PREDICTION (Tanimoto-weighted read-across / k-NN)

def impute_ki_tanimoto(x_idx, obp_col, binding_table, T_matrix, idx_to_pos,
                        t_min=0.35, k=5, p=2.0, z=1.0, exclude_idx=None):
    """Predict a single missing Ki value for (x_idx, obp_col) by averaging
    the k most Tanimoto-similar VOCs' known Ki values (weighted by
    similarity^p), restricted to donors with similarity >= t_min.

    Returns a dict with the central prediction (Ki_pred_uM), a conservative
    lower-confidence estimate (Ki_lower_uM, shifted by z standard deviations
    in pKi space), and diagnostics (T_max, sigma_log, n_donors). Returns None
    if the target has no matrix position or no eligible donors.
    """
    if x_idx not in idx_to_pos:
        return None

    # Ensure numeric parameter types to avoid dtype issues later
    try:
        t_min = float(t_min)
    except Exception:
        t_min = 0.35
    try:
        k = int(k)
    except Exception:
        k = 5
    try:
        p = float(p)
    except Exception:
        p = 2.0
    try:
        z = float(z)
    except Exception:
        z = 1.0

    pos_x = idx_to_pos[x_idx]
    col = binding_table[obp_col]

    # Collect candidate donors: other VOCs with a known, positive Ki for this
    # OBP and Tanimoto similarity to the target >= t_min.
    donors = []
    for cand_idx, pos_cand in idx_to_pos.items():
        if cand_idx == x_idx or cand_idx == exclude_idx:
            continue
        ki_cand = col.get(cand_idx, np.nan)
        if pd.isna(ki_cand) or ki_cand <= 0:
            continue
        t = float(T_matrix[pos_x, pos_cand])
        if np.isnan(t) or t < t_min:
            continue
        donors.append((cand_idx, ki_cand, t))

    if not donors:
        return None

    donors.sort(key=lambda d: -d[2])   # sort by Tanimoto similarity, descending
    donors = donors[:k]                # keep only the k nearest neighbours

    ki_vals = np.asarray([d[1] for d in donors], dtype=float)
    T_vals = np.asarray([d[2] for d in donors], dtype=float)

    # convert to safe numeric types
    ki_vals = ki_vals.astype(float)
    T_vals = T_vals.astype(float)
    p = float(p)

    pKi = -np.log10(ki_vals * 1e-6)    # µM → M → pKi
    # Similarity^p used as the weight of each donor (use numpy power to be explicit and robust)
    w = np.power(T_vals, p)

    # Weighted mean pKi prediction, plus a weighted standard deviation (sigma)
    # used to build the conservative lower-confidence bound below.
    pKi_hat = float((w * pKi).sum() / w.sum())
    sigma = float(np.sqrt((w * (pKi - pKi_hat) ** 2).sum() / w.sum()))

    return {
        "Ki_pred_uM":  10 ** (-pKi_hat) * 1e6,               # central estimate
        "Ki_lower_uM": 10 ** (-(pKi_hat + z * sigma)) * 1e6,  # conservative estimate (higher Ki = weaker binding)
        "T_max":       float(T_vals.max()),                  # similarity of the closest donor
        "sigma_log":   sigma,                                 # spread of donor pKi values (uncertainty)
        "n_donors":    len(donors),
    }


#  LEAVE-ONE-OUT (LOO) VALIDATION

def run_loo(binding_table, T_matrix, idx_to_pos, obp_name_list,
            t_min=0.35, k=5, p=2.0):
    """For every measured Ki cell, temporarily hide it (exclude_idx) and
    predict it from the remaining data, then compare prediction vs. the real
    value. Returns the overall RMSE (in pKi/log units) and a DataFrame with
    one row of prediction error per validated cell.
    """
    errors = []
    total_cols = len(obp_name_list)

    for col_i, obp_col in enumerate(obp_name_list, start=1):
        measured = binding_table[obp_col].dropna()
        measured = measured[measured > 0]

        for target_idx in measured.index:
            real_ki = measured[target_idx]
            # exclude_idx=target_idx hides the true value so it can't be used
            # to predict itself (this is what makes it "leave-one-out")
            result = impute_ki_tanimoto(
                target_idx, obp_col, binding_table, T_matrix, idx_to_pos,
                t_min=t_min, k=k, p=p, exclude_idx=target_idx,
            )
            if result is None:
                continue
            pred_pki = -np.log10(result["Ki_pred_uM"] * 1e-6)
            real_pki = -np.log10(real_ki * 1e-6)
            errors.append({
                "obp": obp_col,
                "voc_idx": target_idx,
                "pred_pki": pred_pki,
                "real_pki": real_pki,
                "error": pred_pki - real_pki,
                "T_max": result["T_max"],
                "n_donors": result["n_donors"],
            })

        if col_i % 50 == 0 or col_i == total_cols:
            print(f"    LOO: {col_i}/{total_cols} columnes OBP processades "
                  f"({len(errors)} prediccions acumulades)...")

    if not errors:
        print("  LOO no ha pogut validar cap cel·la (revisa t_min/k).")
        return None, pd.DataFrame()

    df_err = pd.DataFrame(errors)
    rmse = float(np.sqrt((df_err["error"] ** 2).mean()))
    return rmse, df_err


def rmse_by_tmax_bin(df_err, bins=(0.35, 0.5, 0.7, 1.01)):
    # Break LOO results down by the similarity (T_max) of the best donor used,
    # so we can see whether prediction accuracy depends on how similar the
    # nearest known neighbour was.
    labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    df_err = df_err.copy()
    df_err["bin"] = pd.cut(df_err["T_max"], bins=bins, labels=labels, right=False)
    resumen = df_err.groupby("bin")["error"].apply(
        lambda e: np.sqrt((e ** 2).mean())
    ).rename("RMSE")
    counts = df_err.groupby("bin").size().rename("n_cel·les")
    return pd.concat([resumen, counts], axis=1)


#  FILL ALL REAL GAPS

def fill_gaps(binding_table, T_matrix, idx_to_pos, obp_name_list,
              t_min=0.35, k=5, p=2.0, z=1.0):
    """
    Omple TOTS els buits reals de binding_table (no és LOO, és l'aplicació
    final). Treballa sobre una còpia: mai sobreescriu cel·les mesurades.

    Retorna:
      binding_imputed : còpia de binding_table amb Ki_lower_uM a les cel·les
                         imputades (valor conservador, per a selectivitat)
      binding_pred     : còpia paral·lela amb Ki_pred_uM (valor central,
                         per a mostrar/reportar)
      diag_table       : taula llarga amb una fila per cada cel·la imputada
                         (CAS, OBP, Ki_pred, Ki_lower, T_max, sigma, n_donors,
                         decision: 'confident' / 'exploratory' / 'no_impute')
    """
    binding_imputed = binding_table.copy()
    binding_pred     = binding_table.copy()
    diag_rows = []

    total_cols = len(obp_name_list)
    for col_i, obp_col in enumerate(obp_name_list, start=1):
        missing_idx = binding_table[binding_table[obp_col].isna()].index

        for x_idx in missing_idx:
            result = impute_ki_tanimoto(
                x_idx, obp_col, binding_table, T_matrix, idx_to_pos,
                t_min=t_min, k=k, p=p, z=z,
            )
            if result is None:
                continue

            # Classify the imputation's reliability per the protocol's
            # decision table (based on donor similarity and prediction spread)
            if result["T_max"] >= 0.5 and result["sigma_log"] < 0.3:
                decision = "confident"
            elif result["T_max"] >= t_min:
                decision = "exploratory"
            else:
                decision = "no_impute"

            binding_imputed.at[x_idx, obp_col] = result["Ki_lower_uM"]
            binding_pred.at[x_idx, obp_col]     = result["Ki_pred_uM"]

            diag_rows.append({
                "row_idx":     x_idx,
                "OBP":         obp_col,
                "Ki_pred_uM":  result["Ki_pred_uM"],
                "Ki_lower_uM": result["Ki_lower_uM"],
                "T_max":       result["T_max"],
                "sigma_log":   result["sigma_log"],
                "n_donors":    result["n_donors"],
                "decision":    decision,
            })

        if col_i % 50 == 0 or col_i == total_cols:
            print(f"    Omplint: {col_i}/{total_cols} columnes OBP processades "
                  f"({len(diag_rows)} cel·les imputades fins ara)...")

    diag_table = pd.DataFrame(diag_rows)
    return binding_imputed, binding_pred, diag_table
