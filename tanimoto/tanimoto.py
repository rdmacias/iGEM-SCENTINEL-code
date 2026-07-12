
# Builds the VOC x VOC Tanimoto similarity matrix from SMILES strings.
# Reads the binding table + compound info CSV, computes Morgan fingerprints
# for every VOC, and saves the pairwise Tanimoto similarity matrix plus its
# CAS-number index to disk (used later by tanimoto_impute.py).

import json
import os

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # silence RDKit's noisy parsing warnings

import tanimoto_impute as ti

# Input files
BINDING_FILE = "Compound_OBP_binding.csv"
COMPOUND_INFO_FILE = "Compound_info (1).csv"
# Output files (matrix + index used to look up a VOC's row/column by CAS number)
OUT_MATRIX_FILE = "tanimoto_data/tanimoto_matrix.npy"
OUT_INDEX_FILE = "tanimoto_data/voc_index.json"
OUT_MISSING_FILE = "tanimoto_data/voc_sense_smiles.csv"

# Morgan (ECFP-like) fingerprint parameters
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)

os.makedirs("tanimoto_data", exist_ok=True)


#  FINGERPRINTS AND TANIMOTO SIMILARITY

def build_fingerprint(smiles):
    # Convert a single SMILES string into a Morgan fingerprint bit vector.
    # Returns None if the SMILES is missing/empty or fails to parse.
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MORGAN_GEN.GetFingerprint(mol)


def build_tanimoto_matrix(fingerprints):
    # Compute the full pairwise Tanimoto similarity matrix for a list of
    # fingerprints. Entries involving a missing (None) fingerprint stay NaN.
    n = len(fingerprints)
    matrix = np.full((n, n), np.nan, dtype=np.float32)

    valid_idx = [i for i, fp in enumerate(fingerprints) if fp is not None]
    print(f"\n  Calculant Tanimoto per a {len(valid_idx)}/{n} VOCs amb SMILES vàlid...")

    for count, i in enumerate(valid_idx, 1):
        fp_i = fingerprints[i]
        # Bulk similarity: compare fingerprint i against all valid fingerprints at once
        sims = DataStructs.BulkTanimotoSimilarity(
            fp_i, [fingerprints[j] for j in valid_idx]
        )
        for sim, j in zip(sims, valid_idx):
            matrix[i, j] = sim
        if count % 50 == 0 or count == len(valid_idx):
            print(f"    {count}/{len(valid_idx)} VOCs processats...")

    return matrix




def main():
    print("="*70)
    print("  GENERACIÓ MATRIU TANIMOTO VOC×VOC")
    print("="*70)

    if not os.path.isfile(BINDING_FILE):
        print(f"   ERROR: no es troba '{BINDING_FILE}' en aquesta carpeta.")
        return

    print(f"\n  Llegint {BINDING_FILE}...")
    df = pd.read_csv(BINDING_FILE)
    cas_col, name_col = df.columns[0], df.columns[1]
    print(f"  {len(df)} VOCs trobats")

    # Look up each VOC's SMILES string by CAS number from the compound info file
    compound_info_smiles, info_path = ti.load_compound_info_smiles(COMPOUND_INFO_FILE)
    print(f"\n  Usant {info_path} com a font única de SMILES...")
    cas_list, smiles_list = [], []
    sense_smiles = []  # VOCs for which no SMILES could be resolved

    for idx, row in df.iterrows():
        cas = str(row[cas_col]).strip()
        name = row[name_col]
        smiles = compound_info_smiles.get(cas)

        cas_list.append(cas)
        smiles_list.append(smiles)

        status = "ok" if smiles else "no"
        if (idx + 1) % 25 == 0 or smiles is None:
            print(f"    [{idx+1}/{len(df)}] {status}  {str(name)[:50]}")

        if not smiles:
            sense_smiles.append({"CAS-number": cas, "Compound name": name})

    # Save the list of VOCs that couldn't be resolved for manual review
    if sense_smiles:
        pd.DataFrame(sense_smiles).to_csv(OUT_MISSING_FILE, index=False)
        print(f"   {len(sense_smiles)} VOCs sense SMILES resolt → {OUT_MISSING_FILE}")
        print(f"    (no participaran a la matriu Tanimoto; revisa'ls a mà si vols)")

    print(f"\n  Generant fingerprints Morgan (radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS})...")
    fingerprints = [build_fingerprint(s) for s in smiles_list]
    n_ok = sum(1 for fp in fingerprints if fp is not None)
    print(f"  → {n_ok}/{len(fingerprints)} fingerprints generats correctament")

    # Build the full similarity matrix and persist it alongside its CAS index
    matrix = build_tanimoto_matrix(fingerprints)

    np.save(OUT_MATRIX_FILE, matrix)
    print(f"\n   Matriu Tanimoto ({matrix.shape[0]}×{matrix.shape[1]}) desada a {OUT_MATRIX_FILE}")

    with open(OUT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(cas_list, f, indent=2, ensure_ascii=False)
    print(f"   Índex (ordre CAS) desat a {OUT_INDEX_FILE}")

    print("\n" + "="*70)
    print("  RESUM")
    print("="*70)
    print(f"  VOCs totals             : {len(df)}")
    print(f"  Amb SMILES resolt       : {n_ok}")
    print(f"  Sense SMILES (revisar)  : {len(sense_smiles)}")
    print(f"\n  Per carregar la matriu després:")
    print(f"    import numpy as np, json")
    print(f"    matrix = np.load('{OUT_MATRIX_FILE}')")
    print(f"    cas_index = json.load(open('{OUT_INDEX_FILE}'))")
    print(f"    i = cas_index.index('CAS_DEL_VOC')")
    print(f"    fila_similituds = matrix[i]   # Tanimoto contra tots els altres VOCs")


if __name__ == "__main__":
    main()