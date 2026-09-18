
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
OUT_MISSING_FILE = "tanimoto_data/voc_missing_smiles.csv"

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
    print(f"\n  Computing Tanimoto for {len(valid_idx)}/{n} VOCs with a valid SMILES...")

    for count, i in enumerate(valid_idx, 1):
        fp_i = fingerprints[i]
        # Bulk similarity: compare fingerprint i against all valid fingerprints at once
        sims = DataStructs.BulkTanimotoSimilarity(
            fp_i, [fingerprints[j] for j in valid_idx]
        )
        for sim, j in zip(sims, valid_idx):
            matrix[i, j] = sim
        if count % 50 == 0 or count == len(valid_idx):
            print(f"    {count}/{len(valid_idx)} VOCs processed...")

    return matrix




def main():
    print("="*70)
    print("  BUILDING THE VOC x VOC TANIMOTO MATRIX")
    print("="*70)

    if not os.path.isfile(BINDING_FILE):
        print(f"   ERROR: '{BINDING_FILE}' not found in this folder.")
        return

    print(f"\n  Reading {BINDING_FILE}...")
    df = pd.read_csv(BINDING_FILE)
    cas_col, name_col = df.columns[0], df.columns[1]
    print(f"  {len(df)} VOCs found")

    # Look up each VOC's SMILES string by CAS number from the compound info file
    compound_info_smiles, info_path = ti.load_compound_info_smiles(COMPOUND_INFO_FILE)
    print(f"\n  Using {info_path} as the single source of SMILES...")
    cas_list, smiles_list = [], []
    missing_smiles = []  # VOCs for which no SMILES could be resolved

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
            missing_smiles.append({"CAS-number": cas, "Compound name": name})

    # Save the list of VOCs that couldn't be resolved for manual review
    if missing_smiles:
        pd.DataFrame(missing_smiles).to_csv(OUT_MISSING_FILE, index=False)
        print(f"   {len(missing_smiles)} VOCs with no resolved SMILES -> {OUT_MISSING_FILE}")
        print(f"    (they are left out of the Tanimoto matrix; review them by hand if you want)")

    print(f"\n  Generating Morgan fingerprints (radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS})...")
    fingerprints = [build_fingerprint(s) for s in smiles_list]
    n_ok = sum(1 for fp in fingerprints if fp is not None)
    print(f"  -> {n_ok}/{len(fingerprints)} fingerprints generated successfully")

    # Build the full similarity matrix and persist it alongside its CAS index
    matrix = build_tanimoto_matrix(fingerprints)

    np.save(OUT_MATRIX_FILE, matrix)
    print(f"\n   Tanimoto matrix ({matrix.shape[0]}x{matrix.shape[1]}) saved to {OUT_MATRIX_FILE}")

    with open(OUT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(cas_list, f, indent=2, ensure_ascii=False)
    print(f"   Index (CAS order) saved to {OUT_INDEX_FILE}")

    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Total VOCs              : {len(df)}")
    print(f"  With resolved SMILES    : {n_ok}")
    print(f"  Without SMILES (review) : {len(missing_smiles)}")
    print(f"\n  To load the matrix later:")
    print(f"    import numpy as np, json")
    print(f"    matrix = np.load('{OUT_MATRIX_FILE}')")
    print(f"    cas_index = json.load(open('{OUT_INDEX_FILE}'))")
    print(f"    i = cas_index.index('VOC_CAS')")
    print(f"    similarity_row = matrix[i]   # Tanimoto against every other VOC")


if __name__ == "__main__":
    main()