# Final imputation step: uses the best (t_min, k, p) parameters found by
# tanimoto_calibrate.py to fill in every missing Ki cell in the binding table
# via tanimoto_impute.fill_gaps(), then writes:
#   1. the "central prediction" CSV      (Ki_pred)
#   2. the "conservative lower bound" CSV (Ki_lower)
#   3. the diagnostic table              (T_max, sigma_log, decision, c1)
#
# (3) is essential: without it you have the imputed values but no trace of how
# reliable each one is. It is where the affinity confidence comes from:
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

CALIBRATION_SUMMARY = BASE_DIR / "results" / "loo_summary_combinations.csv"
OUTPUT_DIAG_DIRS = [BASE_DIR / "results", OBP_SEARCHER_DIR / "results"]
DIAG_FILENAME = "tanimoto_diag_table.csv"


# ─────────────────────── DECISION CONSTANTS ───────────────────────
# Conservativeness lever for the _lower file: how many sigmas we shift towards
# weaker binding. z=0 -> Ki_lower == Ki_pred. z=1 -> one step of uncertainty.
# It is NEVER calibrated (LOO only validates Ki_pred): it is your own call.
Z = 1.0

# Sigma normalization for the c1 confidence. sigma_log=1.0 means the donors
# disagree by a factor of 10x in Ki -> zero confidence.
SIGMA_MAX = 1.0

# Minimum fraction of EMPTY cells we want to be able to fill (hard floor).
# Note: the fill rate depends ONLY on t_min (k and p have no influence at all:
# k trims the donor list after the filter, p only weights it).
MIN_FILL = 0.30

# RMSE tolerance: within this margin the error difference is irrelevant and we
# prefer the combination that fills MORE of the matrix.
# 0.05 in log10 = about 12% relative precision in Ki: imperceptible when
# deciding whether an OBP is a good candidate. Set 0.0 for "RMSE rules" again.
RMSE_TOL = 0.05

# Fallback parameters used when no calibration is available
FALLBACK = (0.35, 5, 2.0)

# Output format. Do NOT use "%.2f": a Ki of 0.004 would become 0.00, and
# convert_ki + impute_ki_tanimoto discard values <= 0.
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
    Which fraction of the EMPTY cells could be imputed with this t_min?

    Nothing has to be predicted: it is enough to know whether some donor exists
    (a VOC with a measured Ki for that OBP) with similarity >= t_min. This is a
    matrix operation, far cheaper than running fill_gaps().
    """
    rows_ok = np.array(sorted(idx_to_pos.keys()))
    pos_ok = np.array([idx_to_pos[r] for r in rows_ok])

    n_empty = n_imputable = 0
    for obp in obp_name_list:
        col = binding_table[obp]
        measured = [r for r in rows_ok if pd.notna(col[r]) and col[r] > 0]
        empty_rows = binding_table.index[col.isna()]
        n_empty += len(empty_rows)

        if not measured:
            continue  # no possible donor in this column

        donor_pos = np.array([idx_to_pos[r] for r in measured])
        # empty rows that are also present in the Tanimoto matrix
        empty_ok = [r for r in empty_rows if r in idx_to_pos]
        if not empty_ok:
            continue
        empty_pos = np.array([idx_to_pos[r] for r in empty_ok])

        sub = T_matrix[np.ix_(empty_pos, donor_pos)]
        with np.errstate(invalid="ignore"):
            best = np.nanmax(np.where(np.isnan(sub), -np.inf, sub), axis=1)
        n_imputable += int((best >= t_min).sum())

    return (n_imputable / n_empty) if n_empty else 0.0


def get_best_params(binding_table, obp_name_list, T_matrix, idx_to_pos):
    """
    Pick (t_min, k, p) from the calibration, but under two separate constraints:

      - t_min  -> controls HOW MANY cells get filled (fill rate)
      - k, p   -> control HOW WELL they get filled (RMSE)

    The original calibration took the minimum RMSE without looking at coverage,
    and that can pick a high t_min that leaves half the matrix unimputed in
    exchange for an imperceptible error improvement (RMSE 0.30 -> 0.34 = factor
    2.00x -> 2.19x in Ki).
    """
    if not CALIBRATION_SUMMARY.is_file():
        print(f"  ! Cannot find {CALIBRATION_SUMMARY.name} -> fallback parameters {FALLBACK}")
        return FALLBACK

    try:
        df = pd.read_csv(CALIBRATION_SUMMARY)
    except Exception as e:
        print(f"  ! Cannot read the calibration ({e}) -> fallback {FALLBACK}")
        return FALLBACK

    if df.empty:
        print(f"  ! Empty calibration -> fallback {FALLBACK}")
        return FALLBACK

    # Real fill rate for each unique t_min (only 3-4 values: cheap)
    print("\n  Real fill rate over the EMPTY cells (depends only on t_min):")
    fill = {}
    for t in sorted(df["t_min"].unique()):
        fill[t] = fill_rate_for_t_min(binding_table, obp_name_list, T_matrix, idx_to_pos, float(t))
        mark = "OK " if fill[t] >= MIN_FILL else "discarded"
        print(f"     t_min = {t:<5} -> {fill[t]:6.1%}   {mark}")

    df = df.copy()
    df["fill_rate"] = df["t_min"].map(fill)

    valid = df[df["fill_rate"] >= MIN_FILL]
    if valid.empty:
        best = df.sort_values("RMSE").iloc[0]
        print(f"\n  ! No t_min reaches a fill rate >= {MIN_FILL:.0%}. "
              f"Taking the lowest RMSE one (fill {best['fill_rate']:.1%}).")
        rmse_ref = None
    else:
        # 1) the best possible RMSE among those that clear the fill floor
        rmse_ref = valid["RMSE"].min()
        # 2) among those close to it (within RMSE_TOL), take the one that
        #    fills MORE of the matrix. Fill-rate tie -> the lower RMSE.
        tied = valid[valid["RMSE"] <= rmse_ref + RMSE_TOL]
        best = tied.sort_values(["fill_rate", "RMSE"],
                                ascending=[False, True]).iloc[0]

    print(f"\n  Chosen combination:  t_min={best['t_min']}  "
          f"k={int(best['k'])}  p={best['p']}")
    print(f"     RMSE = {best['RMSE']:.3f}  (error factor ~{10**best['RMSE']:.2f}x in Ki)")
    print(f"     fill rate = {best['fill_rate']:.1%} of the empty cells")

    if rmse_ref is not None and best["RMSE"] > rmse_ref + 1e-9:
        f_ref = 10 ** rmse_ref
        f_chosen = 10 ** best["RMSE"]
        alt = valid[valid["RMSE"] <= rmse_ref + 1e-9].iloc[0]
        print(f"     (giving up RMSE {rmse_ref:.3f} / fill {alt['fill_rate']:.1%}: "
              f"the error factor goes from {f_ref:.2f}x to {f_chosen:.2f}x, "
              f"but I gain {best['fill_rate'] - alt['fill_rate']:+.1%} of matrix)")

    return float(best["t_min"]), int(best["k"]), float(best["p"])


def summarize_diag(diag_table):
    """Add the c1 column (affinity confidence) and print a summary."""
    if diag_table.empty:
        print("  ! diag_table is empty: no cell was imputed.")
        return diag_table

    d = diag_table.copy()
    d["c1"] = (d["T_max"] * (1.0 - d["sigma_log"] / SIGMA_MAX).clip(lower=0.0)).round(3)

    print("\n  Decision breakdown:")
    for dec, n in d["decision"].value_counts().items():
        print(f"     {dec:<14} {n:>8,}  ({n/len(d):.1%})")

    print(f"\n  T_max      -> median {d['T_max'].median():.3f}  ·  min {d['T_max'].min():.3f}")
    print(f"  sigma_log  -> median {d['sigma_log'].median():.3f}  ·  max {d['sigma_log'].max():.3f}")
    print(f"  c1         -> median {d['c1'].median():.3f}  ·  >0.7 in {(d['c1'] > 0.7).sum():,} cells")
    return d


def main():
    print("=" * 70)
    print("  GENERATING TANIMOTO IMPUTATION CSVS")
    print("=" * 70)

    if not BINDING_FILE.is_file():
        raise FileNotFoundError(f"Binding file not found: {BINDING_FILE}")

    print(f"\n  Reading {BINDING_FILE.name}...")
    df = pd.read_csv(BINDING_FILE)
    cas_col = df.columns[0]
    obp_name_list = list(df.columns[2:])

    binding_table = df.copy()
    for col in obp_name_list:
        binding_table[col] = binding_table[col].apply(convert_ki)

    n_cells = len(binding_table) * len(obp_name_list)
    n_known = int(binding_table[obp_name_list].notna().sum().sum())
    print(f"  -> {len(binding_table)} VOCs x {len(obp_name_list)} OBPs = {n_cells:,} cells")
    print(f"  -> {n_known:,} measured ({n_known/n_cells:.1%})")

    print(f"\n  Loading the Tanimoto matrix...")
    T_matrix, cas_to_pos = ti.load_tanimoto_data(
        matrix_path=str(MATRIX_FILE),
        index_path=str(INDEX_FILE),
    )
    idx_to_pos = ti.build_binding_idx_to_matrix_pos(binding_table, cas_col, cas_to_pos)
    n_outside = len(binding_table) - len(idx_to_pos)
    print(f"  -> {len(idx_to_pos)}/{len(binding_table)} VOCs with a valid position in the matrix")
    if n_outside:
        print(f"     ({n_outside} VOCs with no resolved SMILES: they get no imputation "
              f"and cannot act as donors)")

    # Parameter choice: t_min from the fill rate, k/p from the RMSE
    t_min, k, p = get_best_params(binding_table, obp_name_list, T_matrix, idx_to_pos)

    print(f"\n  Running fill_gaps() with: t_min={t_min}, k={k}, p={p}, z={Z}")
    binding_imputed, binding_pred, diag_table = ti.fill_gaps(
        binding_table, T_matrix, idx_to_pos, obp_name_list,
        t_min=t_min, k=k, p=p, z=Z,
    )

    # Affinity confidence for each imputed cell
    diag_table = summarize_diag(diag_table)

    # Warn if rounding could wipe out small values
    ki_min = float(binding_pred[obp_name_list].min().min())
    if ki_min < 0.005:
        print(f"\n  ! Minimum Ki = {ki_min:.5f} uM. With '%.2f' it would become 0.00 and be "
              f"discarded as invalid. Using {FLOAT_FMT}.")

    # binding_pred: central prediction values; binding_imputed: conservative
    # lower-bound values (safer for downstream selectivity decisions).
    print()
    for out_dir in OUTPUT_PRED_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / PRED_FILENAME
        binding_pred.to_csv(out_path, index=False, float_format=FLOAT_FMT)
        print(f"  Central CSV saved to: {out_path}")

    for out_dir in OUTPUT_LOWER_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / LOWER_FILENAME
        binding_imputed.to_csv(out_path, index=False, float_format=FLOAT_FMT)
        print(f"  Conservative CSV saved to: {out_path}")

    # THE LINE THAT WAS MISSING: without it you lose T_max/sigma/c1 for every
    # cell. Written to both directories (like PRED/LOWER): obp_searcher_aborrit/
    # main.py reads it from its own results/, not from the one in tanimoto/.
    for out_dir in OUTPUT_DIAG_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / DIAG_FILENAME
        diag_table.to_csv(out_path, index=False)
        print(f"  Diagnostic saved to:  {out_path}")

    n_filled = int(binding_pred[obp_name_list].notna().sum().sum())
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Filled cells: {n_known:,} -> {n_filled:,}  "
          f"({n_known/n_cells:.1%} -> {n_filled/n_cells:.1%})")
    print(f"  Imputed: {len(diag_table):,}  ·  still empty: {n_cells - n_filled:,}")


if __name__ == "__main__":
    main()
