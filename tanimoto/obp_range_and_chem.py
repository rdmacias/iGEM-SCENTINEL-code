"""
obp_range_and_chem.py
─────────────────────────────────────────────────────────────────────────────
Two numbers for every OBP:

  1) AFFINITY DIVERSITY
     Is there a Ki gradient inside this OBP (very strong + medium + weak
     binders), or does everything sit at one extreme?

     Measured with the standard deviation of pKi (= -log10 Ki uM/1e6).
     pKi is log10, so:
        std(pKi) ~ 0.3  -> everything within one order of magnitude (narrow range)
        std(pKi) ~ 1.0  -> a factor of 10x between the extremes (normal range)
        std(pKi) ~ 1.5+ -> very strong and very weak binders at once (wide range)

     On top of that there is an "is everything represented?" check: the Ki
     range is split into three bands (strong / medium / weak) and we look at
     whether all three have members.

  2) CHEMICAL DIVERSITY
     MEAN Tanimoto similarity between pairs of this OBP's ligands.
     High = they are all chemically similar (one family).
     Low  = they are chemically diverse.

Combining the two yields 4 interesting kinds of OBP (see the thresholds
below). Output: results/obp_range_and_chem.csv, one row per OBP.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display window, only saves PNG files
import matplotlib.pyplot as plt


# ─────────────────────────── CONFIG ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BINDING_FILE = BASE_DIR / "Compound_OBP_binding.csv"
MATRIX_FILE = BASE_DIR / "tanimoto_data" / "tanimoto_matrix.npy"
INDEX_FILE = BASE_DIR / "tanimoto_data" / "voc_index.json"
OUTPUT_CSV = BASE_DIR / "results" / "obp_range_and_chem.csv"
OUTPUT_PNG = BASE_DIR / "results" / "obp_range_and_chem.png"
TEAL, GREY, RED, BLUE = "#0E7C7B", "#B9C7C5", "#C0392B", "#2471A3"

# "enough data" thresholds, only used to color the plot (they do not affect any computation)
N_DATA_MID = 10         # n_used > this -> mid color
N_DATA_HIGH = 25        # n_used > this -> strong color

# Only ligands with a Ki measured inside this range (uM) are counted.
# Filters out values clearly outside the biological scale.
KI_MIN_UM = 0.001
KI_MAX_UM = 1000.0

# Optional cap on the number of ligands considered per OBP.
TOP_N = None            # None = every valid measured ligand
MIN_BINDERS = 3         # minimum needed for std and tanimoto to mean anything

# Classification thresholds (in log10 of Ki, i.e. pKi)
STD_NARROW = 0.35       # std(pKi) <= this -> "narrow range"
STD_WIDE = 0.75         # std(pKi) >= this -> "wide range"

# Affinity bands for the "is everything represented?" check
# Ki < 5 uM = strong   ·   5-40 uM = medium   ·   >40 uM = weak
KI_STRONG = 5.0
KI_MEDIUM = 40.0

# Chemical similarity thresholds
CHEM_SIMILAR = 0.40     # mean tanimoto >= this -> chemical family
CHEM_DIVERS = 0.20      # mean tanimoto <= this -> chemically diverse
# ────────────────────────────────────────────────────────────────────────


def convert_ki(raw):
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


def load_binding(path):
    df = pd.read_csv(path)
    cas_col = df.columns[0]
    obp_cols = list(df.columns[2:])
    for c in obp_cols:
        df[c] = df[c].apply(convert_ki)
    return df, cas_col, obp_cols


def load_tanimoto(matrix_path, index_path):
    T_matrix = np.load(matrix_path)
    with open(index_path, encoding="utf-8") as f:
        cas_index = json.load(f)
    return T_matrix, {cas: i for i, cas in enumerate(cas_index)}


def build_idx_to_pos(df, cas_col, cas_to_pos):
    mapping = {}
    for row_idx, cas in df[cas_col].astype(str).str.strip().items():
        if cas in cas_to_pos:
            mapping[row_idx] = cas_to_pos[cas]
    return mapping


def classify_range(std_pki, has_strong, has_medium, has_weak):
    """Affinity diversity: narrow vs wide range, and whether there is "a bit of everything"."""
    if std_pki <= STD_NARROW:
        # Everything sits at one extreme. Which one?
        if has_strong and not has_weak:
            return "all strong"
        if has_weak and not has_strong:
            return "all weak"
        return "narrow range"
    if std_pki >= STD_WIDE:
        coverage = has_strong + has_medium + has_weak
        if coverage == 3:
            return "wide range (strong + medium + weak)"
        return "wide range"
    return "medium range"


def classify_chem(mean_t):
    if np.isnan(mean_t):
        return "no chemistry"
    if mean_t >= CHEM_SIMILAR:
        return "chemical family"
    if mean_t <= CHEM_DIVERS:
        return "chemically diverse"
    return "mixed chemistry"


def combined_type(range_cls, chem_cls):
    """The 4 quadrants + variants."""
    # wide range x chemically similar -> finely discriminating pocket
    if "wide" in range_cls and chem_cls == "chemical family":
        return "fine discriminator (same family, full range)"
    if "wide" in range_cls and chem_cls == "chemically diverse":
        return "free agent (accepts everything, discriminates by Ki)"
    if range_cls == "all strong" and chem_cls == "chemical family":
        return "super-selective (specific motif)"
    if range_cls == "all strong" and chem_cls == "chemically diverse":
        return "SUSPICIOUS (binds different things strongly)"
    if range_cls == "all weak":
        return "poor sensor (no strong binding)"
    return f"{range_cls} · {chem_cls}"


def analyze_obp(obp_col, df, idx_to_pos, T_matrix,
                ki_min=KI_MIN_UM, ki_max=KI_MAX_UM,
                top_n=TOP_N, min_binders=MIN_BINDERS):
    col = df[obp_col].dropna()
    col = col[(col >= ki_min) & (col <= ki_max)]
    n_measured = int(len(col))

    if top_n:
        col = col.sort_values().head(top_n)

    base = dict(OBP=obp_col, n_measured=n_measured, n_used=int(len(col)))

    if len(col) < min_binders:
        return {**base,
                "Ki_min_uM": np.nan, "Ki_max_uM": np.nan,
                "std_pki": np.nan, "range_pki": np.nan,
                "has_strong": False, "has_medium": False, "has_weak": False,
                "range_class": "insufficient data",
                "mean_tanimoto": np.nan, "n_pairs_chem": 0,
                "chem_class": "insufficient data",
                "combined_type": "insufficient data"}

    # --- AFFINITY DIVERSITY ---------------------------------------------
    pki = -np.log10(col.values * 1e-6)     # Ki (uM) -> pKi
    std_pki = float(np.std(pki, ddof=1)) if len(pki) > 1 else 0.0
    range_pki = float(pki.max() - pki.min())

    has_strong = bool((col < KI_STRONG).any())
    has_medium = bool(((col >= KI_STRONG) & (col < KI_MEDIUM)).any())
    has_weak = bool((col >= KI_MEDIUM).any())
    range_class = classify_range(std_pki, has_strong, has_medium, has_weak)

    # --- CHEMICAL DIVERSITY ----------------------------------------------
    valid_idx = [i for i in col.index if i in idx_to_pos]
    if len(valid_idx) < min_binders:
        mean_t = np.nan
        n_pairs = 0
    else:
        pos = [idx_to_pos[i] for i in valid_idx]
        sub = T_matrix[np.ix_(pos, pos)]
        iu = np.triu_indices(len(pos), k=1)
        vals = sub[iu]
        vals = vals[~np.isnan(vals)]
        mean_t = float(np.mean(vals)) if len(vals) else np.nan
        n_pairs = int(len(vals))

    chem_class = classify_chem(mean_t)

    return {
        **base,
        "Ki_min_uM": round(float(col.min()), 3),
        "Ki_max_uM": round(float(col.max()), 3),
        "std_pki": round(std_pki, 3),
        "range_pki": round(range_pki, 3),
        "has_strong": has_strong, "has_medium": has_medium, "has_weak": has_weak,
        "range_class": range_class,
        "mean_tanimoto": round(mean_t, 3) if not np.isnan(mean_t) else np.nan,
        "n_pairs_chem": n_pairs,
        "chem_class": chem_class,
        "combined_type": combined_type(range_class, chem_class),
    }


def plot_scatter(result, out_png):
    """Scatter of std(pKi) vs mean tanimoto, one point per OBP, colored only
    by how many measurements (n_used) back it, with the threshold lines that
    define the 4 quadrants of interest."""
    data = result.dropna(subset=["std_pki", "mean_tanimoto"])
    if data.empty:
        print("\n  (not enough valid data to draw the plot)")
        return

    few = data["n_used"] <= N_DATA_MID
    mid = (data["n_used"] > N_DATA_MID) & (data["n_used"] <= N_DATA_HIGH)
    high = data["n_used"] > N_DATA_HIGH

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(data.loc[few, "std_pki"], data.loc[few, "mean_tanimoto"],
               s=22, alpha=.5, edgecolors="none", color=GREY,
               label=f"n<={N_DATA_MID} (n={int(few.sum())})")
    ax.scatter(data.loc[mig, "std_pki"], data.loc[mig, "mean_tanimoto"],
               s=28, alpha=.8, edgecolors="none", color=RED,
               label=f"n>{N_DATA_MID} (n={int(mid.sum())})")
    ax.scatter(data.loc[high, "std_pki"], data.loc[high, "mean_tanimoto"],
               s=32, alpha=.9, edgecolors="none", color=TEAL,
               label=f"n>{N_DATA_HIGH} (n={int(high.sum())})")

    ax.axvline(STD_NARROW, color="k", lw=.8, ls="--")
    ax.axvline(STD_WIDE, color="k", lw=.8, ls="--")
    ax.axhline(CHEM_SIMILAR, color="k", lw=.8, ls=":")
    ax.axhline(CHEM_DIVERS, color="k", lw=.8, ls=":")

    ax.set_xlabel("std(pKi)  ·  AFFINITY diversity  (narrow -> wide)")
    ax.set_ylabel("Mean Tanimoto  ·  CHEMICAL diversity  (diverse -> family)")
    ax.set_title("Affinity diversity x chemical diversity per OBP")
    ax.legend(fontsize=7, markerscale=1.4, loc="upper left", framealpha=.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    print("=" * 74)
    print("  AFFINITY DIVERSITY x CHEMICAL DIVERSITY PER OBP")
    print("=" * 74)

    if not BINDING_FILE.is_file():
        raise FileNotFoundError(f"Cannot find {BINDING_FILE}")
    if not MATRIX_FILE.is_file() or not INDEX_FILE.is_file():
        raise FileNotFoundError(f"Cannot find the Tanimoto matrix ({MATRIX_FILE})")

    print(f"\n  Reading {BINDING_FILE.name}...")
    df, cas_col, obp_cols = load_binding(BINDING_FILE)
    print(f"  -> {len(df)} VOCs x {len(obp_cols)} OBPs")

    print(f"\n  Loading the Tanimoto matrix...")
    T_matrix, cas_to_pos = load_tanimoto(MATRIX_FILE, INDEX_FILE)
    idx_to_pos = build_idx_to_pos(df, cas_col, cas_to_pos)
    print(f"  -> {len(idx_to_pos)}/{len(df)} VOCs with a valid position in the matrix\n")

    print(f"  Ki bands: strong < {KI_STRONG} uM · medium {KI_STRONG}-{KI_MEDIUM} uM · weak > {KI_MEDIUM} uM")
    print(f"  std(pKi): narrow <= {STD_NARROW} · wide >= {STD_WIDE}")
    print(f"  Tanimoto: similar >= {CHEM_SIMILAR} · diverse <= {CHEM_DIVERS}\n")

    rows = []
    for i, obp in enumerate(obp_cols, 1):
        rows.append(analyze_obp(obp, df, idx_to_pos, T_matrix))
        if i % 100 == 0 or i == len(obp_cols):
            print(f"    {i}/{len(obp_cols)} OBPs processed...")

    result = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["combined_type", "n_measured"],
                       ascending=[True, False]).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)

    print(f"\n  Breakdown by AFFINITY RANGE:")
    for cls, n in result["range_class"].value_counts().items():
        print(f"     {cls:<40} {n:>5}")

    print(f"\n  Breakdown by CHEMICAL DIVERSITY:")
    for cls, n in result["chem_class"].value_counts().items():
        print(f"     {cls:<40} {n:>5}")

    print(f"\n  Breakdown by COMBINED TYPE:")
    for cls, n in result["combined_type"].value_counts().items():
        print(f"     {cls:<50} {n:>5}")

    # Examples from each interesting quadrant
    for kind in ["fine discriminator (same family, full range)",
                 "free agent (accepts everything, discriminates by Ki)",
                 "super-selective (specific motif)",
                 "SUSPICIOUS (binds different things strongly)"]:
        subset = result[result["combined_type"] == kind].head(5)
        if not subset.empty:
            print(f"\n  --- Examples of: {kind} ---")
            for _, r in subset.iterrows():
                print(f"     {r['OBP']:<20} std_pKi={r['std_pki']:.2f}  "
                      f"tanimoto={r['mean_tanimoto']:.2f}  "
                      f"n={r['n_used']}  ({r['Ki_min_uM']}-{r['Ki_max_uM']} uM)")

    plot_scatter(result, OUTPUT_PNG)

    print(f"\n  Saved to: {OUTPUT_CSV}")
    print(f"  Plot saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
