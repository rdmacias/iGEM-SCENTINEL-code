#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_analysis_families_range.py
--------------------------------
Analyzes Leave-One-Out errors (loo_errors_detail.csv) along two branches:

  BRANCH 1 - by OBP FAMILY (Classic OBP, PBP, CSP, GOBP1/2, Minus-C, ...):
             RMSE, bias (pred-real), MAE, n, and Spearman rho per family.
             -> CSV table + bar chart + overall scatter colored by family.

  BRANCH 2 - by AFFINITY RANGE (bins of real_pKi):
             checks whether the error is uniform across the whole range or
             whether strong/weak binders are predicted worse.
             -> CSV table + bar chart (RMSE per bin).

How to run it:
  1) Keep this file inside the tanimoto/ folder (next to the calibration scripts).
  2) Adjust the paths in the CONFIG block if needed.
  3) Run it AFTER calibration, once loo_errors_detail.csv already exists:
         python error_analysis_families_range.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display window, only saves PNG files
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
BASE       = os.path.dirname(os.path.abspath(__file__))
ERR_FILE   = os.path.join(BASE, "results", "loo_errors_detail.csv")  # LOO output
INFO_FILE  = os.path.join(BASE, "OBP_info_new.csv")                   # OBP metadata
OUT_DIR    = os.path.join(BASE, "results")
MIN_N      = 15    # a family/bin with fewer than N predictions is flagged as unreliable
N_BINS_RANGE = 6   # number of affinity bins (quantiles of real_pKi)
TEAL, GREY, RED, BLUE = "#0E7C7B", "#B9C7C5", "#C0392B", "#2471A3"
# ===============================================================


def detect_columns(df):
    """Find the OBP / real_pKi / pred_pKi columns even if the header uses
    different names, by matching a list of exact aliases or substrings."""
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(exact, contains=None):
        # Try exact alias matches first, then fall back to substring matching
        for k in exact:
            if k in cols:
                return cols[k]
        if contains:
            for lc, orig in cols.items():
                if all(t in lc for t in contains):
                    return orig
        return None

    c_obp  = pick(["obp", "obp_name", "binding protein name", "protein"], ["obp"])
    c_real = pick(["real_pki", "pki_real", "real", "y_true", "true_pki"], ["real"])
    c_pred = pick(["pred_pki", "pki_pred", "pred", "y_pred", "prediction"], ["pred"])
    if not all([c_obp, c_real, c_pred]):
        raise SystemExit(
            "Could not detect the columns automatically.\n"
            f"Columns found: {list(df.columns)}\n"
            "Edit 'detect_columns' or rename the columns in the CSV."
        )
    return c_obp, c_real, c_pred


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def spearman(g):
    """Spearman correlation between real and predicted values; NaN if there
    isn't enough variation (fewer than 3 rows or 3 distinct values)."""
    if len(g) < 3 or g["real"].nunique() < 3 or g["pred"].nunique() < 3:
        return np.nan
    return g["real"].corr(g["pred"], method="spearman")


def load_errors():
    # Load the LOO error detail CSV and keep only the OBP/real/pred columns
    df = pd.read_csv(ERR_FILE)
    c_obp, c_real, c_pred = detect_columns(df)
    df = (df[[c_obp, c_real, c_pred]]
          .rename(columns={c_obp: "OBP", c_real: "real", c_pred: "pred"})
          .dropna(subset=["real", "pred"]))
    df["err"] = df["pred"] - df["real"]        # >0 overestimates, <0 underestimates

    # Attach each OBP's family/type from the metadata file
    info = pd.read_csv(INFO_FILE)
    fam_map = dict(zip(info["Binding Protein Name"], info["Binding Protein Type"]))
    df["family"] = df["OBP"].map(fam_map)
    # if the name has a suffix (e.g. mutant "AlinOBP5 K74A") and doesn't match,
    # retry using just the base name (first word)
    missing = df["family"].isna()
    if missing.any():
        base = df.loc[missing, "OBP"].astype(str).str.split().str[0]
        df.loc[missing, "family"] = base.map(fam_map).values
    df["family"] = df["family"].fillna("Unknown")
    return df


# ============================ BRANCH 1 ============================
def by_family(df):
    # Aggregate error stats (RMSE, bias, MAE, Spearman) per OBP family
    rows = []
    for fam, g in df.groupby("family"):
        rows.append(dict(family=fam, n=len(g), RMSE=rmse(g["err"]),
                         bias=g["err"].mean(), MAE=g["err"].abs().mean(),
                         Spearman=spearman(g)))
    tab = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    tab.to_csv(os.path.join(OUT_DIR, "loo_error_by_family.csv"), index=False)

    print("\n=== BRANCH 1 · ERROR BY FAMILY (sorted by RMSE) ===")
    print(tab.to_string(index=False, formatters={
        "RMSE": "{:.3f}".format, "bias": "{:+.3f}".format,
        "MAE": "{:.3f}".format, "Spearman": "{:.3f}".format}))

    # Chart: RMSE (left) + bias (right), grey bars if n < MIN_N (unreliable)
    reliable = tab["n"] >= MIN_N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    ax1.barh(tab["family"], tab["RMSE"],
             color=[TEAL if f else GREY for f in reliable])
    ax1.invert_yaxis(); ax1.set_xlabel("RMSE (pKi)  ·  lower = better")
    ax1.set_title("How wrong each family is")
    for i, (r, n) in enumerate(zip(tab["RMSE"], tab["n"])):
        ax1.text(r + 0.006, i, f"n={n}", va="center", fontsize=8, color="#555")
    ax2.barh(tab["family"], tab["bias"],
             color=[RED if b > 0 else BLUE for b in tab["bias"]])
    ax2.invert_yaxis(); ax2.axvline(0, color="k", lw=.8)
    ax2.set_xlabel("Mean bias (pred - real)")
    ax2.set_title("Overestimates (>0) / underestimates (<0)")
    fig.suptitle(f"LOO by OBP family   ·   grey = n<{MIN_N} (unreliable)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_error_by_family.png"), dpi=140)
    plt.close(fig)
    return tab


def scatter_by_family(df, rho_global, rmse_global):
    # Scatter of predicted vs. real pKi, colored by OBP family, to visually
    # spot whether any family is systematically off.
    fams = sorted(df["family"].unique())
    cmap = plt.get_cmap("tab10")
    cmap_map = {f: (GREY if f == "Unknown" else cmap(i % 10))
                for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(8.2, 8))
    lo = min(df["real"].min(), df["pred"].min())
    hi = max(df["real"].max(), df["pred"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", label="Perfect prediction", zorder=1)
    for fam in fams:
        g = df[df["family"] == fam]
        ax.scatter(g["real"], g["pred"], s=11, alpha=.5, edgecolors="none",
                   color=cmap_map[fam], label=f"{fam} (n={len(g)})")
    ax.set_xlabel("Real pKi (experimental)")
    ax.set_ylabel("Predicted pKi (Tanimoto read-across)")
    ax.set_title(f"Predicted vs Real (LOO), colored by family\n"
                 f"global Spearman rho = {rho_global:.3f}   RMSE = {rmse_global:.3f}")
    ax.legend(fontsize=7, markerscale=1.6, loc="upper left", framealpha=.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_scatter_by_family.png"), dpi=140)
    plt.close(fig)


# ============================ BRANCH 2 ============================
def by_range(df, rmse_global):
    # Split predictions into affinity bins (quantiles of real pKi) and
    # compute error stats per bin, to see if weak/strong binders are
    # predicted worse than the rest of the range.
    edges = np.unique(np.round(np.quantile(df["real"], np.linspace(0, 1, N_BINS_RANGE + 1)), 3))
    df = df.assign(bin=pd.cut(df["real"], bins=edges, include_lowest=True))
    rows = []
    for b, g in df.groupby("bin", observed=True):
        ki_lo = 10 ** (-b.right) * 1e6      # high pKi -> low Ki (uM)
        ki_hi = 10 ** (-b.left) * 1e6
        rows.append(dict(range_pKi=f"{b.left:.2f}-{b.right:.2f}",
                         Ki_uM_approx=f"{ki_lo:.2g}-{ki_hi:.2g}",
                         n=len(g), RMSE=rmse(g["err"]),
                         bias=g["err"].mean(), Spearman=spearman(g)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT_DIR, "loo_error_by_range.csv"), index=False)

    print("\n=== BRANCH 2 · ERROR BY AFFINITY RANGE (real_pKi) ===")
    print(tab.to_string(index=False, formatters={
        "RMSE": "{:.3f}".format, "bias": "{:+.3f}".format, "Spearman": "{:.3f}".format}))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(tab))
    ax.bar(x, tab["RMSE"], color=TEAL, width=.62)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(~{k} uM)" for r, k in zip(tab["range_pKi"], tab["Ki_uM_approx"])],
                       fontsize=8)
    ax.set_xlabel("Real pKi band   ·   left = WEAK binding  ·  right = STRONG binding")
    ax.set_ylabel("RMSE (pKi)")
    ax.set_title("Is the error the same across the whole affinity range?")
    for i, (r, n) in enumerate(zip(tab["RMSE"], tab["n"])):
        ax.text(i, r + 0.006, f"n={n}", ha="center", fontsize=8, color="#555")
    ax.axhline(rmse_global, color=RED, ls="--", lw=1, label=f"global RMSE = {rmse_global:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_error_by_range.png"), dpi=140)
    plt.close(fig)
    return tab


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_errors()
    rho_g, rmse_g = spearman(df), rmse(df["err"])  # global stats used as a baseline for both branches
    print(f"GLOBAL:  n={len(df)}   RMSE={rmse_g:.3f}   Spearman={rho_g:.3f}")

    by_family(df)
    scatter_by_family(df, rho_g, rmse_g)
    by_range(df, rmse_g)

    print(f"\nSaved to: {OUT_DIR}")
    for f in ["loo_error_by_family.csv", "loo_error_by_family.png",
              "loo_scatter_by_family.png",
              "loo_error_by_range.csv", "loo_error_by_range.png"]:
        print("   " + f)


if __name__ == "__main__":
    main()
