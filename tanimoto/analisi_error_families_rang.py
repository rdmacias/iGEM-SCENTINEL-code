#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisi_error_families_rang.py
--------------------------------
Analitza els errors del Leave-One-Out (loo_errors_detall.csv) en dues branques:

  BRANCA 1 — per FAMILIA d'OBP (Classic OBP, PBP, CSP, GOBP1/2, Minus-C, ...):
             RMSE, biaix (pred-real), MAE, n i Spearman de cada familia.
             -> taula CSV + grafic de barres + scatter general colorejat per familia.

  BRANCA 2 — per RANG D'AFINITAT (bins de real_pKi):
             comprova si l'error es igual a tot el rang o si els binders forts/febles
             es prediuen pitjor.  -> taula CSV + grafic de barres (RMSE per tram).

Com fer-lo servir:
  1) Deixa aquest fitxer dins de la carpeta tanimoto/ (al costat del calibratge).
  2) Ajusta, si cal, les rutes del bloc CONFIG.
  3) Executa'l DESPRES del calibratge (quan ja existeix loo_errors_detall.csv):
         python analisi_error_families_rang.py

--------------------------------------------------------------------
Analyzes Leave-One-Out errors (loo_errors_detall.csv) along two branches:

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
  3) Run it AFTER calibration, once loo_errors_detall.csv already exists:
         python analisi_error_families_rang.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display window, only saves PNG files
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
BASE       = os.path.dirname(os.path.abspath(__file__))
ERR_FILE   = os.path.join(BASE, "results", "loo_errors_detall.csv")  # LOO output
INFO_FILE  = os.path.join(BASE, "OBP_info_new.csv")                   # OBP metadata
OUT_DIR    = os.path.join(BASE, "results")
MIN_N      = 15    # a family/bin with fewer than N predictions is flagged as unreliable
N_BINS_RANG = 6    # number of affinity bins (quantiles of real_pKi)
TEAL, GREY, RED, BLUE = "#0E7C7B", "#B9C7C5", "#C0392B", "#2471A3"
# ===============================================================


def detecta_columnes(df):
    """Find the OBP / real_pKi / pred_pKi columns even if the header uses
    different names, by matching a list of exact aliases or substrings."""
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(exactes, contes=None):
        # Try exact alias matches first, then fall back to substring matching
        for k in exactes:
            if k in cols:
                return cols[k]
        if contes:
            for lc, orig in cols.items():
                if all(t in lc for t in contes):
                    return orig
        return None

    c_obp  = pick(["obp", "obp_name", "binding protein name", "protein"], ["obp"])
    c_real = pick(["real_pki", "pki_real", "real", "y_true", "true_pki"], ["real"])
    c_pred = pick(["pred_pki", "pki_pred", "pred", "y_pred", "prediccio"], ["pred"])
    if not all([c_obp, c_real, c_pred]):
        raise SystemExit(
            "No he pogut detectar les columnes automaticament.\n"
            f"Columnes trobades: {list(df.columns)}\n"
            "Edita 'detecta_columnes' o reanomena les columnes del CSV."
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


def carrega():
    # Load the LOO error detail CSV and keep only the OBP/real/pred columns
    df = pd.read_csv(ERR_FILE)
    c_obp, c_real, c_pred = detecta_columnes(df)
    df = (df[[c_obp, c_real, c_pred]]
          .rename(columns={c_obp: "OBP", c_real: "real", c_pred: "pred"})
          .dropna(subset=["real", "pred"]))
    df["err"] = df["pred"] - df["real"]        # >0 overestimates, <0 underestimates

    # Attach each OBP's family/type from the metadata file
    info = pd.read_csv(INFO_FILE)
    fam_map = dict(zip(info["Binding Protein Name"], info["Binding Protein Type"]))
    df["familia"] = df["OBP"].map(fam_map)
    # if the name has a suffix (e.g. mutant "AlinOBP5 K74A") and doesn't match,
    # retry using just the base name (first word)
    manca = df["familia"].isna()
    if manca.any():
        base = df.loc[manca, "OBP"].astype(str).str.split().str[0]
        df.loc[manca, "familia"] = base.map(fam_map).values
    df["familia"] = df["familia"].fillna("Desconeguda")
    return df


# ============================ BRANCH 1 ============================
def per_familia(df):
    # Aggregate error stats (RMSE, bias, MAE, Spearman) per OBP family
    files = []
    for fam, g in df.groupby("familia"):
        files.append(dict(familia=fam, n=len(g), RMSE=rmse(g["err"]),
                          biaix=g["err"].mean(), MAE=g["err"].abs().mean(),
                          Spearman=spearman(g)))
    tab = pd.DataFrame(files).sort_values("RMSE").reset_index(drop=True)
    tab.to_csv(os.path.join(OUT_DIR, "loo_error_per_familia.csv"), index=False)

    print("\n=== BRANCA 1 · ERROR PER FAMILIA (ordenat per RMSE) ===")
    print(tab.to_string(index=False, formatters={
        "RMSE": "{:.3f}".format, "biaix": "{:+.3f}".format,
        "MAE": "{:.3f}".format, "Spearman": "{:.3f}".format}))

    # Chart: RMSE (left) + bias (right), grey bars if n < MIN_N (unreliable)
    fiable = tab["n"] >= MIN_N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    ax1.barh(tab["familia"], tab["RMSE"],
             color=[TEAL if f else GREY for f in fiable])
    ax1.invert_yaxis(); ax1.set_xlabel("RMSE (pKi)  ·  mes baix = millor")
    ax1.set_title("Quant s'equivoca cada familia")
    for i, (r, n) in enumerate(zip(tab["RMSE"], tab["n"])):
        ax1.text(r + 0.006, i, f"n={n}", va="center", fontsize=8, color="#555")
    ax2.barh(tab["familia"], tab["biaix"],
             color=[RED if b > 0 else BLUE for b in tab["biaix"]])
    ax2.invert_yaxis(); ax2.axvline(0, color="k", lw=.8)
    ax2.set_xlabel("Biaix mitja (pred - real)")
    ax2.set_title("Sobreestima (>0) / subestima (<0)")
    fig.suptitle(f"LOO per familia d'OBP   ·   gris = n<{MIN_N} (poc fiable)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_error_per_familia.png"), dpi=140)
    plt.close(fig)
    return tab


def scatter_per_familia(df, rho_global, rmse_global):
    # Scatter of predicted vs. real pKi, colored by OBP family, to visually
    # spot whether any family is systematically off.
    fams = sorted(df["familia"].unique())
    cmap = plt.get_cmap("tab10")
    cmap_map = {f: (GREY if f == "Desconeguda" else cmap(i % 10))
                for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(8.2, 8))
    lo = min(df["real"].min(), df["pred"].min())
    hi = max(df["real"].max(), df["pred"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="grey", label="Prediccio perfecta", zorder=1)
    for fam in fams:
        g = df[df["familia"] == fam]
        ax.scatter(g["real"], g["pred"], s=11, alpha=.5, edgecolors="none",
                   color=cmap_map[fam], label=f"{fam} (n={len(g)})")
    ax.set_xlabel("pKi real (experimental)")
    ax.set_ylabel("pKi predit (Tanimoto read-across)")
    ax.set_title(f"Prediccio vs Real (LOO), colorejat per familia\n"
                 f"Spearman global rho = {rho_global:.3f}   RMSE = {rmse_global:.3f}")
    ax.legend(fontsize=7, markerscale=1.6, loc="upper left", framealpha=.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_scatter_per_familia.png"), dpi=140)
    plt.close(fig)


# ============================ BRANCH 2 ============================
def per_rang(df, rmse_global):
    # Split predictions into affinity bins (quantiles of real pKi) and
    # compute error stats per bin, to see if weak/strong binders are
    # predicted worse than the rest of the range.
    edges = np.unique(np.round(np.quantile(df["real"], np.linspace(0, 1, N_BINS_RANG + 1)), 3))
    df = df.assign(bin=pd.cut(df["real"], bins=edges, include_lowest=True))
    files = []
    for b, g in df.groupby("bin", observed=True):
        ki_lo = 10 ** (-b.right) * 1e6      # high pKi -> low Ki (uM)
        ki_hi = 10 ** (-b.left) * 1e6
        files.append(dict(rang_pKi=f"{b.left:.2f}–{b.right:.2f}",
                          Ki_uM_aprox=f"{ki_lo:.2g}–{ki_hi:.2g}",
                          n=len(g), RMSE=rmse(g["err"]),
                          biaix=g["err"].mean(), Spearman=spearman(g)))
    tab = pd.DataFrame(files)
    tab.to_csv(os.path.join(OUT_DIR, "loo_error_per_rang.csv"), index=False)

    print("\n=== BRANCA 2 · ERROR PER RANG D'AFINITAT (real_pKi) ===")
    print(tab.to_string(index=False, formatters={
        "RMSE": "{:.3f}".format, "biaix": "{:+.3f}".format, "Spearman": "{:.3f}".format}))

    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(tab))
    ax.bar(x, tab["RMSE"], color=TEAL, width=.62)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(~{k} µM)" for r, k in zip(tab["rang_pKi"], tab["Ki_uM_aprox"])],
                       fontsize=8)
    ax.set_xlabel("Tram de pKi real   ·   esquerra = unio FEBLE  ·  dreta = unio FORTA")
    ax.set_ylabel("RMSE (pKi)")
    ax.set_title("L'error es igual a tot el rang d'afinitat?")
    for i, (r, n) in enumerate(zip(tab["RMSE"], tab["n"])):
        ax.text(i, r + 0.006, f"n={n}", ha="center", fontsize=8, color="#555")
    ax.axhline(rmse_global, color=RED, ls="--", lw=1, label=f"RMSE global = {rmse_global:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loo_error_per_rang.png"), dpi=140)
    plt.close(fig)
    return tab


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = carrega()
    rho_g, rmse_g = spearman(df), rmse(df["err"])  # global stats used as a baseline for both branches
    print(f"GLOBAL:  n={len(df)}   RMSE={rmse_g:.3f}   Spearman={rho_g:.3f}")

    per_familia(df)
    scatter_per_familia(df, rho_g, rmse_g)
    per_rang(df, rmse_g)

    print(f"\nDesat a: {OUT_DIR}")
    for f in ["loo_error_per_familia.csv", "loo_error_per_familia.png",
              "loo_scatter_per_familia.png",
              "loo_error_per_rang.csv", "loo_error_per_rang.png"]:
        print("   " + f)


if __name__ == "__main__":
    main()
