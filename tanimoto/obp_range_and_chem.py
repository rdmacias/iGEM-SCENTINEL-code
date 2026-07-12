"""
obp_range_and_chem.py
─────────────────────────────────────────────────────────────────────────────
Per a cada OBP, dos números:

  1) DIVERSITAT D'AFINITAT
     Hi ha un gradient de Ki dins d'aquest OBP (superforts + mitjans + fluixos),
     o tot està en un extrem?

     Es mesura amb la desviació estàndard del pKi (= -log10 Ki uM/1e6).
     pKi és log10, així que:
        std(pKi) ~ 0.3  → tot en un ordre de magnitud (rang estret)
        std(pKi) ~ 1.0  → factor 10x entre els extrems (rang normal)
        std(pKi) ~ 1.5+ → super-forts i super-fluixos alhora (rang ampli)

     A més es fa un check "hi ha de tot?": es divideix el rang de Ki en
     tres franges (fort / mitjà / fluix) i es mira si les tres tenen
     representants.

  2) DIVERSITAT QUÍMICA
     Semblança Tanimoto MITJANA entre parells de lligands d'aquest OBP.
     Alta = tots són químicament similars (una família).
     Baixa = són químicament diversos.

Combinant els dos surten 4 tipus d'OBP interessants (mira els llindars a
sota). Sortida: results/obp_range_and_chem.csv, una fila per OBP.
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

# Només compten els lligands amb Ki mesurada dins d'aquest rang (µM).
# Filtra els valors clarament fora d'escala biològica.
KI_MIN_UM = 0.001
KI_MAX_UM = 1000.0

# Cap opcional al nombre de lligands considerats per OBP.
TOP_N = None            # None = tots els mesurats vàlids
MIN_BINDERS = 3         # mínim per calcular std i tanimoto amb sentit

# Llindars de classificació (en log10 de Ki, o sigui pKi)
STD_ESTRET = 0.35       # std(pKi) <= això -> "rang estret"
STD_AMPLI = 0.75        # std(pKi) >= això -> "rang ampli"

# Franges d'afinitat per al check "hi ha de tot?"
# Ki < 5 µM = fort   ·   5-40 µM = mitjà   ·   >40 µM = fluix
KI_FORT = 5.0
KI_MITJA = 40.0

# Llindars de semblança química
CHEM_SIMILAR = 0.40     # tanimoto mitjà >= això -> família química
CHEM_DIVERS = 0.20      # tanimoto mitjà <= això -> químicament divers
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


def classify_range(std_pki, has_fort, has_mitja, has_fluix):
    """Diversitat d'afinitat: rang estret vs ampli, i si hi ha 'de tot'."""
    if std_pki <= STD_ESTRET:
        # Tot està en un extrem. On?
        if has_fort and not has_fluix:
            return "tot fort"
        if has_fluix and not has_fort:
            return "tot fluix"
        return "rang estret"
    if std_pki >= STD_AMPLI:
        cobertura = has_fort + has_mitja + has_fluix
        if cobertura == 3:
            return "rang ampli (fort + mitjà + fluix)"
        return "rang ampli"
    return "rang mitjà"


def classify_chem(mean_t):
    if np.isnan(mean_t):
        return "sense química"
    if mean_t >= CHEM_SIMILAR:
        return "família química"
    if mean_t <= CHEM_DIVERS:
        return "químicament divers"
    return "química mixta"


def combined_type(range_cls, chem_cls):
    """Els 4 quadrants + variants."""
    # rang ampli × similar química -> butxaca discriminadora fina
    if "ampli" in range_cls and chem_cls == "família química":
        return "discriminador fi (mateixa família, rang complet)"
    if "ampli" in range_cls and chem_cls == "químicament divers":
        return "fitxatge lliure (accepta tot, discrimina per Ki)"
    if range_cls == "tot fort" and chem_cls == "família química":
        return "super-selectiu (motiu concret)"
    if range_cls == "tot fort" and chem_cls == "químicament divers":
        return "SOSPITÓS (uneix fort coses diferents)"
    if range_cls == "tot fluix":
        return "mal sensor (cap unió forta)"
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
                "has_fort": False, "has_mitja": False, "has_fluix": False,
                "range_class": "dades insuficients",
                "mean_tanimoto": np.nan, "n_pairs_chem": 0,
                "chem_class": "dades insuficients",
                "combined_type": "dades insuficients"}

    # ─── DIVERSITAT D'AFINITAT ──────────────────────────────────────────
    pki = -np.log10(col.values * 1e-6)     # Ki (µM) -> pKi
    std_pki = float(np.std(pki, ddof=1)) if len(pki) > 1 else 0.0
    range_pki = float(pki.max() - pki.min())

    has_fort = bool((col < KI_FORT).any())
    has_mitja = bool(((col >= KI_FORT) & (col < KI_MITJA)).any())
    has_fluix = bool((col >= KI_MITJA).any())
    range_class = classify_range(std_pki, has_fort, has_mitja, has_fluix)

    # ─── DIVERSITAT QUÍMICA ─────────────────────────────────────────────
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
        "has_fort": has_fort, "has_mitja": has_mitja, "has_fluix": has_fluix,
        "range_class": range_class,
        "mean_tanimoto": round(mean_t, 3) if not np.isnan(mean_t) else np.nan,
        "n_pairs_chem": n_pairs,
        "chem_class": chem_class,
        "combined_type": combined_type(range_class, chem_class),
    }


def plot_scatter(result, out_png):
    """Núvol de punts std(pKi) vs tanimoto mitjà, un punt per OBP, colorejat
    pel tipus combinat, amb les línies de llindar que defineixen els 4
    quadrants d'interès."""
    data = result.dropna(subset=["std_pki", "mean_tanimoto"])
    if data.empty:
        print("\n  (no hi ha prou dades vàlides per fer el gràfic)")
        return

    color_map = {
        "discriminador fi (mateixa família, rang complet)": TEAL,
        "fitxatge lliure (accepta tot, discrimina per Ki)": BLUE,
        "super-selectiu (motiu concret)": "#8E44AD",
        "SOSPITÓS (uneix fort coses diferents)": RED,
    }
    resta = ~data["combined_type"].isin(color_map)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(data.loc[resta, "std_pki"], data.loc[resta, "mean_tanimoto"],
               s=22, alpha=.5, edgecolors="none", color=GREY,
               label=f"altres (n={int(resta.sum())})")
    for tipus, color in color_map.items():
        g = data[data["combined_type"] == tipus]
        if g.empty:
            continue
        ax.scatter(g["std_pki"], g["mean_tanimoto"], s=32, alpha=.85,
                   edgecolors="none", color=color, label=f"{tipus} (n={len(g)})")

    ax.axvline(STD_ESTRET, color="k", lw=.8, ls="--")
    ax.axvline(STD_AMPLI, color="k", lw=.8, ls="--")
    ax.axhline(CHEM_SIMILAR, color="k", lw=.8, ls=":")
    ax.axhline(CHEM_DIVERS, color="k", lw=.8, ls=":")

    ax.set_xlabel("std(pKi)  ·  diversitat d'AFINITAT  (estret → ampli)")
    ax.set_ylabel("Tanimoto mitjà  ·  diversitat QUÍMICA  (divers → família)")
    ax.set_title("Diversitat d'afinitat × diversitat química per OBP")
    ax.legend(fontsize=7, markerscale=1.4, loc="upper left", framealpha=.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    print("=" * 74)
    print("  DIVERSITAT D'AFINITAT × DIVERSITAT QUÍMICA PER OBP")
    print("=" * 74)

    if not BINDING_FILE.is_file():
        raise FileNotFoundError(f"No trobo {BINDING_FILE}")
    if not MATRIX_FILE.is_file() or not INDEX_FILE.is_file():
        raise FileNotFoundError(f"No trobo la matriu Tanimoto ({MATRIX_FILE})")

    print(f"\n  Llegint {BINDING_FILE.name}...")
    df, cas_col, obp_cols = load_binding(BINDING_FILE)
    print(f"  -> {len(df)} VOCs x {len(obp_cols)} OBPs")

    print(f"\n  Carregant matriu Tanimoto...")
    T_matrix, cas_to_pos = load_tanimoto(MATRIX_FILE, INDEX_FILE)
    idx_to_pos = build_idx_to_pos(df, cas_col, cas_to_pos)
    print(f"  -> {len(idx_to_pos)}/{len(df)} VOCs amb posició vàlida a la matriu\n")

    print(f"  Franges de Ki: fort < {KI_FORT} µM · mitjà {KI_FORT}-{KI_MITJA} µM · fluix > {KI_MITJA} µM")
    print(f"  std(pKi): estret <= {STD_ESTRET} · ampli >= {STD_AMPLI}")
    print(f"  Tanimoto: similar >= {CHEM_SIMILAR} · divers <= {CHEM_DIVERS}\n")

    rows = []
    for i, obp in enumerate(obp_cols, 1):
        rows.append(analyze_obp(obp, df, idx_to_pos, T_matrix))
        if i % 100 == 0 or i == len(obp_cols):
            print(f"    {i}/{len(obp_cols)} OBPs processats...")

    result = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values(["combined_type", "n_measured"],
                       ascending=[True, False]).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 74)
    print("  RESUM")
    print("=" * 74)

    print(f"\n  Repartiment per RANG D'AFINITAT:")
    for cls, n in result["range_class"].value_counts().items():
        print(f"     {cls:<40} {n:>5}")

    print(f"\n  Repartiment per DIVERSITAT QUÍMICA:")
    for cls, n in result["chem_class"].value_counts().items():
        print(f"     {cls:<40} {n:>5}")

    print(f"\n  Repartiment per TIPUS COMBINAT:")
    for cls, n in result["combined_type"].value_counts().items():
        print(f"     {cls:<50} {n:>5}")

    # Exemples de cada quadrant interessant
    for tipus in ["discriminador fi (mateixa família, rang complet)",
                  "fitxatge lliure (accepta tot, discrimina per Ki)",
                  "super-selectiu (motiu concret)",
                  "SOSPITÓS (uneix fort coses diferents)"]:
        subset = result[result["combined_type"] == tipus].head(5)
        if not subset.empty:
            print(f"\n  --- Exemples de: {tipus} ---")
            for _, r in subset.iterrows():
                print(f"     {r['OBP']:<20} std_pKi={r['std_pki']:.2f}  "
                      f"tanimoto={r['mean_tanimoto']:.2f}  "
                      f"n={r['n_used']}  ({r['Ki_min_uM']}-{r['Ki_max_uM']} µM)")

    plot_scatter(result, OUTPUT_PNG)

    print(f"\n  Guardat a: {OUTPUT_CSV}")
    print(f"  Gràfic desat a: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
