"""
confianca.py — Sistema de confiança per a l'OBP Finder (V3)
────────────────────────────────────────────────────────────
Separa dues preguntes que el programa abans confonia:

    "És bo el candidat?"      → Score        (el que ja calculava)
    "Quanta evidència tinc?"  → Confiança    (nou)

Cap candidat s'exclou mai per dades incompletes: el que baixa és la confiança.

Tres números per candidat, mai fusionats en un de sol:
    Conservador = Score × Confiança   → què puc afirmar avui
    Potencial   = Score               → fins on podria arribar si tot fos mesurat
    Guany       = Potencial − Conservador  → quanta ignorància hi ha en joc

La Taula 3 (feina pendent) s'ordena per Guany × Potencial: inverteix l'esforç
experimental allà on més incertesa redueixes sobre un candidat que val la pena.
"""

import os
import numpy as np
import pandas as pd


# ── CONSTANTS ────────────────────────────────────────────────────────────────

# Normalització de sigma. sigma_log = 1.0 vol dir que els donants discrepen en
# un factor 10× en Ki → confiança zero.
SIGMA_MAX = 1.0

# Penalització per pocs donants. Amb n_donors=1 la sigma és 0 per força
# matemàtica (no hi ha amb qui discrepar), no perquè hi hagi acord.
#   suport = 1 − 1/(1+n)   →  1 donant: 0.50 · 2: 0.67 · 5: 0.83 · 8: 0.89
# El salt d'1 a 2 donants és el que més val; de 7 a 8 gairebé no aporta res.
USE_SUPPORT_FACTOR = True

# Pes d'un interferent imputat respecte d'un de mesurat, al càlcul de c2.
PES_IMPUTAT = 0.5

# Confiança per defecte d'una cel·la imputada si no trobem el diag_table.
C1_FALLBACK = 0.5

# Llindars de color de la confiança global
CONF_ALTA = 0.80
CONF_MITJA = 0.50


# ── CÀRREGA DEL DIAGNÒSTIC ───────────────────────────────────────────────────

def load_diag_c1(diag_path, sigma_max=SIGMA_MAX, use_support=USE_SUPPORT_FACTOR):
    """
    Llegeix results/tanimoto_diag_table.csv i retorna:
        {(row_idx, OBP): c1}   per a cada cel·la IMPUTADA

    La presència d'una clau vol dir "aquesta cel·la és predita".
    La seva absència (havent-hi valor al CSV) vol dir "mesurada" → c1 = 1.0

    c1 = T_max × max(0, 1 − sigma/sigma_max) × suport(n_donors)

    Es multiplica (no se suma) perquè les tres condicions són NECESSÀRIES:
    un veí proper, donants d'acord, i prou donants. Si una falla, cau tot.
    """
    if not diag_path or not os.path.isfile(diag_path):
        return None

    d = pd.read_csv(diag_path)
    needed = {"row_idx", "OBP", "T_max", "sigma_log", "n_donors"}
    if not needed.issubset(d.columns):
        raise ValueError(
            f"{diag_path} no té les columnes esperades {sorted(needed)}.\n"
            f"Trobades: {list(d.columns)}"
        )

    dispersio = (1.0 - d["sigma_log"] / sigma_max).clip(lower=0.0)
    if use_support:
        suport = 1.0 - 1.0 / (1.0 + d["n_donors"])
    else:
        suport = 1.0

    c1 = (d["T_max"] * dispersio * suport).clip(0.0, 1.0)
    return {(int(r), str(o)): float(v)
            for r, o, v in zip(d["row_idx"], d["OBP"], c1)}


def check_files_match(df_selectivitat, df_afinitat, path_sel, path_af):
    """
    Els dos CSV surten de la MATEIXA crida a fill_gaps(): mateixes files,
    mateixes columnes. Si no coincideixen, alguna cosa greu ha passat
    (fitxers de execucions diferents, carpetes barrejades...).

    Millor petar aquí que no pas fer servir en silenci la Ki conservadora
    com a afinitat — que és exactament el bug que volem evitar.
    """
    cols_sel = set(df_selectivitat.columns[2:])
    cols_af = set(df_afinitat.columns[2:])
    if cols_sel != cols_af:
        nomes_sel = sorted(cols_sel - cols_af)[:5]
        nomes_af = sorted(cols_af - cols_sel)[:5]
        raise ValueError(
            "Els fitxers de selectivitat i afinitat no coincideixen.\n"
            "Han de venir de la mateixa execució de tanimoto_omplir.py.\n"
            f"  selectivitat: {path_sel}\n"
            f"  afinitat:     {path_af}\n"
            f"  Només a selectivitat: {nomes_sel}\n"
            f"  Només a afinitat:     {nomes_af}"
        )
    if len(df_selectivitat) != len(df_afinitat):
        raise ValueError(
            f"Nombre de files diferent: {len(df_selectivitat)} vs {len(df_afinitat)}"
        )


# ── CÀLCUL DE LES CONFIANCES ─────────────────────────────────────────────────

def _c1_cell(row_idx, obp, diag_c1, es_predit):
    """Confiança de l'afinitat d'una cel·la concreta."""
    if not es_predit:
        return 1.0                       # font experimental pura
    if diag_c1 is None:
        return C1_FALLBACK               # no tenim diag: valor pla
    key = (int(row_idx), str(obp))
    if key in diag_c1:
        return diag_c1[key]              # cel·la imputada
    return 1.0                           # hi era al CSV original → mesurada


def compute_confidences(result_table, weights, diana_row_idx,
                        interferent_row_idx, diag_c1, es_predit):
    """
    Afegeix a result_table:
        c1, c2, c4, c5, Confianca, Conservador, Potencial, Guany, Prioritat

    interferent_row_idx : {nom_interferent: row_idx}  (o None si no n'hi ha)
    diag_c1             : {(row_idx, OBP): c1}  del tanimoto_diag_table.csv
    es_predit           : True si la font de dades és la matriu imputada
    """
    df = result_table.copy()
    n_interf = int(df["N_interferents_total"].iloc[0]) if len(df) else 0

    c1_list, c2_list, c4_list, c5_list = [], [], [], []

    for _, row in df.iterrows():
        obp = row["OBP"]

        # ── c1: afinitat del VOC diana contra aquest OBP ──────────────────
        c1_list.append(_c1_cell(diana_row_idx, obp, diag_c1, es_predit))

        # ── c2: selectivitat. mesurat=1 · imputat=C1_i · falta=0 ────────
        if n_interf == 0:
            c2 = 0.0
        else:
            missing = set(row["Missing_interferents"])
            c2_total = 0.0
            for nom, r_idx in (interferent_row_idx or {}).items():
                if nom in missing:
                    continue                      # falta → suma 0
                if not es_predit:
                    c2_total += 1.0
                    continue
                key = (int(r_idx), str(obp))
                if diag_c1 is None:
                    c2_total += C1_FALLBACK
                elif key in diag_c1:
                    c2_total += diag_c1[key]
                else:
                    c2_total += 1.0
            c2 = c2_total / n_interf
        c2_list.append(float(c2))

        # ── c4: estabilitat. Depèn només de si coneixem el tipus d'OBP ────
        c4_list.append(1.0 if row["has_s4_data"] else 0.0)

        # ── c5: promiscuïtat. Suport segons quants competidors hi ha ──────
        # Un binari (has_s5_data) tracta igual 2 competidors que 100. Fem
        # servir el mateix "suport" que c1 (1 − 1/(1+n)): la confiança creix
        # amb n_competidors independentment de si s5 surt bo o dolent —
        # un s5=0.99 amb 2 competidors és molt menys fiable que amb 100.
        n_comp = int(row["N_competitors_s5"])
        c5_list.append(1.0 - 1.0 / (1.0 + n_comp))

    df["c1"] = c1_list
    df["c2"] = c2_list
    df["c4"] = c4_list
    df["c5"] = c5_list

    # ── Confiança global: mateixos pesos que el score ─────────────────────
    w = weights
    df["Confianca"] = (
        w["w_affinity"] * df["c1"] +
        w["w_selectivity"] * df["c2"] +
        w["w_stability"] * df["c4"] +
        w["w_promiscuity"] * df["c5"]
    ).clip(0.0, 1.0)

    # ── Els tres números. MAI fusionats en un de sol ──────────────────────
    # Conservador: descompta el score segons quant te'n pots fiar.
    # Es multiplica perquè és un valor esperat: resultat × probabilitat.
    df["Conservador"] = df["Score"] * df["Confianca"]
    df["Potencial"] = df["Score"]
    df["Guany"] = df["Potencial"] - df["Conservador"]

    # Prioritat de la feina pendent: val la pena el candidat (Potencial) I
    # aprendria alguna cosa mesurant-lo (Guany)?
    df["Prioritat"] = df["Guany"] * df["Potencial"]

    return df.sort_values("Conservador", ascending=False).reset_index(drop=True)


# ── SORTIDA ──────────────────────────────────────────────────────────────────

def _color(conf):
    if conf >= CONF_ALTA:
        return "verd"
    if conf >= CONF_MITJA:
        return "groc"
    return "vermell"


def show_unified_table(df, how_many, es_predit=False, min_conf=None):
    """
    Substitueix les Taules 1 i 2. Una sola llista: ningú s'amaga.

    Les antigues taules partien els candidats per un criteri binari (tinc TOTS
    els interferents, sí/no). Amb una confiança contínua, aquesta partició és
    massa grollera — i deixava fora el millor binder mesurat només perquè li
    faltava un interferent.
    """
    if df.empty:
        print("\n  Cap candidat amb dades.")
        return df

    vista = df if min_conf is None else df[df["Confianca"] >= min_conf]
    if vista.empty:
        print(f"\n  Cap candidat supera la confiança {min_conf:.2f}.")
        return vista

    sep = "─" * 108
    print(f"\n{sep}")
    print(f"  CANDIDATS — llista única, ordenada per Score conservador — top {how_many}")
    print(f"  Conservador = Score × Confiança   ·   Potencial = Score si tot fos mesurat")
    print(sep)

    if es_predit:
        hdr = (f"  {'#':>3}  {'OBP':<20} {'Conserv.':>9} {'Potenc.':>8} {'Guany':>7} "
               f"{'Conf.':>6} {'c1':>5} {'c2':>5}  {'s1':>5} {'s2':>5} {'s4':>5} {'s5':>5} {'Ki(μM)':>8} "
               f"{'Interf':>7}  {'Tipus':<12}")
    else:
        hdr = (f"  {'#':>3}  {'OBP':<20} {'Conserv.':>9} {'Potenc.':>8} {'Guany':>7} "
               f"{'Conf.':>6} {'c2':>5}  {'s1':>5} {'s2':>5} {'s4':>5} {'s5':>5} {'Ki(μM)':>8} "
               f"{'Interf':>7}  {'Tipus':<12}")
    print(hdr)
    print(sep)

    for pos, (_, r) in enumerate(vista.head(how_many).iterrows()):
        n_tot = int(r["N_interferents_total"])
        n_miss = int(r["N_interferents_missing"])
        interf = f"{n_tot - n_miss}/{n_tot}" if n_tot else "  —"
        marca = {"verd": "*", "groc": "~", "vermell": "!"}[_color(r["Confianca"]) ]
        if es_predit:
            print(f"  {pos+1:>3}. {r['OBP']:<20} {r['Conservador']:>9.4f} "
                  f"{r['Potencial']:>8.4f} {r['Guany']:>7.4f} "
                  f"{r['Confianca']:>5.2f}{marca} {r['c1']:>5.2f} {r['c2']:>5.2f}  "
                  f"{r['s1_affinity']:>5.3f} {r['s2_selectivity']:>5.3f} {r['s4_stability']:>5.3f} {r['s5_promiscuity']:>5.3f} "
                  f"{r['Ki_diana_uM']:>8.2f} {interf:>7}  {str(r['Type']):<12}")
        else:
            print(f"  {pos+1:>3}. {r['OBP']:<20} {r['Conservador']:>9.4f} "
                  f"{r['Potencial']:>8.4f} {r['Guany']:>7.4f} "
                  f"{r['Confianca']:>5.2f}{marca} {r['c2']:>5.2f}  "
                  f"{r['s1_affinity']:>5.3f} {r['s2_selectivity']:>5.3f} {r['s4_stability']:>5.3f} {r['s5_promiscuity']:>5.3f} "
                  f"{r['Ki_diana_uM']:>8.2f} {interf:>7}  {str(r['Type']):<12}")

    print(sep)
    print(f"  * conf ≥ {CONF_ALTA:.2f} (fiable)   "
          f"~ {CONF_MITJA:.2f}–{CONF_ALTA:.2f} (parcial)   "
          f"! < {CONF_MITJA:.2f} (poca evidència)")
    if es_predit:
        print(f"  c1 < 1.00 → l'afinitat és PREDITA (Tanimoto), no mesurada.")
    print(f"  Total de candidats: {len(df)}")
    return vista


def show_worklist(df, how_many=25):
    """
    Taula 3 — feina pendent, ordenada per Guany × Potencial.

    Ordenar per score seria subòptim: el candidat que ja té 4/5 interferents té
    poc a guanyar amb l'última mesura. En canvi, un candidat a les fosques (1/5)
    amb bona afinitat pot saltar molt amunt — o enfonsar-se. Cada mesura hi
    informa moltíssim.

    El Potencial evita perseguir candidats amb sostre baix: encara que els
    mesuressis tot, mai serien bons.
    """
    pendents = df[df["N_interferents_missing"] > 0].copy()
    if pendents.empty:
        sep = "─" * 78
        print(f"\n{sep}")
        print(f"  TAULA 3 — Feina pendent")
        print(f"  Cap parella OBP/interferent per calcular.")
        print(sep)
        return

    pendents = pendents.sort_values("Prioritat", ascending=False)

    sep = "─" * 90
    print(f"\n{sep}")
    print(f"  TAULA 3 — Feina pendent, ordenada per PRIORITAT = Guany × Potencial")
    print(f"  Inverteix l'esforç allà on més ignorància redueixes (amb bon sostre)")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<20} {'Prior.':>7} {'Guany':>7} {'Potenc.':>8}  "
          f"{'Interferent que falta':<28}")
    print(sep)

    i = 0
    for _, r in pendents.iterrows():
        for interf in sorted(r["Missing_interferents"]):
            i += 1
            if i > how_many:
                break
            print(f"  {i:>3}. {r['OBP']:<20} {r['Prioritat']:>7.4f} "
                  f"{r['Guany']:>7.4f} {r['Potencial']:>8.4f}  {interf:<28}")
        if i > how_many:
            break

    total = int(pendents["N_interferents_missing"].sum())
    print(sep)
    print(f"  Total de parelles OBP/interferent a calcular: {total}")
