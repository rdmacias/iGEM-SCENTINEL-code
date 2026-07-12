"""
generate_report.py  —  OBP Finder | Informe PDF complet (iGEM URV 2025)

Executa els 3 mètodes (determinista, Perturbació, SMAA pur) i genera
un informe PDF descarregable a cada execució.
"""

import os
import sys
import re
import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from main import (
    load_csv_files,
    validar_voc,
    read_interferent_file,
    validar_llista_interferents,
    ask_user_for_weights,
    ask_user_for_tau,
    build_obp_ranking,
    resolve_data_file,
    BINDING_FILE_EXPERIMENTAL,
    BINDING_FILE_PREDICTED,
    BINDING_FILE_PREDICTED_AFFINITY,
    INFO_FILE,
)
import confianca as cf
from SMAAA import smaa_complet

# ── Paleta de colors ──────────────────────────────────────────────────────────
C_S1        = "#4C72B0"
C_S2        = "#DD8452"
C_S4        = "#55A868"
C_S5        = "#C44E52"
C_SCORE     = "#8172B2"
C_HIGHLIGHT = "#F4C430"
C_BG        = "#F5F7FA"
C_HEADER    = "#1a1a2e"


def _fig(landscape=True):
    size = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor(C_BG)
    return fig


def _save(pdf, fig):
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ── Pàgina 1: Portada ────────────────────────────────────────────────────────

def page_cover(pdf, voc_name, voc_cas, weights, tau, how_many,
               interferent_list, n_iter, c_val, ts):
    fig = _fig()
    kw = dict(ha='center', va='center', transform=fig.transFigure)

    fig.text(0.5, 0.85, "OBP FINDER — Informe d'Anàlisi",
             fontsize=22, fontweight='bold', color=C_HEADER, **kw)
    fig.text(0.5, 0.78, "iGEM URV 2025  ·  Informe automàtic complet (3 mètodes)",
             fontsize=12, color='#4C72B0', **kw)
    fig.text(0.5, 0.72, f"Generat: {ts}",
             fontsize=10, color='#888888', **kw)

    # línia decorativa
    ax_line = fig.add_axes([0.1, 0.68, 0.8, 0.003])
    ax_line.set_facecolor('#4C72B0')
    ax_line.axis('off')

    w = weights
    interf_str = ', '.join(interferent_list) if interferent_list else '— cap —'
    info = (
        f"  VOC diana        :  {voc_name}\n"
        f"  CAS              :  {voc_cas}\n"
        f"  Interferents     :  {interf_str}\n"
        f"  Top N candidats  :  {how_many}\n"
        f"\n"
        f"  ─── Pesos ───────────────────────────────\n"
        f"  τ (selectivitat) :  {tau:.0f}\n"
        f"  w₁ afinitat      :  {w['w_affinity']:.3f}\n"
        f"  w₂ selectivitat  :  {w['w_selectivity']:.3f}\n"
        f"  w₄ estabilitat   :  {w['w_stability']:.3f}\n"
        f"  w₅ promiscuïtat  :  {w['w_promiscuity']:.3f}\n"
        f"\n"
        f"  ─── Configuració SMAA ───────────────────\n"
        f"  Mètode 2 (Perturbació) :  c = {c_val:.0f}%,  {n_iter:,} iteracions\n"
        f"  Mètode 3 (SMAA pur)  :  {n_iter:,} iteracions\n"
    )
    fig.text(0.12, 0.64, info, ha='left', va='top',
             fontsize=10.5, fontfamily='monospace', color='#222222',
             linespacing=1.65, transform=fig.transFigure)

    _save(pdf, fig)


# ── Pàgina 2: M1 — gràfic scores + subscores ─────────────────────────────────

def page_mode1_chart(pdf, result_table, voc_name, weights, how_many):
    top = result_table.head(how_many).copy()
    top['s4_stability']   = top['s4_stability'].fillna(0)
    top['s2_selectivity'] = top['s2_selectivity'].fillna(0)
    n = len(top)
    names = [r[:22] for r in top['OBP']]
    w = weights

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"Mètode 1 — Pesos Deterministes  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)

    # ── Esquerra: Score total ─────────────────────────────────────────────────
    ax1.set_facecolor('white')
    colors = [C_HIGHLIGHT if i == 0 else C_SCORE for i in range(n)]
    ax1.barh(range(n), top['Score'], color=colors, edgecolor='white', height=0.65)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Score final', fontsize=9)
    ax1.set_title('Ranking — Score total', fontsize=10)
    ax1.set_xlim(0, 1.12)
    ax1.axvline(0, color='#cccccc', linewidth=0.5)
    for i, val in enumerate(top['Score']):
        ax1.text(val + 0.012, i, f"{val:.4f}", va='center', fontsize=7.5)
    star_idx = [i for i, p in enumerate(top['Preferred']) if p]
    for i in star_idx:
        ax1.text(-0.04, i, '★', va='center', fontsize=9, color='#DAA520')

    # ── Dreta: subscores apilats (contribucions pesades) ─────────────────────
    ax2.set_facecolor('white')
    s1c = top['s1_affinity']    * w['w_affinity']
    s2c = top['s2_selectivity'] * w['w_selectivity']
    s4c = top['s4_stability']   * w['w_stability']
    s5c = top['s5_promiscuity'] * w['w_promiscuity']
    y = range(n)
    ax2.barh(y, s1c, color=C_S1, label=f's1 afinitat  (w={w["w_affinity"]:.2f})', height=0.65)
    ax2.barh(y, s2c, left=s1c, color=C_S2,
             label=f's2 selectivitat  (w={w["w_selectivity"]:.2f})', height=0.65)
    ax2.barh(y, s4c, left=s1c + s2c, color=C_S4,
             label=f's4 estabilitat  (w={w["w_stability"]:.2f})', height=0.65)
    ax2.barh(y, s5c, left=s1c + s2c + s4c, color=C_S5,
             label=f's5 promiscuïtat  (w={w["w_promiscuity"]:.2f})', height=0.65)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(names, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Contribució pesada (w × subscore)', fontsize=9)
    ax2.set_title('Descomposició subscores (apilat)', fontsize=10)
    ax2.legend(fontsize=7.5, loc='lower right')
    ax2.set_xlim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(pdf, fig)


# ── Pàgina 3: M1 — Taula detallada ───────────────────────────────────────────

def page_mode1_table(pdf, result_table, voc_name, how_many):
    top = result_table.head(how_many).copy()

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"Mètode 1 — Taula Detallada  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)
    ax.axis('off')

    cols = ['#', 'OBP', 'Score', 's1', 's2', 's4', 's5',
            'Ki (μM)', 'Cys', 'Pref', 'Ki interf', 'Tipus']
    rows = []
    for i, (_, row) in enumerate(top.iterrows()):
        cys = str(int(row['Cystines'])) if not pd.isna(row['Cystines']) else '?'
        ki_i = (f"{row['Min_Ki_interferent_uM']:.2f}"
                if not pd.isna(row['Min_Ki_interferent_uM']) else '—')
        s4_disp = (f"{row['s4_stability']:.3f}"
                   if not pd.isna(row['s4_stability']) else '?')
        rows.append([
            str(i + 1),
            row['OBP'][:20],
            f"{row['Score']:.4f}",
            f"{row['s1_affinity']:.3f}",
            f"{row['s2_selectivity']:.3f}",
            s4_disp,
            f"{row['s5_promiscuity']:.3f}",
            f"{row['Ki_diana_uM']:.2f}",
            cys,
            '★' if row['Preferred'] else '',
            ki_i,
            str(row['Type'])[:12],
        ])

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#4C72B0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(len(rows)):
        bg = '#EBF2FF' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)
        if rows[i][9] == '★':
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor('#FFFACD')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(pdf, fig)


# ── Pàgina 4: M1 — Taula 1 (selectivitat fiable) / Taula 2 (prometedors) ────

def page_table1_table2(pdf, result_table, voc_name, how_many):
    """Taula 1: OBPs amb TOTS els interferents mesurats (s2 fiable).
    Taula 2: OBPs als quals falta ≥1 interferent (s2 descartada, s1+s4+s5).
    Retorna la Taula 2 ordenada, per construir després la Taula 3 (llista Boltz)."""
    df_t1 = result_table[result_table['has_all_interferents']].reset_index(drop=True)
    df_t2 = result_table[~result_table['has_all_interferents']].copy()

    if not df_t2.empty:
        df_t2 = df_t2.sort_values('Score_sense_s2', ascending=False).reset_index(drop=True)

    if df_t1.empty and df_t2.empty:
        return df_t2

    n_panels = sum([not df_t1.empty, not df_t2.empty])
    fig, axes = plt.subplots(1, n_panels, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"Taula 1 / Taula 2 — Selectivitat fiable vs. prometedors  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)
    if n_panels == 1:
        axes = [axes]

    if df_t1.empty:
        n_interferents_total = (result_table['N_interferents_total'].iloc[0]
                                 if 'N_interferents_total' in result_table.columns
                                 and not result_table.empty else 0)
        if n_interferents_total == 0:
            msg = "✗ No hi ha dades d'interferents — la selectivitat no es pot avaluar.\nLa Taula 1 no es mostra. Saltem directament a la Taula 2."
        else:
            msg = ("✗ Cap OBP té TOTS els interferents mesurats — selectivitat no fiable per a cap "
                   "candidat.\nLa Taula 1 no es mostra. Saltem directament a la Taula 2.")
        fig.text(0.5, 0.90, msg, ha='center', va='top', fontsize=9, color='#B33A3A', wrap=True)

    ax_idx = 0
    if not df_t1.empty:
        ax = axes[ax_idx]; ax_idx += 1
        top_e = df_t1.head(how_many)
        n = len(top_e)
        ax.set_facecolor('white')
        ax.barh(range(n), top_e['Score'], color='#4C72B0', edgecolor='white', height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[:22] for r in top_e['OBP']], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Score (4 criteris, s2 fiable)', fontsize=9)
        ax.set_title(f'TAULA 1 — tots els interferents mesurats (top {n})', fontsize=10)
        for i, val in enumerate(top_e['Score']):
            ax.text(val + 0.008, i, f"{val:.4f}", va='center', fontsize=7)

    if not df_t2.empty:
        ax = axes[ax_idx]
        top_x = df_t2.head(how_many)
        n = len(top_x)
        ax.set_facecolor('white')
        ax.barh(range(n), top_x['Score_sense_s2'], color='#DD8452', edgecolor='white', height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[:22] for r in top_x['OBP']], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Score (s1+s4+s5, sense s2)', fontsize=9)
        ax.set_title(f'TAULA 2 — falta ≥1 interferent (top {n})', fontsize=10)
        for i, val in enumerate(top_x['Score_sense_s2']):
            ax.text(val + 0.008, i, f"{val:.4f}", va='center', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(pdf, fig)

    return df_t2


# ── Pàgina 5: M1 — Taula 3 (llista de feina Boltz) ───────────────────────────

def page_table3_boltz(pdf, df_table2_sorted, voc_name, max_rows=35):
    """Llista de parelles OBP/interferent que falten i s'han de calcular,
    ordenada segons la Taula 2 (primer les dels OBPs més prometedors)."""
    if df_table2_sorted is None or df_table2_sorted.empty:
        return

    pairs = []
    for _, row in df_table2_sorted.iterrows():
        for interf in sorted(row['Missing_interferents']):
            pairs.append((row['OBP'], row['Score_sense_s2'], interf))

    if not pairs:
        fig = _fig()
        fig.suptitle(f"Taula 3 — Llista de feina Boltz  |  VOC: {voc_name}",
                     fontsize=13, fontweight='bold', color=C_HEADER, y=0.92)
        fig.text(0.5, 0.5,
                  "✗ No s'han definit interferents — no hi ha cap parella OBP/interferent a calcular.",
                  ha='center', va='center', fontsize=11, color='#B33A3A')
        _save(pdf, fig)
        return

    truncated = len(pairs) > max_rows
    shown = pairs[:max_rows]

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"Taula 3 — Llista de feina Boltz  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)
    ax.axis('off')

    cols = ['#', 'OBP', 'Score Taula 2', 'Interferent que falta']
    rows = [
        [str(i + 1), obp[:22], f"{score:.4f}", interf[:30]]
        for i, (obp, score, interf) in enumerate(shown)
    ]

    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#4C72B0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(len(rows)):
        bg = '#FFF3E0' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[i + 1, j].set_facecolor(bg)

    footer = f"Total de parelles OBP/interferent a calcular: {len(pairs)}"
    if truncated:
        footer += f"  (es mostren les {max_rows} primeres, prioritzades per la Taula 2)"
    fig.text(0.5, 0.04, footer, ha='center', fontsize=9, color='#555555')

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    _save(pdf, fig)


# ── Pàgina opcional: Comparació experimental vs computacional ───────────────

def page_top10_comparison(pdf, experimental_table, predicted_table, voc_name, how_many):
    exp_top = experimental_table.head(how_many)
    pred_top = predicted_table.head(how_many)

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(
        f"Top {how_many} candidats — Experimental vs. Computacional  |  VOC: {voc_name}",
        fontsize=13, fontweight='bold', color=C_HEADER, y=0.98
    )

    for ax, title, df, color in [
        (axes[0], "Dades experimentals", exp_top, C_SCORE),
        (axes[1], "Dades computacionals", pred_top, '#DD8452'),
    ]:
        ax.set_facecolor('white')
        values = df['Score'].tolist()
        names = df['OBP'].str[:20].tolist()
        ax.barh(range(len(values)), values, color=color, edgecolor='white', height=0.65)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Score final', fontsize=9)
        xlim = max(max(values) * 1.18, 0.08) if values else 1.0
        ax.set_xlim(0, xlim)
        for i, val in enumerate(values):
            ax.text(val + xlim * 0.01, i, f"{val:.4f}", va='center', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(pdf, fig)


def page_missing_predicted(pdf, voc_name, message):
    fig = _fig()
    fig.suptitle(f"Rànquing computacional no disponible  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.92)
    fig.text(0.5, 0.58, message,
             ha='center', va='center', fontsize=11, color='#B33A3A', wrap=True)
    _save(pdf, fig)


# ── Pàgines 5/6: SMAA ────────────────────────────────────────────────────────

def page_smaa(pdf, result_table, rank_accept, central_w, voc_name, how_many, mode_label):
    m = len(result_table)
    n_rank_cols = min(m, 5)
    order = np.argsort(-rank_accept[:, 0])
    top_idx = order[:how_many]
    names   = [result_table.iloc[i]['OBP'][:22] for i in top_idx]
    p_best  = rank_accept[top_idx, 0]

    has_cw = central_w is not None and any(central_w.get(i) is not None for i in top_idx)

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"{mode_label}  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)

    if has_cw:
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38)
        ax_bar  = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_cw   = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.4, wspace=0.38)
        ax_bar  = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_cw   = None

    # ── Bar chart p(rang 1) ──
    ax_bar.set_facecolor('white')
    bar_colors = [C_HIGHLIGHT if i == 0 else C_SCORE for i in range(len(top_idx))]
    ax_bar.barh(range(len(top_idx)), p_best, color=bar_colors,
                edgecolor='white', height=0.65)
    ax_bar.set_yticks(range(len(top_idx)))
    ax_bar.set_yticklabels(names, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('p(rang 1)', fontsize=9)
    ax_bar.set_title('Probabilitat de ser el millor', fontsize=10)
    xlim = max(p_best.max() * 1.18, 0.08)
    ax_bar.set_xlim(0, xlim)
    for i, val in enumerate(p_best):
        ax_bar.text(val + xlim * 0.01, i, f"{val:.3f}", va='center', fontsize=7.5)

    # ── Heatmap d'acceptació ──
    heat_data = rank_accept[top_idx, :n_rank_cols]
    im = ax_heat.imshow(heat_data, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax_heat.set_xticks(range(n_rank_cols))
    ax_heat.set_xticklabels([f"Rang {i+1}" for i in range(n_rank_cols)], fontsize=8)
    ax_heat.set_yticks(range(len(top_idx)))
    ax_heat.set_yticklabels(names, fontsize=8.5)
    ax_heat.set_title("Matriu d'acceptació de rangs", fontsize=10)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    for i in range(len(top_idx)):
        for j in range(n_rank_cols):
            val = heat_data[i, j]
            txt_color = 'white' if val > 0.55 else 'black'
            ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=7, color=txt_color)

    # ── Central weights (mode 3) ──
    if has_cw and ax_cw is not None:
        ax_cw.set_facecolor('white')
        cw_labels = ['s1 Afinitat', 's2 Selectivitat', 's4 Estabilitat', 's5 Promiscuïtat']
        cw_colors = [C_S1, C_S2, C_S4, C_S5]
        x = np.arange(len(top_idx))
        width = 0.2
        for ci, (label, color) in enumerate(zip(cw_labels, cw_colors)):
            vals = []
            for obp_idx in top_idx:
                cw = central_w.get(obp_idx)
                vals.append(float(cw[ci]) if cw is not None else 0.0)
            ax_cw.bar(x + ci * width, vals, width, label=label, color=color)
        ax_cw.set_xticks(x + 1.5 * width)
        ax_cw.set_xticklabels(names, rotation=28, ha='right', fontsize=8)
        ax_cw.set_ylabel('Pes central', fontsize=9)
        ax_cw.set_title('Perfil de pesos en els quals cada OBP guanya', fontsize=10)
        ax_cw.legend(fontsize=7.5, ncol=4)
        ax_cw.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save(pdf, fig)


# ── Pàgina final: Comparació 3 mètodes ───────────────────────────────────────

def page_comparison(pdf, result_table, rank_accept_d, rank_accept_s, voc_name, how_many):
    top_det = result_table.head(how_many)
    order_d = np.argsort(-rank_accept_d[:, 0])
    order_s = np.argsort(-rank_accept_s[:, 0])

    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(f"Comparació dels 3 Mètodes  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_HEADER, y=0.98)

    datasets = [
        ("M1: Pesos Deterministes\n(Score final)",
         top_det['OBP'].str[:22].tolist(),
         top_det['Score'].tolist(),
         C_SCORE),
        ("M2: Perturbació\np(rang 1)",
         [result_table.iloc[i]['OBP'][:22] for i in order_d[:how_many]],
         [rank_accept_d[i, 0] for i in order_d[:how_many]],
         '#4C72B0'),
        ("M3: SMAA Pur\np(rang 1)",
         [result_table.iloc[i]['OBP'][:22] for i in order_s[:how_many]],
         [rank_accept_s[i, 0] for i in order_s[:how_many]],
         '#55A868'),
    ]

    for ax, (title, names, vals, col) in zip(axes, datasets):
        n = len(names)
        ax.set_facecolor('white')
        bar_colors = [C_HIGHLIGHT if i == 0 else col for i in range(n)]
        ax.barh(range(n), vals, color=bar_colors, edgecolor='white', height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        xlim = max(max(vals) * 1.18, 0.08) if vals else 1.0
        ax.set_xlim(0, xlim)
        for i, v in enumerate(vals):
            ax.text(v + xlim * 0.01, i, f"{v:.3f}", va='center', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(pdf, fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("OBP FINDER — Generació d'Informe PDF | iGEM URV 2025")
    print("Executa els 3 mètodes i genera un informe gràfic descarregable.\n")

    exp_binding_file = resolve_data_file(BINDING_FILE_EXPERIMENTAL)
    info_file = resolve_data_file(INFO_FILE)

    for csv_path in (exp_binding_file, info_file):
        if not os.path.isfile(csv_path):
            print(f"ERROR: No es troba '{csv_path}'.")
            sys.exit(1)

    (binding_table, obp_info_table, cas_col, name_col,
     obp_name_list, LITERATURE_MIN, LITERATURE_MAX) = load_csv_files(exp_binding_file, info_file)

    # ── VOC diana ─────────────────────────────────────────────────────────────
    print("Cerca el VOC diana pel nom o CAS (ex: 3391-86-4):")
    while True:
        user_query = input("Nom o CAS del VOC diana: ").strip()
        if not user_query:
            print("  Cal escriure alguna cosa.")
            continue
        chosen_voc = validar_voc(binding_table, name_col, cas_col, user_query)
        if chosen_voc is None:
            print("  Torna-ho a intentar.")
            continue
        print(f"\n  VOC seleccionat: {chosen_voc[name_col]}  (CAS: {chosen_voc[cas_col]})")
        break

    voc_display   = chosen_voc[name_col]
    voc_cas       = str(chosen_voc[cas_col])
    voc_name_safe = re.sub(r'[^\w]+', '_', voc_display)[:40]
    ki_values_diana = pd.Series({col: chosen_voc[col] for col in obp_name_list})

    # ── Interferents ──────────────────────────────────────────────────────────
    print("\nFitxer d'interferents (noms o CAS, un per línia):")
    interferent_list = []
    while True:
        interf_path = input("Fitxer d'interferents (buit per saltar): ").strip().strip('"').strip("'")
        if not interf_path:
            print("  → Continuem sense interferents.")
            break
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents: {interferent_list}")
            interferent_list = validar_llista_interferents(
                interferent_list, binding_table, name_col, cas_col, interf_path
            )
            print(f"  ✓ {len(interferent_list)} vàlids")
            break
        print(f"  ✗ '{interf_path}' no trobat. Torna-ho a intentar o deixa buit.")

    # ── Pesos i tau ───────────────────────────────────────────────────────────
    weights = ask_user_for_weights()
    tau     = ask_user_for_tau()

    top_input = input("\nQuants candidats vols veure? [per defecte: 10]: ").strip()
    how_many  = int(top_input) if top_input.isdigit() and int(top_input) > 0 else 10

    # ── Paràmetres SMAA ───────────────────────────────────────────────────────
    print("\n── Configuració SMAA (Mètode 2: pesos aleatoris al voltant del valor introduït) ──────────────────────────")
    c_raw = input("  Dispersió c% per a la Perturbació [per defecte 50]: ").strip()
    try:
        c_val = float(c_raw)
        if not (0.1 <= c_val <= 100):
            raise ValueError
    except ValueError:
        c_val = 50.0
        print(f"  → Usant c={c_val:.0f}%")

    n_raw  = input("  Iteracions SMAA [per defecte 100000]: ").strip()
    n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 100_000

    # ── Ranking base ──────────────────────────────────────────────────────────
    print("\nCalculant ranking...")
    result_table = build_obp_ranking(
        ki_values_diana=ki_values_diana,
        obp_info_table=obp_info_table,
        binding_table=binding_table,
        cas_col=cas_col, name_col=name_col,
        interferent_list=interferent_list,
        obp_name_list=obp_name_list,
        weights=weights,
        ki_min_matrix=LITERATURE_MIN,
        ki_max_matrix=LITERATURE_MAX,
        tau=tau,
    )

    if result_table.empty:
        print("  Cap OBP té dades per a aquest VOC.")
        sys.exit(0)

    how_many = min(how_many, len(result_table))

    pesos_base_list = [
        weights['w_affinity'], weights['w_selectivity'],
        weights['w_stability'], weights['w_promiscuity'],
    ]
    subscores = result_table[['s1_affinity', 's2_selectivity',
                               's4_stability', 's5_promiscuity']].values
    te_dada = np.column_stack([
        np.ones(len(result_table), dtype=bool),
        result_table['has_s2_data'].values,
        np.ones(len(result_table), dtype=bool),
        result_table['has_s5_data'].values,
    ])

    # ── Mode 2: Perturbació de pesos ────────────────────────────────────────────
    print(f"\nExecutant Mètode 2 — Perturbació de pesos ({n_iter:,} iter, c={c_val:.0f}%)...")
    rank_accept_d, _, _ = smaa_complet(
        subscores=subscores, te_dada=te_dada,
        n_iter=n_iter, pesos_base=pesos_base_list, c=c_val,
        mode="perturbacio", seed=42,
    )

    # ── Mode 3: SMAA pur ──────────────────────────────────────────────────────
    print(f"Executant Mètode 3 — SMAA pur ({n_iter:,} iter)...")
    rank_accept_s, _, central_w = smaa_complet(
        subscores=subscores, te_dada=te_dada,
        n_iter=n_iter, mode="smaa", seed=42,
    )

    # ── Generar PDF ───────────────────────────────────────────────────────────
    now       = datetime.datetime.now()
    ts_file   = now.strftime("%Y-%m-%d_%H-%M-%S")
    ts_human  = now.strftime("%d/%m/%Y %H:%M:%S")
    pdf_name  = f"informe_OBP_{voc_name_safe}_{ts_file}.pdf"
    pdf_path  = os.path.join(os.getcwd(), pdf_name)

    print(f"\nGenerant informe PDF: {pdf_name}")

    have_predicted_page = False
    predicted_result_table = None
    predicted_text = None

    predicted_binding_file = resolve_data_file(BINDING_FILE_PREDICTED)
    predicted_affinity_file = resolve_data_file(BINDING_FILE_PREDICTED_AFFINITY)
    if os.path.isfile(predicted_binding_file) and os.path.isfile(predicted_affinity_file):
        print(f"\nGenerant també rànquing computacional amb dades predites...")
        pred_binding_table, _, _, _, _, _, _ = load_csv_files(predicted_binding_file, info_file)
        affinity_binding_table, _, _, _, _, _, _ = load_csv_files(predicted_affinity_file, info_file)

        chosen_voc_pred = validar_voc(pred_binding_table, name_col, cas_col, voc_cas)
        chosen_voc_aff = validar_voc(affinity_binding_table, name_col, cas_col, voc_cas)

        if chosen_voc_pred is not None:
            ki_values_diana_pred = pd.Series({col: chosen_voc_pred[col] for col in obp_name_list})
        else:
            ki_values_diana_pred = pd.Series({col: chosen_voc[col] for col in obp_name_list})

        ki_values_diana_affinity_pred = None
        if chosen_voc_aff is not None:
            ki_values_diana_affinity_pred = pd.Series({col: chosen_voc_aff[col] for col in obp_name_list})

        experimental_ki_values_diana = pd.Series({col: chosen_voc[col] for col in obp_name_list})
        predicted_result_table = build_obp_ranking(
            ki_values_diana=ki_values_diana_pred,
            obp_info_table=obp_info_table,
            binding_table=pred_binding_table,
            cas_col=cas_col, name_col=name_col,
            interferent_list=interferent_list,
            obp_name_list=obp_name_list,
            weights=weights,
            ki_min_matrix=LITERATURE_MIN,
            ki_max_matrix=LITERATURE_MAX,
            tau=tau,
            experimental_ki_values_diana=experimental_ki_values_diana,
            binding_source_label="predicted",
            ki_values_diana_affinity=ki_values_diana_affinity_pred,
        )

        if not predicted_result_table.empty:
            have_predicted_page = True
        else:
            predicted_text = (
                "✗ No s'ha pogut generar el rànquing predit. "
                "Revisa els fitxers de dades i torna-ho a executar."
            )
    else:
        predicted_text = (
            "✗ Fitxers de dades computacionals no disponibles: "
            f"{BINDING_FILE_PREDICTED} / {BINDING_FILE_PREDICTED_AFFINITY}."
        )

    with PdfPages(pdf_path) as pdf:
        page_cover(pdf, voc_display, voc_cas, weights, tau, how_many,
                   interferent_list, n_iter, c_val, ts_human)

        page_mode1_chart(pdf, result_table, voc_display, weights, how_many)
        page_mode1_table(pdf, result_table, voc_display, how_many)

        if have_predicted_page:
            page_top10_comparison(pdf, result_table, predicted_result_table,
                                  voc_display, how_many)
        elif predicted_text is not None:
            page_missing_predicted(pdf, voc_display, predicted_text)

        has_incomplete_data = (~result_table['has_all_interferents']).any()
        n_extra_pages = 0
        if has_incomplete_data:
            df_table2 = page_table1_table2(pdf, result_table, voc_display, how_many)
            page_table3_boltz(pdf, df_table2, voc_display)
            n_extra_pages = 2

        page_smaa(pdf, result_table, rank_accept_d, None,
                  voc_display, how_many, "Mètode 2 — Perturbació de pesos")

        page_smaa(pdf, result_table, rank_accept_s, central_w,
                  voc_display, how_many, "Mètode 3 — SMAA Pur")

        page_comparison(pdf, result_table, rank_accept_d, rank_accept_s,
                        voc_display, how_many)

    n_pages = 6 + n_extra_pages + (1 if have_predicted_page or predicted_text is not None else 0)
    print(f"\n✓ Informe guardat ({n_pages} pàgines): {pdf_path}")


if __name__ == '__main__':
    main()
