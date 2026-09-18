"""
generate_report.py  -  SCENTINEL | PDF report (iGEM URV 2026)
 
Runs the 3 methods (deterministic, Perturbation, pure SMAA) and generates a
downloadable PDF report on every run.
 
"""
 
import os
import sys
import re
import datetime
import textwrap
import numpy as np
import pandas as pd
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
 
from main import (
    load_csv_files,
    validate_voc,
    read_interferent_file,
    validate_interferent_list,
    ask_user_for_weights,
    ask_user_for_tau,
    build_obp_ranking,
    resolve_data_file,
    BINDING_FILE_EXPERIMENTAL,
    BINDING_FILE_PREDICTED,
    BINDING_FILE_PREDICTED_AFFINITY,
    INFO_FILE,
    DANGER_INTERFERENT_MARGIN,
    MAX_DANGER_NAMES,
    SUPERBINDER_KI_MAX,
    MAX_SUPERBINDER_NAMES,
)
import confidence as cf
from SMAAA import smaa_full
 
# -- SCENTINEL palette (lilac + green, liquid glass) -------------------------
C_LILAC      = "#8E7CC3"   # primary lilac
C_LILAC_DK   = "#5B4B8A"   # deep lilac (headers / text)
C_LILAC_LT   = "#D9CFF0"   # pale lilac (row tint / panels)
C_GREEN      = "#8FD14F"   # primary green
C_GREEN_DK   = "#4E9A2A"   # deep green
C_GREEN_LT   = "#E2F5CE"   # pale green (row tint / panels)
C_INK        = "#33304A"   # main text
C_MUTE       = "#8A85A0"   # muted text
C_BG         = "#F7F5FC"   # page background (very pale lilac)
C_PANEL      = "#FFFFFF"   # glass panel base
C_GLOW       = "#B7A9E0"   # soft glow
C_DANGER     = "#C0508A"   # danger accent (magenta-ish, on-brand)
 
# sub-score accents kept but re-tinted to the SCENTINEL family
C_S1 = "#7C9CE0"   # affinity  - soft blue
C_S2 = "#C0508A"   # selectivity - danger magenta
C_S4 = C_GREEN     # stability - green
C_S5 = C_LILAC     # promiscuity - lilac
 
# green->lilac colormap for heatmaps
CMAP_SCENTINEL = LinearSegmentedColormap.from_list(
    "scentinel", [C_GREEN_LT, C_GREEN, C_LILAC, C_LILAC_DK]
)
 
 
# -- Small helpers -----------------------------------------------------------
 
def _fig(landscape=True):
    size = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig = plt.figure(figsize=size)
    fig.patch.set_facecolor(C_BG)
    return fig
 
 
def _save(pdf, fig):
    pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
 
 
def _bubble(fig, x, y, w, h, facecolor=C_PANEL, edgecolor=C_GLOW,
            alpha=0.92, lw=1.4, pad=0.014, zorder=0):
    """Draw a rounded 'liquid-glass' panel in figure coordinates."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_zorder(zorder)
    box = FancyBboxPatch(
        (0.006, 0.006), 0.988, 0.988,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        transform=ax.transAxes,
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, mutation_aspect=h / max(w, 1e-6),
    )
    ax.add_patch(box)
    # top glass highlight
    hi = FancyBboxPatch(
        (0.04, 0.60), 0.92, 0.36,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        transform=ax.transAxes,
        facecolor="white", edgecolor="none", alpha=0.14,
    )
    ax.add_patch(hi)
    return ax
 
 
def _draw_orb(ax, cx=0.5, cy=0.5, r=0.42, green=True):
    """Draw a simple SCENTINEL liquid-glass orb as a logo."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    base = C_GREEN if green else C_LILAC
    edge = C_GREEN_DK if green else C_LILAC_DK
    # outer sphere
    for i, (rr, a) in enumerate([(r, 0.30), (r * 0.82, 0.35), (r * 0.6, 0.45)]):
        ax.add_patch(plt.Circle((cx, cy), rr, color=base, alpha=a, ec='none'))
    ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=edge, lw=1.6, alpha=0.7))
    # inner bright core (liquid-glass look)
    ax.add_patch(plt.Circle((cx, cy - r * 0.15), r * 0.5, color='white', alpha=0.55, ec='none'))
    # top highlight cap
    ax.add_patch(plt.Circle((cx, cy + r * 0.4), r * 0.42, color='white', alpha=0.4, ec='none'))
 
 
def _conf_color(conf):
    """Row tint by confidence: green (high) -> lilac (mid) -> pale danger (low)."""
    if pd.isna(conf):
        return "#FFFFFF"
    if conf >= 0.80:
        return C_GREEN_LT
    if conf >= 0.50:
        return C_LILAC_LT
    return "#F6E1EC"   # pale danger
 
 
def _fmt(v, nd=3, dash='-'):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return dash
    return f"{v:.{nd}f}"
 
 
# -- get-column helpers (tolerant to naming) ---------------------------------
 
def _col(row, *names, default=np.nan):
    """Return the first present column among names."""
    for n in names:
        if n in row.index:
            return row[n]
    return default
 
 
# -- Page 1: Cover -----------------------------------------------------------
 
def page_cover(pdf, voc_name, voc_cas, weights, tau, how_many,
               interferent_list, n_iter, c_val, ts):
    fig = _fig()
 
    # background wash panel
    _bubble(fig, 0.06, 0.08, 0.88, 0.84, facecolor="#FBFAFF",
            edgecolor=C_GLOW, alpha=0.9, pad=0.03)
 
    # logo orb
    ax_orb = fig.add_axes([0.42, 0.74, 0.16, 0.16])
    _draw_orb(ax_orb, green=True)
 
    kw = dict(ha='center', va='center', transform=fig.transFigure)
    fig.text(0.5, 0.70, "SCENTINEL", fontsize=30, fontweight='bold',
             color=C_LILAC_DK, **kw)
    fig.text(0.5, 0.655, "OBP candidate selection  ·  analysis report",
             fontsize=12.5, color=C_GREEN_DK, **kw)
    fig.text(0.5, 0.63, "iGEM URV 2026", fontsize=10, color=C_MUTE, **kw)
    fig.text(0.5, 0.605, f"Generated: {ts}", fontsize=9, color=C_MUTE, **kw)
 
    w = weights
    interf_str = ', '.join(interferent_list) if interferent_list else '- none -'
    if len(interf_str) > 90:
        interf_str = interf_str[:87] + '...'
 
    left = (
        f"Target VOC       :  {voc_name}\n"
        f"CAS              :  {voc_cas}\n"
        f"Top N candidates :  {how_many}\n"
        f"Interferents     :  {interf_str}\n"
    )
    right = (
        f"tau (selectivity):  {tau:.0f}\n"
        f"w1 affinity      :  {w['w_affinity']:.3f}\n"
        f"w2 selectivity   :  {w['w_selectivity']:.3f}\n"
        f"w4 stability     :  {w['w_stability']:.3f}\n"
        f"w5 promiscuity   :  {w['w_promiscuity']:.3f}\n"
    )
    smaa = (
        f"Method 2 (Perturbation):  c = {c_val:.0f}%,  {n_iter:,} iterations\n"
        f"Method 3 (pure SMAA)   :  {n_iter:,} iterations"
    )
 
    _bubble(fig, 0.12, 0.30, 0.50, 0.20, facecolor=C_GREEN_LT, alpha=0.55)
    fig.text(0.145, 0.475, left, ha='left', va='top', fontsize=10,
             fontfamily='monospace', color=C_INK, linespacing=1.7)
 
    _bubble(fig, 0.66, 0.30, 0.22, 0.20, facecolor=C_LILAC_LT, alpha=0.55)
    fig.text(0.675, 0.475, right, ha='left', va='top', fontsize=10,
             fontfamily='monospace', color=C_INK, linespacing=1.7)
 
    _bubble(fig, 0.12, 0.15, 0.76, 0.11, facecolor="#FFFFFF", alpha=0.9)
    fig.text(0.145, 0.205, "SMAA configuration", ha='left', va='top',
             fontsize=10, fontweight='bold', color=C_LILAC_DK)
    fig.text(0.145, 0.175, smaa, ha='left', va='top', fontsize=9.5,
             fontfamily='monospace', color=C_INK, linespacing=1.6)
 
    _save(pdf, fig)
 
 
# -- Page 2: unified detailed table + vineyard interferent detail ------------
 
def page_detailed_table(pdf, result_table, voc_name, how_many, is_predicted=False):
    top = result_table.head(how_many).copy()
 
    fig = _fig()
    fig.suptitle(f"SCENTINEL  ·  Detailed results  |  VOC: {voc_name}",
                 fontsize=14, fontweight='bold', color=C_LILAC_DK, y=0.975)
 
    # ---- main table (glass panel behind) ----
    _bubble(fig, 0.03, 0.46, 0.94, 0.46, facecolor=C_PANEL, alpha=0.95, pad=0.02)
    ax = fig.add_axes([0.045, 0.47, 0.91, 0.44])
    ax.axis('off')
 
    cols = ['#', 'OBP', 'Conserv.', 'Conf.', 'c2', 's1', 's2', 's4', 's5',
            'Ki (uM)', 'Interf', 'Danger', 'Type']
    rows = []
    tints = []
    for i, (_, row) in enumerate(top.iterrows()):
        conf   = _col(row, 'Confidence', 'Confianca')
        cons   = _col(row, 'Conservative', 'Conservador')
        c2     = _col(row, 'c2')
        n_tot  = int(_col(row, 'N_interferents_total', default=0) or 0)
        n_miss = int(_col(row, 'N_interferents_missing', default=0) or 0)
        n_dang = int(_col(row, 'N_danger_interferents', default=0) or 0)
        interf = f"{n_tot - n_miss}/{n_tot}" if n_tot else "-"
        s4 = _col(row, 's4_stability')
        rows.append([
            str(i + 1),
            str(row['OBP'])[:18],
            _fmt(cons, 4),
            _fmt(conf, 2),
            _fmt(c2, 2),
            _fmt(row['s1_affinity']),
            _fmt(row['s2_selectivity']),
            _fmt(s4) if not pd.isna(s4) else '?',
            _fmt(row['s5_promiscuity']),
            _fmt(_col(row, 'Ki_target_uM', 'Ki_diana_uM'), 2),
            interf,
            (f"! {n_dang}" if n_dang else "-"),
            str(row['Type'])[:11],
        ])
        tints.append(_conf_color(conf))
 
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.42)
 
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor(C_LILAC_DK)
        c.set_text_props(color='white', fontweight='bold')
        c.set_edgecolor('white')
 
    for i in range(len(rows)):
        for j in range(len(cols)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(tints[i])
            cell.set_edgecolor('white')
        # highlight the danger column if there are dangerous interferents
        if rows[i][11] != '-':
            tbl[i + 1, 11].set_text_props(color=C_DANGER, fontweight='bold')
    # top row accent
    tbl[1, 0].set_text_props(fontweight='bold')
 
    # ---- legend line ----
    fig.text(0.045, 0.445,
             "Conserv. = Score x Confidence   ·   c2 = selectivity confidence   ·   "
             "Interf = measured/total   ·   Danger = vineyard interferents binding "
             "as strong as the target",
             ha='left', va='top', fontsize=7.6, color=C_MUTE)
    fig.text(0.045, 0.428,
             "Row tint:  green = confidence >= 0.80    lilac = 0.50-0.80    "
             "pink = < 0.50",
             ha='left', va='top', fontsize=7.6, color=C_MUTE)
 
    # ---- vineyard interferent detail block (per OBP) ----
    _bubble(fig, 0.03, 0.06, 0.94, 0.34, facecolor="#FFFFFF", alpha=0.95, pad=0.02)
    fig.text(0.05, 0.375, "Dangerous vineyard interferents per candidate",
             ha='left', va='top', fontsize=10.5, fontweight='bold', color=C_DANGER)
    fig.text(0.05, 0.355,
             f"(Ki_interf <= Ki_target x {DANGER_INTERFERENT_MARGIN:.2f};  v = binds stronger than the target)",
             ha='left', va='top', fontsize=8, color=C_MUTE)
 
    y = 0.335
    any_danger = False
    line_h = 0.028
    max_lines = 10
    shown = 0
    for i, (_, row) in enumerate(top.iterrows()):
        perill = _col(row, 'Danger_interferents', default=[])
        if isinstance(perill, float):  # NaN
            perill = []
        if not perill:
            continue
        any_danger = True
        shown += 1
        if shown > max_lines:
            fig.text(0.05, y, "...", ha='left', va='top', fontsize=9, color=C_MUTE)
            break
        ki_d = _col(row, 'Ki_target_uM', 'Ki_diana_uM')
        names = MAX_DANGER_NAMES if MAX_DANGER_NAMES is not None else len(perill)
        labels = []
        for nom, ki_i in perill[:names]:
            arrow = "v" if (ki_i < ki_d and not np.isclose(ki_i, ki_d)) else ""
            labels.append(f"{nom}({ki_i:.1f}{arrow})")
        extra = f"  +{len(perill) - names}" if (MAX_DANGER_NAMES is not None
                                                and len(perill) > names) else ""
        txt = f"{i+1:>2}. {str(row['OBP'])[:16]:<16} Ki={ki_d:>6.2f} [{len(perill)}]  " \
              + ", ".join(labels) + extra
        if len(txt) > 120:
            txt = txt[:117] + "..."
        fig.text(0.05, y, txt, ha='left', va='top', fontsize=8,
                 fontfamily='monospace', color=C_INK)
        y -= line_h
 
    if not any_danger:
        fig.text(0.05, y, "OK  No candidate in the top has measured vineyard "
                          "interferents binding as strong as the target.",
                 ha='left', va='top', fontsize=9, color=C_GREEN_DK)
 
    _save(pdf, fig)
 
    # ---- second page: s5 superbinders ----
    _page_superbinders(pdf, result_table, voc_name, how_many)
 
 
def _page_superbinders(pdf, result_table, voc_name, how_many):
    """s5 superbinders page - VOCs from the whole database each OBP binds strongly.
    * = Ki <= SUPERBINDER_KI_MAX (strong)   v = binds stronger than the target."""
    if 'Superbinders' not in result_table.columns:
        return
 
    top = result_table.head(how_many).copy()
 
    fig = _fig()
    fig.suptitle(f"SCENTINEL  ·  s5 superbinders  |  VOC: {voc_name}",
                 fontsize=14, fontweight='bold', color=C_LILAC_DK, y=0.975)
 
    _bubble(fig, 0.03, 0.10, 0.94, 0.80, facecolor="#FFFFFF", alpha=0.95, pad=0.02)
    fig.text(0.05, 0.86, "Superbinders - VOCs (any, not only vineyard) bound strongly",
             ha='left', va='top', fontsize=11, fontweight='bold', color=C_LILAC_DK)
    fig.text(0.05, 0.835,
             f"* = Ki <= {SUPERBINDER_KI_MAX:.1f} uM (strong)    "
             f"v = binds stronger than the target    [n] = total superbinders",
             ha='left', va='top', fontsize=8.5, color=C_MUTE)
 
    y = 0.80
    line_h = 0.032
    none = True
    for i, (_, row) in enumerate(top.iterrows()):
        supers = row['Superbinders']
        if isinstance(supers, float):   # NaN
            supers = []
        if not supers:
            continue
        none = False
        shown = supers if MAX_SUPERBINDER_NAMES is None else supers[:MAX_SUPERBINDER_NAMES]
        labels = []
        for name, ki_v, is_strong, beats in shown:
            mark = ("*" if is_strong else "") + ("v" if beats else "")
            labels.append(f"{name}({ki_v:.1f}{mark})")
        extra = (f"  +{len(supers) - MAX_SUPERBINDER_NAMES}"
                 if (MAX_SUPERBINDER_NAMES is not None
                     and len(supers) > MAX_SUPERBINDER_NAMES) else "")
        head = f"{i+1:>2}. {str(row['OBP'])[:16]:<16} [{len(supers)}]  "
        # Wrap instead of truncating: the page has room and cutting the line
        # would silently drop the '+N' that says how many are not shown.
        wrapped = textwrap.wrap(", ".join(labels) + extra,
                                width=118 - len(head)) or [""]
        for k, piece in enumerate(wrapped):
            fig.text(0.05, y, (head if k == 0 else " " * len(head)) + piece,
                     ha='left', va='top', fontsize=8,
                     fontfamily='monospace', color=C_INK)
            y -= line_h
            if y < 0.12:
                fig.text(0.05, y, "...", ha='left', va='top', fontsize=9, color=C_MUTE)
                break
        if y < 0.12:
            break
 
    if none:
        fig.text(0.05, y, "OK  No candidate in the top binds any strong superbinder.",
                 ha='left', va='top', fontsize=9, color=C_GREEN_DK)
 
    _save(pdf, fig)
 
 
# -- Page: experimental vs computational -------------------------------------
 
def page_exp_vs_comp(pdf, experimental_table, predicted_table, voc_name, how_many):
    exp_top = experimental_table.head(how_many)
    pred_top = predicted_table.head(how_many)
 
    fig = _fig()
    fig.suptitle(f"SCENTINEL  ·  Experimental vs Computational  |  VOC: {voc_name}",
                 fontsize=14, fontweight='bold', color=C_LILAC_DK, y=0.975)
 
    panels = [
        (0.05, "Experimental data", exp_top, C_GREEN, C_GREEN_LT),
        (0.53, "Computational data (predicted)", pred_top, C_LILAC, C_LILAC_LT),
    ]
    for x0, title, df, color, tint in panels:
        _bubble(fig, x0, 0.12, 0.42, 0.78, facecolor=tint, alpha=0.5, pad=0.02)
        ax = fig.add_axes([x0 + 0.05, 0.16, 0.34, 0.66])
        ax.set_facecolor('none')
        values = df['Score'].tolist()
        names = df['OBP'].str[:20].tolist()
        n = len(values)
        bar_colors = [C_GREEN_DK if i == 0 else color for i in range(n)]
        ax.barh(range(n), values, color=bar_colors, edgecolor='white', height=0.66)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8, color=C_INK)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, color=C_LILAC_DK, fontweight='bold')
        ax.set_xlabel('Final score', fontsize=9, color=C_MUTE)
        xlim = max(max(values) * 1.2, 0.08) if values else 1.0
        ax.set_xlim(0, xlim)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        for i, val in enumerate(values):
            ax.text(val + xlim * 0.01, i, f"{val:.4f}", va='center',
                    fontsize=7, color=C_INK)
 
    _save(pdf, fig)
 
 
def page_missing_predicted(pdf, voc_name, message):
    fig = _fig()
    _bubble(fig, 0.1, 0.35, 0.8, 0.3, facecolor="#F6E1EC", alpha=0.7)
    fig.suptitle(f"Computational ranking not available  |  VOC: {voc_name}",
                 fontsize=13, fontweight='bold', color=C_LILAC_DK, y=0.7)
    fig.text(0.5, 0.5, message, ha='center', va='center', fontsize=11,
             color=C_DANGER, wrap=True)
    _save(pdf, fig)
 
 
# -- SMAA page ---------------------------------------------------------------
 
def page_smaa(pdf, result_table, rank_accept, central_w, voc_name, how_many, mode_label):
    m = len(result_table)
    n_rank_cols = min(m, 5)
    order = np.argsort(-rank_accept[:, 0])
    top_idx = order[:how_many]
    names   = [result_table.iloc[i]['OBP'][:22] for i in top_idx]
    p_best  = rank_accept[top_idx, 0]
 
    has_cw = central_w is not None and any(central_w.get(i) is not None for i in top_idx)
 
    fig = _fig()
    fig.suptitle(f"SCENTINEL  ·  {mode_label}  |  VOC: {voc_name}",
                 fontsize=14, fontweight='bold', color=C_LILAC_DK, y=0.975)
 
    if has_cw:
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.4,
                               left=0.08, right=0.95, top=0.9, bottom=0.1)
        ax_bar  = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_cw   = fig.add_subplot(gs[1, :])
    else:
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4,
                               left=0.08, right=0.95, top=0.85, bottom=0.15)
        ax_bar  = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_cw   = None
 
    # bar p(rank 1)
    ax_bar.set_facecolor('none')
    bar_colors = [C_GREEN_DK if i == 0 else C_LILAC for i in range(len(top_idx))]
    ax_bar.barh(range(len(top_idx)), p_best, color=bar_colors, edgecolor='white', height=0.66)
    ax_bar.set_yticks(range(len(top_idx)))
    ax_bar.set_yticklabels(names, fontsize=8.5, color=C_INK)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('p(rank 1)', fontsize=9, color=C_MUTE)
    ax_bar.set_title('Probability of being the best', fontsize=10.5, color=C_LILAC_DK)
    xlim = max(p_best.max() * 1.2, 0.08)
    ax_bar.set_xlim(0, xlim)
    for s in ax_bar.spines.values():
        s.set_visible(False)
    ax_bar.tick_params(length=0)
    for i, val in enumerate(p_best):
        ax_bar.text(val + xlim * 0.01, i, f"{val:.3f}", va='center', fontsize=7.5, color=C_INK)
 
    # heatmap
    heat_data = rank_accept[top_idx, :n_rank_cols]
    im = ax_heat.imshow(heat_data, aspect='auto', cmap=CMAP_SCENTINEL, vmin=0, vmax=1)
    ax_heat.set_xticks(range(n_rank_cols))
    ax_heat.set_xticklabels([f"Rank {i+1}" for i in range(n_rank_cols)], fontsize=8, color=C_INK)
    ax_heat.set_yticks(range(len(top_idx)))
    ax_heat.set_yticklabels(names, fontsize=8.5, color=C_INK)
    ax_heat.set_title("Rank acceptability", fontsize=10.5, color=C_LILAC_DK)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    for i in range(len(top_idx)):
        for j in range(n_rank_cols):
            val = heat_data[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=7, color='white' if val > 0.55 else C_INK)
 
    # central weights
    if has_cw and ax_cw is not None:
        ax_cw.set_facecolor('none')
        cw_labels = ['s1 Affinity', 's2 Selectivity', 's4 Stability', 's5 Promiscuity']
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
        ax_cw.set_xticklabels(names, rotation=28, ha='right', fontsize=8, color=C_INK)
        ax_cw.set_ylabel('Central weight', fontsize=9, color=C_MUTE)
        ax_cw.set_title('Weight profile in which each OBP wins', fontsize=10.5, color=C_LILAC_DK)
        ax_cw.legend(fontsize=7.5, ncol=4)
        ax_cw.set_ylim(0, 1)
        for s in ax_cw.spines.values():
            s.set_visible(False)
 
    _save(pdf, fig)
 
 
# -- Comparison of the 3 methods + who wins ----------------------------------
 
def page_comparison(pdf, result_table, rank_accept_d, rank_accept_s, voc_name, how_many):
    top_det = result_table.head(how_many)
    order_d = np.argsort(-rank_accept_d[:, 0])
    order_s = np.argsort(-rank_accept_s[:, 0])
 
    fig = _fig()
    fig.suptitle(f"SCENTINEL  ·  Comparison of the 3 methods  |  VOC: {voc_name}",
                 fontsize=14, fontweight='bold', color=C_LILAC_DK, y=0.975)
 
    datasets = [
        (0.04, "M1: Deterministic\n(final Score)",
         top_det['OBP'].str[:20].tolist(),
         top_det['Score'].tolist(),
         C_LILAC, C_LILAC_LT),
        (0.36, "M2: Perturbation\np(rank 1)",
         [result_table.iloc[i]['OBP'][:20] for i in order_d[:how_many]],
         [rank_accept_d[i, 0] for i in order_d[:how_many]],
         C_S1, "#E3EAFB"),
        (0.68, "M3: Pure SMAA\np(rank 1)",
         [result_table.iloc[i]['OBP'][:20] for i in order_s[:how_many]],
         [rank_accept_s[i, 0] for i in order_s[:how_many]],
         C_GREEN, C_GREEN_LT),
    ]
 
    for x0, title, names, vals, col, tint in datasets:
        _bubble(fig, x0, 0.12, 0.29, 0.76, facecolor=tint, alpha=0.5, pad=0.02)
        ax = fig.add_axes([x0 + 0.035, 0.17, 0.23, 0.6])
        ax.set_facecolor('none')
        n = len(names)
        bar_colors = [C_GREEN_DK if i == 0 else col for i in range(n)]
        ax.barh(range(n), vals, color=bar_colors, edgecolor='white', height=0.66)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=7.5, color=C_INK)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=C_LILAC_DK, fontweight='bold')
        xlim = max(max(vals) * 1.22, 0.08) if vals else 1.0
        ax.set_xlim(0, xlim)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        for i, v in enumerate(vals):
            ax.text(v + xlim * 0.01, i, f"{v:.3f}", va='center', fontsize=6.8, color=C_INK)
 
    # who-wins note
    winner_m1 = top_det.iloc[0]['OBP'] if len(top_det) else "-"
    winner_m2 = result_table.iloc[order_d[0]]['OBP'] if len(order_d) else "-"
    winner_m3 = result_table.iloc[order_s[0]]['OBP'] if len(order_s) else "-"
    note = (f"Winner  ·  M1 (deterministic): {winner_m1}    "
            f"M2 (perturbation): {winner_m2}    M3 (pure SMAA): {winner_m3}")
    _bubble(fig, 0.08, 0.03, 0.84, 0.06, facecolor="#FFFFFF", alpha=0.92)
    fig.text(0.5, 0.06, note, ha='center', va='center', fontsize=9,
             color=C_INK, fontweight='bold')
 
    _save(pdf, fig)
 
 
# -- Main --------------------------------------------------------------------
 
def main():
    print("SCENTINEL - PDF Report Generation | iGEM URV 2025")
    print("Runs the 3 methods and generates a downloadable graphical report.\n")
 
    exp_binding_file = resolve_data_file(BINDING_FILE_EXPERIMENTAL)
    info_file = resolve_data_file(INFO_FILE)
 
    for csv_path in (exp_binding_file, info_file):
        if not os.path.isfile(csv_path):
            print(f"ERROR: Cannot find '{csv_path}'.")
            sys.exit(1)
 
    (binding_table, obp_info_table, cas_col, name_col,
     obp_name_list, LITERATURE_MIN, LITERATURE_MAX) = load_csv_files(exp_binding_file, info_file)
 
    # -- Target VOC ----------------------------------------------------------
    print("Search the target VOC by name or CAS (e.g. 3391-86-4):")
    while True:
        user_query = input("Name or CAS of the target VOC: ").strip()
        if not user_query:
            print("  You must type something.")
            continue
        chosen_voc = validate_voc(binding_table, name_col, cas_col, user_query)
        if chosen_voc is None:
            print("  Try again.")
            continue
        print(f"\n  Selected VOC: {chosen_voc[name_col]}  (CAS: {chosen_voc[cas_col]})")
        break
 
    voc_display   = chosen_voc[name_col]
    voc_cas       = str(chosen_voc[cas_col])
    voc_name_safe = re.sub(r'[^\w]+', '_', voc_display)[:40]
    ki_values_target = pd.Series({col: chosen_voc[col] for col in obp_name_list})
 
    # -- Interferents --------------------------------------------------------
    print("\nInterferent file (names or CAS, one per line):")
    interferent_list = []
    while True:
        interf_path = input("Interferent file (leave empty to skip): ").strip().strip('"').strip("'")
        if not interf_path:
            print("  -> Continuing without interferents.")
            break
        if os.path.isfile(interf_path):
            interferent_list = read_interferent_file(interf_path)
            print(f"  {len(interferent_list)} interferents: {interferent_list}")
            interferent_list = validate_interferent_list(
                interferent_list, binding_table, name_col, cas_col, interf_path
            )
            print(f"  ok {len(interferent_list)} valid")
            break
        print(f"  x '{interf_path}' not found. Try again or leave empty.")
 
    # -- Weights and tau -----------------------------------------------------
    weights = ask_user_for_weights()
    tau     = ask_user_for_tau()
 
    top_input = input("\nHow many candidates do you want to see? [default: 10]: ").strip()
    how_many  = int(top_input) if top_input.isdigit() and int(top_input) > 0 else 10
 
    # -- SMAA parameters -----------------------------------------------------
    print("\n-- SMAA configuration (Method 2: random weights around the value you entered) --")
    c_raw = input("  Dispersion c% for the Perturbation [default 50]: ").strip()
    try:
        c_val = float(c_raw)
        if not (0.1 <= c_val <= 100):
            raise ValueError
    except ValueError:
        c_val = 50.0
        print(f"  -> Using c={c_val:.0f}%")
 
    n_raw  = input("  SMAA iterations [default 100000]: ").strip()
    n_iter = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 100_000
 
    # -- Base ranking --------------------------------------------------------
    print("\nComputing ranking...")
    result_table = build_obp_ranking(
        ki_values_target=ki_values_target,
        obp_info_table=obp_info_table,
        binding_table=binding_table,
        cas_col=cas_col, name_col=name_col,
        interferent_list=interferent_list,
        obp_name_list=obp_name_list,
        weights=weights,
        ki_min_matrix=LITERATURE_MIN,
        ki_max_matrix=LITERATURE_MAX,
        tau=tau,
        target_name=voc_display,
    )
 
    if result_table.empty:
        print("  No OBP has data for this VOC.")
        sys.exit(0)
 
    how_many = min(how_many, len(result_table))
 
    # -- Confidence (adds Confidence / c2 / Conservative / Danger cols) ------
    diag_c1 = cf.load_diag_c1(resolve_data_file(os.path.join("results", "tanimoto_diag_table.csv")))
    result_table = cf.compute_confidences(
        result_table, weights,
        target_row_idx=chosen_voc.name,
        interferent_row_idx=result_table.attrs.get('interferent_row_idx', {}),
        diag_c1=diag_c1,
        is_predicted=False,
    )
 
    base_weights_list = [
        weights['w_affinity'], weights['w_selectivity'],
        weights['w_stability'], weights['w_promiscuity'],
    ]
    subscores = result_table[['s1_affinity', 's2_selectivity',
                               's4_stability', 's5_promiscuity']].values
    has_data = np.column_stack([
        np.ones(len(result_table), dtype=bool),
        result_table['has_s2_data'].values,
        np.ones(len(result_table), dtype=bool),
        result_table['has_s5_data'].values,
    ])
 
    print(f"\nRunning Method 2 - weight perturbation ({n_iter:,} iter, c={c_val:.0f}%)...")
    rank_accept_d, _, _ = smaa_full(
        subscores=subscores, has_data=has_data,
        n_iter=n_iter, base_weights=base_weights_list, c=c_val,
        mode="perturbation", seed=42,
    )
 
    print(f"Running Method 3 - pure SMAA ({n_iter:,} iter)...")
    rank_accept_s, _, central_w = smaa_full(
        subscores=subscores, has_data=has_data,
        n_iter=n_iter, mode="smaa", seed=42,
    )
 
    # -- Predicted (computational) ranking -----------------------------------
    now       = datetime.datetime.now()
    ts_file   = now.strftime("%Y-%m-%d_%H-%M-%S")
    ts_human  = now.strftime("%d/%m/%Y %H:%M:%S")
    pdf_name  = f"SCENTINEL_report_{voc_name_safe}_{ts_file}.pdf"
    pdf_path  = os.path.join(os.getcwd(), pdf_name)
 
    print(f"\nGenerating PDF report: {pdf_name}")
 
    have_predicted_page = False
    predicted_result_table = None
    predicted_text = None
 
    predicted_binding_file = resolve_data_file(BINDING_FILE_PREDICTED)
    predicted_affinity_file = resolve_data_file(BINDING_FILE_PREDICTED_AFFINITY)
    if os.path.isfile(predicted_binding_file) and os.path.isfile(predicted_affinity_file):
        print("Also generating the computational ranking with predicted data...")
        pred_binding_table, _, _, _, _, _, _ = load_csv_files(predicted_binding_file, info_file)
        affinity_binding_table, _, _, _, _, _, _ = load_csv_files(predicted_affinity_file, info_file)
 
        chosen_voc_pred = validate_voc(pred_binding_table, name_col, cas_col, voc_cas)
        chosen_voc_aff = validate_voc(affinity_binding_table, name_col, cas_col, voc_cas)
 
        if chosen_voc_pred is not None:
            ki_values_target_pred = pd.Series({col: chosen_voc_pred[col] for col in obp_name_list})
        else:
            ki_values_target_pred = pd.Series({col: chosen_voc[col] for col in obp_name_list})
 
        ki_values_target_affinity_pred = None
        if chosen_voc_aff is not None:
            ki_values_target_affinity_pred = pd.Series({col: chosen_voc_aff[col] for col in obp_name_list})
 
        experimental_ki_values_target = pd.Series({col: chosen_voc[col] for col in obp_name_list})
        predicted_result_table = build_obp_ranking(
            ki_values_target=ki_values_target_pred,
            obp_info_table=obp_info_table,
            binding_table=pred_binding_table,
            cas_col=cas_col, name_col=name_col,
            interferent_list=interferent_list,
            obp_name_list=obp_name_list,
            weights=weights,
            ki_min_matrix=LITERATURE_MIN,
            ki_max_matrix=LITERATURE_MAX,
            tau=tau,
            experimental_ki_values_target=experimental_ki_values_target,
            binding_source_label="predicted",
            ki_values_target_affinity=ki_values_target_affinity_pred,
            target_name=voc_display,
        )
 
        if not predicted_result_table.empty:
            have_predicted_page = True
        else:
            predicted_text = ("x Could not generate the predicted ranking. "
                              "Check the data files and run it again.")
    else:
        predicted_text = ("x Computational data files not available: "
                          f"{BINDING_FILE_PREDICTED} / {BINDING_FILE_PREDICTED_AFFINITY}.")
 
    # -- Build the PDF -------------------------------------------------------
    with PdfPages(pdf_path) as pdf:
        page_cover(pdf, voc_display, voc_cas, weights, tau, how_many,
                   interferent_list, n_iter, c_val, ts_human)
 
        page_detailed_table(pdf, result_table, voc_display, how_many)
 
        if have_predicted_page:
            page_exp_vs_comp(pdf, result_table, predicted_result_table,
                             voc_display, how_many)
        elif predicted_text is not None:
            page_missing_predicted(pdf, voc_display, predicted_text)
 
        page_smaa(pdf, result_table, rank_accept_d, None,
                  voc_display, how_many, "Method 2 - Weight Perturbation")
        page_smaa(pdf, result_table, rank_accept_s, central_w,
                  voc_display, how_many, "Method 3 - Pure SMAA")
 
        page_comparison(pdf, result_table, rank_accept_d, rank_accept_s,
                        voc_display, how_many)
 
    print(f"\nOK Report saved: {pdf_path}")
 
 
if __name__ == '__main__':
    main()