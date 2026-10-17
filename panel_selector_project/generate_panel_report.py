"""
generate_panel_report.py  -  SCENTINEL | PDF report of the panel tools (iGEM URV 2026)

Runs Tool 1 (specificity: nothing imitates the target) with experimental data and with
predicted (Tanimoto) data, and - if a vineyard input table is given - Tool 2
(discrimination: healthy vs diseased, Fisher J) in both modes, then writes one PDF:

    1  Cover: target, interferents, settings and the verdict of every run
    2  Tool 1 - experimental: panel, residual build-up, verdict
    3  Tool 1 - experimental: what the panel bets on + pairs to measure
    4-5  same with predicted data
    6  Tool 1: experimental vs predicted
    7-10 Tool 2 (if --profiles): best single OBPs, panel J, Ki known, verdict and bets
    11 Lab priorities: the measurements that decide whether the panels can be trusted
    12 Method: how every number is computed and how missing Ki are handled

Examples
    python generate_panel_report.py --target "methyl salicylate" \
        --interferents data/interferents_example.txt --start-with DcitOBP3 \
        --profiles examples/tool2_mesa_interferentes/mesa_interferentes_real_48hpi.csv
    python generate_panel_report.py            (asks for everything)
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from panel_selector import ScentinelData, CompoundNotFound, CompoundAmbiguous, read_interferent_file, error_from_j
import scentinel_tool1_specificity as t1
import scentinel_tool2_discrimination as t2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- SCENTINEL palette (same as generate_report.py) --------------------------
C_LILAC = "#8E7CC3"
C_LILAC_DK = "#5B4B8A"
C_LILAC_LT = "#D9CFF0"
C_GREEN = "#8FD14F"
C_GREEN_DK = "#4E9A2A"
C_GREEN_LT = "#E2F5CE"
C_INK = "#33304A"
C_MUTE = "#8A85A0"
C_BG = "#F7F5FC"
C_PANEL = "#FFFFFF"
C_GLOW = "#B7A9E0"
C_DANGER = "#C0508A"
C_DANGER_LT = "#F6E1EC"

GRADE_COLORS = {"SOLID": (C_GREEN_DK, C_GREEN_LT), "CONDITIONAL": (C_LILAC_DK, C_LILAC_LT),
                "WEAK": (C_DANGER, C_DANGER_LT), "FAILS": (C_DANGER, C_DANGER_LT),
                "NOT ENOUGH": (C_DANGER, C_DANGER_LT), "N/A": (C_MUTE, "#EEEEEE")}
SHORT = {"experimental": "exp.", "predicted": "pred."}
GRADE_MEANING = {"SOLID": "no single unverified value can plausibly break it",
                 "CONDITIONAL": "rests on unverified values, predictions do not contradict them",
                 "WEAK": "a bet goes against the prediction, or all predictions together break it",
                 "FAILS": "does not reach the limit even with the data used",
                 "NOT ENOUGH": "does not reach the limit even with the data used",
                 "N/A": "could not be computed"}


# -- page helpers (same look as generate_report.py) ---------------------------
def _fig():
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(C_BG)
    return fig


def _save(pdf, fig):
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def _bubble(fig, x, y, w, h, facecolor=C_PANEL, edgecolor=C_GLOW, alpha=0.92, lw=1.4, pad=0.014):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_zorder(0)
    ax.add_patch(FancyBboxPatch((0.006, 0.006), 0.988, 0.988, boxstyle=f"round,pad=0,rounding_size={pad}",
                                transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor,
                                linewidth=lw, alpha=alpha, mutation_aspect=h / max(w, 1e-6)))
    ax.add_patch(FancyBboxPatch((0.04, 0.60), 0.92, 0.36, boxstyle=f"round,pad=0,rounding_size={pad}",
                                transform=ax.transAxes, facecolor="white", edgecolor="none", alpha=0.14))
    return ax


def _draw_orb(ax, green=True):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    base, edge = (C_GREEN, C_GREEN_DK) if green else (C_LILAC, C_LILAC_DK)
    for rr, a in [(0.42, 0.30), (0.34, 0.35), (0.25, 0.45)]:
        ax.add_patch(plt.Circle((0.5, 0.5), rr, color=base, alpha=a, ec="none"))
    ax.add_patch(plt.Circle((0.5, 0.5), 0.42, fill=False, ec=edge, lw=1.6, alpha=0.7))
    ax.add_patch(plt.Circle((0.5, 0.44), 0.21, color="white", alpha=0.55, ec="none"))
    ax.add_patch(plt.Circle((0.5, 0.67), 0.18, color="white", alpha=0.4, ec="none"))


def _title(fig, text, sub=None):
    fig.suptitle(text, fontsize=14, fontweight="bold", color=C_LILAC_DK, y=0.965)
    if sub:
        fig.text(0.5, 0.925, sub, ha="center", va="center", fontsize=9, color=C_MUTE)


def _table(fig, rect, cols, rows, widths, fontsize=7.6, tints=None, row_h=1.35, wrap=None, bold_col=None,
           color_cells=None):
    """Matplotlib table in a glass panel. wrap = {col index: chars} wraps long text."""
    _bubble(fig, rect[0] - 0.012, rect[1] - 0.012, rect[2] + 0.024, rect[3] + 0.024, pad=0.02)
    ax = fig.add_axes(rect); ax.axis("off")
    if not rows:
        ax.text(0.02, 0.9, "(nothing to show)", fontsize=9, color=C_MUTE, va="top")
        return ax
    if wrap:
        rows = [[("\n".join(textwrap.wrap(str(v), wrap[j])) if j in wrap else v) for j, v in enumerate(r)]
                for r in rows]
    tbl = ax.table(cellText=rows, colLabels=cols, colWidths=widths, loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, row_h)
    for j in range(len(cols)):
        c = tbl[0, j]
        c.set_facecolor(C_LILAC_DK); c.set_text_props(color="white", fontweight="bold"); c.set_edgecolor("white")
    for i, r in enumerate(rows):
        n_lines = max(str(v).count("\n") + 1 for v in r)
        for j in range(len(cols)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(tints[i] if tints else "#FFFFFF")
            cell.set_edgecolor("white")
            if n_lines > 1:
                cell.set_height(cell.get_height() * n_lines * 0.85)
            if color_cells and (i, j) in color_cells:
                cell.set_text_props(color=color_cells[(i, j)], fontweight="bold")
        if bold_col is not None:
            tbl[i + 1, bold_col].set_text_props(fontweight="bold")
    return ax


def _badge(fig, x, y, w, h, grade, label, sub=""):
    fg, bg = GRADE_COLORS.get(grade, GRADE_COLORS["N/A"])
    _bubble(fig, x, y, w, h, facecolor=bg, edgecolor=fg, alpha=0.9, lw=1.6)
    fig.text(x + w / 2, y + h * 0.72, label, ha="center", va="center", fontsize=9, color=C_INK)
    fig.text(x + w / 2, y + h * 0.42, grade, ha="center", va="center", fontsize=15, fontweight="bold", color=fg)
    if sub:
        fig.text(x + w / 2, y + h * 0.15, sub, ha="center", va="center", fontsize=7.5, color=C_MUTE)


def _f(v, fmt="{:.3g}", dash="-"):
    return dash if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def _err(J):
    e = error_from_j(J)
    return f"{100 * e:.1f} %" if e >= 0.001 else "<0.1 %"


def _page_unavailable(pdf, title, message):
    fig = _fig()
    _title(fig, title)
    _bubble(fig, 0.12, 0.38, 0.76, 0.26, facecolor=C_DANGER_LT, alpha=0.75)
    fig.text(0.5, 0.51, "\n".join(textwrap.wrap(message, 95)), ha="center", va="center", fontsize=11,
             color=C_DANGER)
    _save(pdf, fig)


# -- cover ----------------------------------------------------------------------
def page_cover(pdf, ctx, runs1, runs2, ts):
    fig = _fig()
    _bubble(fig, 0.06, 0.06, 0.88, 0.86, facecolor="#FBFAFF", pad=0.03)
    ax = fig.add_axes([0.44, 0.76, 0.12, 0.12]); _draw_orb(ax)
    kw = dict(ha="center", va="center")
    fig.text(0.5, 0.73, "SCENTINEL", fontsize=28, fontweight="bold", color=C_LILAC_DK, **kw)
    fig.text(0.5, 0.69, "OBP panel selection  ·  specificity & discrimination report", fontsize=12,
             color=C_GREEN_DK, **kw)
    fig.text(0.5, 0.665, f"iGEM URV 2026   ·   Generated: {ts}", fontsize=9, color=C_MUTE, **kw)

    interf = ", ".join(ctx["interf_short"])
    left = (f"Target VOC      :  {ctx['target_short']}  (CAS {ctx['cas']})\n"
            f"Starting OBP    :  {', '.join(ctx['start'])}\n"
            f"Interferents ({len(ctx['interf_short'])}):  "
            + "\n                   ".join(textwrap.wrap(interf, 50)[:3])
            + ("  ..." if len(textwrap.wrap(interf, 50)) > 3 else "") + "\n"
            f"Vineyard profile:  {ctx['profiles_name'] or '- not given (Tool 2 skipped) -'}")
    right = (f"Tool 1 noise floor   :  {ctx['noise']}\n"
             f"Predicted min. conf. :  {ctx['min_conf']}\n"
             f"Tool 2 max. error    :  {ctx['max_error']:.0%}  (J >= {t2.j_for_error(ctx['max_error']):.2f})\n"
             f"Tool 2 sensor noise  :  {ctx['sensor_noise']:.0%}")
    _bubble(fig, 0.10, 0.43, 0.50, 0.19, facecolor=C_GREEN_LT, alpha=0.55)
    fig.text(0.12, 0.605, left, ha="left", va="top", fontsize=9, family="monospace", color=C_INK, linespacing=1.6)
    _bubble(fig, 0.62, 0.43, 0.30, 0.19, facecolor=C_LILAC_LT, alpha=0.55)
    fig.text(0.632, 0.605, right, ha="left", va="top", fontsize=9, family="monospace", color=C_INK,
             linespacing=1.6)

    fig.text(0.5, 0.39, "Verdict of every run", ha="center", fontsize=11, fontweight="bold", color=C_LILAC_DK)
    badges = [("Tool 1 · experimental", runs1.get("experimental")), ("Tool 1 · predicted", runs1.get("predicted")),
              ("Tool 2 · experimental", runs2.get("experimental")), ("Tool 2 · predicted", runs2.get("predicted"))]
    x = 0.10
    for label, run in badges:
        if run is None or isinstance(run, str):
            grade, sub = "N/A", ("not run" if run is None else "not available")
        else:
            grade = run["grade"]
            sub = f"{len(run['panel'])} OBPs"
        _badge(fig, x, 0.21, 0.185, 0.15, grade, label, sub)
        x += 0.205
    fig.text(0.5, 0.15, "SOLID = no single unverified value can break it   ·   CONDITIONAL = rests on unverified values, "
             "predictions agree\nWEAK = a bet goes against the prediction (measure first)   ·   "
             "FAILS / NOT ENOUGH = does not reach the limit", ha="center", va="center", fontsize=8, color=C_MUTE,
             linespacing=1.6)
    _save(pdf, fig)


# -- Tool 1 -----------------------------------------------------------------------
def page_t1_panel(pdf, run, label, D):
    fig = _fig()
    _title(fig, f"Tool 1 · Specificity · {label}",
           f"Smallest OBP panel such that no mixture of the interferents imitates {run['target_short']}  ·  "
           f"{run['n_candidates']} candidate OBPs of {run['n_with_target']} with target data "
           f"({run['n_blind']} blind removed)  ·  priority: {run['priority']}")
    st = run["steps"]
    rows, tints = [], []
    for r in st.itertuples():
        rows.append([str(r.step), r.OBP, r.added_by, f"{r.residual:.4f}", _f(r.Ki_target_uM, "{:.2f}"),
                     f"{r.c2:.2f}", f"{r.cells_measured}/{r.cells_imputed}", r.covers or ""])
        tints.append(C_GREEN_LT if r.c2 >= 0.8 else (C_LILAC_LT if r.c2 >= 0.5 else C_DANGER_LT))
    _table(fig, [0.04, 0.47, 0.50, 0.40], ["#", "OBP", "added by", "residual", "Ki tgt (uM)", "c2", "meas/imp",
                                          "covers"], rows, [0.05, 0.2, 0.12, 0.13, 0.13, 0.08, 0.12, 0.17],
           tints=tints, wrap={7: 18}, fontsize=7.4)
    fig.text(0.04, 0.445, "row tint: c2 >= 0.8 green · 0.5-0.8 lilac · < 0.5 pink   ·   c2 = how much is known about "
             "the interferents on that OBP (measured 1, predicted x confidence, missing 0)",
             fontsize=6.8, color=C_MUTE)

    # residual build-up chart
    _bubble(fig, 0.57, 0.47, 0.40, 0.42, pad=0.02)
    ax = fig.add_axes([0.62, 0.53, 0.33, 0.31]); ax.set_facecolor("none")
    colors = [C_LILAC if a != "cover" else C_GREEN for a in st["added_by"]]
    ax.bar(range(len(st)), st["residual"], color=colors, edgecolor="white")
    ax.axhline(run["noise"], color=C_DANGER, ls="--", lw=1.3)
    ax.text(-0.45, run["noise"], f"noise {run['noise']}", color=C_DANGER, fontsize=7.5, va="bottom", ha="left")
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st["OBP"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("residual (best fake mixture)", fontsize=8, color=C_MUTE)
    ax.set_title("Residual as OBPs are added", fontsize=10, color=C_LILAC_DK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=7)

    # verdict box
    fg, bg = GRADE_COLORS.get(run["grade"], GRADE_COLORS["N/A"])
    _bubble(fig, 0.04, 0.07, 0.93, 0.33, facecolor=bg, edgecolor=fg, alpha=0.6)
    fig.text(0.06, 0.37, f"VERDICT: {run['grade']}", fontsize=15, fontweight="bold", color=fg, va="top")
    lines = [f"Panel: {' + '.join(run['panel'])}",
             f"Residual {run['final_resid']:.4f} vs noise {run['noise']} -> "
             + ("clears the noise floor on the data used." if run["passed"] else "does NOT clear the noise floor."),
             f"Why: {run['why']}."]
    if pd.notna(run["r_all"]):
        lines.append(f"If every not-measured pair took its Tanimoto prediction at once: residual {run['r_all']:.4f} -> "
                     + ("still passes." if run["r_all"] >= run["noise"] else "FAILS."))
    if run["recipe"]:
        lines.append("Closest fake mixture: " + ", ".join(f"{c:.3f} x {n}" for c, n in run["recipe"][:6]))
    for w in run["warnings"]:
        lines.append("! " + w)
    if run["uncovered"]:
        lines.append("! No OBP binds: " + ", ".join(D.short(i) for i in run["uncovered"]) + " (invisible to the panel).")
    txt = "\n".join("\n".join(textwrap.wrap(l, 150)) for l in lines)
    fig.text(0.06, 0.315, txt, fontsize=8.6, color=C_INK, va="top", linespacing=1.55)
    _save(pdf, fig)


def page_t1_bets(pdf, run, label):
    fig = _fig()
    _title(fig, f"Tool 1 · What the panel bets on · {label}",
           "Every value the 'pass' depends on that nobody measured (or measured once and is fragile)  ·  "
           "most fragile first")
    b = run["bets"]
    b = b[b["against"] | (b["off_by"] <= t1.BET_MAX_FOLD)] if b is not None and len(b) else pd.DataFrame()
    rows, colors = [], {}
    for i, r in enumerate(b.head(9).itertuples()):
        risk = "AGAINST prediction" if r.against else f"lost if off by x{r.off_by:.3g}"
        rows.append([str(i + 1), r.bet, r.evidence, risk])
        if r.against:
            colors[(i, 3)] = C_DANGER
    extra = f"  (+{len(b) - 9} more in the bets CSV)" if len(b) > 9 else ""
    fig.text(0.04, 0.89, f"Bets ({len(b)}){extra}", fontsize=10.5, fontweight="bold", color=C_LILAC_DK)
    _table(fig, [0.04, 0.47, 0.92, 0.40], ["#", "for the panel to work, this must be true", "data", "risk"], rows,
           [0.03, 0.45, 0.37, 0.15], wrap={1: 78, 2: 62, 3: 20}, fontsize=7, color_cells=colors, row_h=1.2)

    wl = run["worklist"]
    fig.text(0.04, 0.41, f"Pairs to measure ({len(wl)}), most urgent first", fontsize=10.5, fontweight="bold",
             color=C_LILAC_DK)
    rows, colors = [], {}
    for i, r in enumerate(wl.head(10).itertuples()):
        verdict = getattr(r, "verdict", "")
        verdict = verdict if isinstance(verdict, str) else ""
        bi = getattr(r, "breaks_if", "")
        kp = getattr(r, "Ki_predicted_uM", np.nan)
        rows.append([r.OBP, r.compound, r.status, verdict, bi if isinstance(bi, str) else "",
                     _f(kp), _f(r.confidence, "{:.2f}")])
        if verdict.startswith("BREAKS"):
            colors[(i, 3)] = C_DANGER
    _table(fig, [0.04, 0.06, 0.92, 0.32], ["OBP", "compound", "status", "risk", "breaks the panel if", "Ki pred (uM)",
                                          "conf"], rows, [0.16, 0.22, 0.09, 0.17, 0.16, 0.11, 0.09], fontsize=7.4,
           color_cells=colors, row_h=1.25)
    _save(pdf, fig)


def page_t1_compare(pdf, runs1):
    fig = _fig()
    _title(fig, "Tool 1 · Experimental vs predicted data",
           "Mode 1 picks the best-MEASURED useful OBPs; mode 2 lets the Tanimoto predictions take part in the choice")
    for x0, key, title, tint in [(0.05, "experimental", "Experimental data", C_GREEN_LT),
                                 (0.53, "predicted", "Predicted (Tanimoto) data", C_LILAC_LT)]:
        run = runs1.get(key)
        _bubble(fig, x0, 0.10, 0.42, 0.78, facecolor=tint, alpha=0.5, pad=0.02)
        fig.text(x0 + 0.21, 0.85, title, ha="center", fontsize=11, fontweight="bold", color=C_LILAC_DK)
        if run is None or isinstance(run, str):
            fig.text(x0 + 0.02, 0.78, "\n".join(textwrap.wrap(run or "not run", 60)), fontsize=9, color=C_DANGER,
                     va="top")
            continue
        fg, _ = GRADE_COLORS.get(run["grade"], GRADE_COLORS["N/A"])
        fig.text(x0 + 0.21, 0.80, run["grade"], ha="center", fontsize=16, fontweight="bold", color=fg)
        b = run["bets"] if run["bets"] is not None else pd.DataFrame(columns=["against", "off_by"])
        b = b[b["against"] | (b["off_by"] <= t1.BET_MAX_FOLD)] if len(b) else b
        info = (f"Panel ({len(run['panel'])}): " + " + ".join(run["panel"]) + "\n"
                f"Residual: {run['final_resid']:.4f}   (noise {run['noise']})\n"
                f"Bets: {len(b)}   against the prediction: {int(b['against'].sum()) if len(b) else 0}\n"
                f"Pairs without a lab Ki: {len(run['worklist'])}")
        fig.text(x0 + 0.02, 0.76, "\n".join("\n".join(textwrap.wrap(l, 62)) for l in info.split("\n")),
                 fontsize=8.6, color=C_INK, va="top", linespacing=1.6)
        ax = fig.add_axes([x0 + 0.06, 0.15, 0.32, 0.36]); ax.set_facecolor("none")
        st = run["steps"]
        ax.plot(range(1, len(st) + 1), st["residual"], marker="o", color=C_LILAC_DK)
        ax.axhline(run["noise"], color=C_DANGER, ls="--", lw=1.2)
        ax.set_xticks(range(1, len(st) + 1)); ax.set_xticklabels(st["OBP"], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("residual", fontsize=8, color=C_MUTE)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(labelsize=7)
    _save(pdf, fig)


# -- Tool 2 -----------------------------------------------------------------------
def page_t2_panel(pdf, run, label, D):
    fig = _fig()
    _title(fig, f"Tool 2 · Discrimination (healthy vs diseased) · {label}",
           f"{len(run['comps'])} compounds, mean ± SD (n = {run['n']} per group)  ·  {run['n_candidates']} candidate "
           f"OBPs  ·  J = Fisher ratio, err = plants misclassified = Φ(−√(J/2))")
    rows = []
    for r in run["singles"].head(8).itertuples():
        rows.append([r.OBP, f"{r.J:.2f}", _err(r.J), f"{r.n_measured}/{len(run['comps'])}",
                     f"{r.reading_A:.1f} ± {r.sd_A:.1f}", f"{r.reading_B:.1f} ± {r.sd_B:.1f}", r.binds])
    fig.text(0.04, 0.89, "Best single OBPs", fontsize=10.5, fontweight="bold", color=C_LILAC_DK)
    _table(fig, [0.04, 0.50, 0.55, 0.37], ["OBP", "J", "err", "Ki meas.", "reading healthy", "reading diseased",
                                          "binds (Ki uM)"], rows, [0.15, 0.07, 0.08, 0.08, 0.15, 0.15, 0.32],
           wrap={6: 38}, fontsize=6.9, row_h=1.25)

    # panel J build-up
    st = run["steps"]
    thr = run["thr"][0]
    _bubble(fig, 0.62, 0.50, 0.35, 0.39, pad=0.02)
    ax = fig.add_axes([0.67, 0.58, 0.28, 0.25]); ax.set_facecolor("none")
    ax.bar(range(len(st)), st["J_panel"], color=C_LILAC, edgecolor="white")
    ax.axhline(thr, color=C_DANGER, ls="--", lw=1.3)
    ax.text(-0.45, thr, f"limit J {thr:.2f}", color=C_DANGER, fontsize=7.5, va="bottom", ha="left")
    for i, v in enumerate(st["J_panel"]):
        ax.text(i, v, _err(v), ha="center", va="bottom", fontsize=6.5, color=C_INK)
    ax.set_xticks(range(len(st))); ax.set_xticklabels(st["OBP"], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("J of the panel", fontsize=8, color=C_MUTE)
    ax.set_title("Panel J (label = % misclassified)", fontsize=9.5, color=C_LILAC_DK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=7)

    # Ki known per panel OBP
    kn = run["known"]
    _bubble(fig, 0.04, 0.07, 0.40, 0.37, pad=0.02)
    ax = fig.add_axes([0.13, 0.12, 0.29, 0.26]); ax.set_facecolor("none")
    y = np.arange(len(kn))
    ax.barh(y, kn["measured"], color=C_GREEN, label="measured")
    ax.barh(y, kn["predicted_used"], left=kn["measured"], color=C_LILAC, label="predicted (used)")
    ax.barh(y, kn["missing"], left=kn["measured"] + kn["predicted_used"], color=C_DANGER_LT, label="missing")
    ax.set_yticks(y); ax.set_yticklabels(kn["OBP"], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel(f"compounds (of {len(run['comps'])})", fontsize=7.5, color=C_MUTE)
    ax.set_title("Ki known per panel OBP", fontsize=9.5, color=C_LILAC_DK)
    ax.legend(fontsize=6.5, loc="lower right")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=7)

    fg, bg = GRADE_COLORS.get(run["grade"], GRADE_COLORS["N/A"])
    _bubble(fig, 0.47, 0.07, 0.50, 0.37, facecolor=bg, edgecolor=fg, alpha=0.6)
    fig.text(0.49, 0.41, f"VERDICT: {run['grade']}", fontsize=15, fontweight="bold", color=fg, va="top")
    lines = [f"Panel: {' + '.join(run['panel'])}",
             f"J = {run['J0']:.2f} ({_err(run['J0'])} misclassified); limit J >= {thr:.2f} ({run['thr'][1]}).",
             f"Why: {run['why']}.",
             f"All Tanimoto predictions at once: J = {run['J_all']:.2f} ({_err(run['J_all'])}) -> "
             + ("within the limit." if run["J_all"] >= thr else "BEYOND the limit."),
             "Compound shift (diseased − healthy, SDs): "
             + ", ".join(f"{D.short(c)} {run['shift'][c]:+.1f}" for c in run["comps"] if pd.notna(run["shift"][c]))]
    fig.text(0.49, 0.355, "\n".join("\n".join(textwrap.wrap(l, 82)) for l in lines), fontsize=8.3, color=C_INK,
             va="top", linespacing=1.55)
    _save(pdf, fig)


def page_t2_bets(pdf, run, label):
    b = run["bets"]
    if b is None or b.empty:
        return
    bad = b[b["verdict"] != "safe"]
    near = bad[(bad["verdict"] != "risk") | (bad["off_by"] <= 10)]
    fig = _fig()
    _title(fig, f"Tool 2 · What the panel bets on · {label}",
           "Each unmeasured Ki moved over 0.1 uM - 10 mM (one at a time, and each compound on all panel OBPs at once)")
    rows, colors = [], {}
    for i, r in enumerate(near.head(10).itertuples()):
        risk = "AGAINST prediction" if r.verdict == "BREAKS (prediction)" else \
            (r.verdict if r.verdict != "risk" else f"lost if off by x{r.off_by:.3g}")
        rows.append([str(i + 1), r.bet, f"{r.J_now:.2f} -> {r.J_min:.2f}",
                     f"{_err(r.J_now)} -> {_err(r.J_min)}", f"{r.compound_role} ({r.shift_SD:+.1f} SD)", risk])
        if r.verdict.startswith("BREAKS"):
            colors[(i, 5)] = C_DANGER
    fig.text(0.04, 0.89, f"Bets that can pull J under the limit: {len(near)}", fontsize=10.5, fontweight="bold",
             color=C_LILAC_DK)
    if not rows:
        _bubble(fig, 0.04, 0.62, 0.92, 0.22, facecolor=C_GREEN_LT, edgecolor=C_GREEN_DK, alpha=0.6)
        fig.text(0.5, 0.73, "No unmeasured Ki can pull J under the limit on its own, nor any compound on all panel "
                 "OBPs at once.\nThe table below shows how far J could fall in the worst case.", ha="center",
                 va="center", fontsize=10, color=C_GREEN_DK, linespacing=1.7)
    else:
        _table(fig, [0.04, 0.42, 0.92, 0.45], ["#", "for the panel to work, this must be true", "J now -> worst",
                                          "misclassified", "compound", "risk"], rows,
               [0.03, 0.47, 0.11, 0.13, 0.13, 0.13], wrap={1: 84, 5: 18}, fontsize=7, color_cells=colors,
               row_h=1.25)
    # all scanned cells, worst J
    worst = b.sort_values("J_min").head(10)
    rows = [[r.OBP[:40], r.compound, r.status, _f(r.Ki_predicted_uM), f"{r.J_min:.2f}", _err(r.J_min),
             r.verdict] for r in worst.itertuples()]
    fig.text(0.04, 0.37, "Largest possible drops of J (all scanned Ki, worst first)", fontsize=10.5,
             fontweight="bold", color=C_LILAC_DK)
    _table(fig, [0.04, 0.06, 0.92, 0.29], ["OBP", "compound", "status", "Ki pred (uM)", "worst J",
                                          "misclassified", "verdict"], rows,
           [0.27, 0.2, 0.09, 0.1, 0.08, 0.1, 0.16], fontsize=7.2, row_h=1.2)
    _save(pdf, fig)


# -- lab priorities and method ------------------------------------------------------
def page_lab_priorities(pdf, runs1, runs2):
    items = []
    for key, run in runs1.items():
        if isinstance(run, dict) and run["bets"] is not None:
            for r in run["bets"].itertuples():
                if r.kind == "unmeasured" and (r.against or r.off_by <= 2):
                    items.append((0 if r.against else 1, r.off_by, "Tool 1 " + SHORT[key], r.OBP, r.compound,
                                  "AGAINST prediction" if r.against else f"off by x{r.off_by:.3g}"))
                elif r.kind == "fragile value" and r.off_by <= 1.1:
                    items.append((2, r.off_by, "Tool 1 " + SHORT[key], r.OBP, r.compound, "repeat (fragile value)"))
    for key, run in runs2.items():
        if isinstance(run, dict):
            for r in run["bets"].itertuples():
                if r.verdict != "safe" and (r.verdict != "risk" or r.off_by <= 2) and r.status != "joint":
                    items.append((0 if r.verdict == "BREAKS (prediction)" else 1, r.off_by, "Tool 2 " + SHORT[key],
                                  r.OBP, r.compound, r.verdict))
    df = pd.DataFrame(items, columns=["p", "f", "run", "OBP", "compound", "why"])
    fig = _fig()
    _title(fig, "Lab priorities", "The measurements that decide whether the panels can be trusted (all runs, "
                                  "most urgent first; same pair listed once)")
    if df.empty:
        _bubble(fig, 0.12, 0.40, 0.76, 0.2, facecolor=C_GREEN_LT)
        fig.text(0.5, 0.5, "No urgent measurement: no run depends on an unverified value that could plausibly "
                           "break it.", ha="center", fontsize=11, color=C_GREEN_DK)
        _save(pdf, fig)
        return
    df = df.sort_values(["p", "f"])
    g = df.groupby(["OBP", "compound"], sort=False).agg(runs=("run", lambda x: ", ".join(dict.fromkeys(x))),
                                                         why=("why", "first"), p=("p", "min")).reset_index()
    g = g.sort_values("p")
    rows, colors = [], {}
    for i, r in enumerate(g.head(22).itertuples()):
        action = "repeat the measurement (replicates)" if "repeat" in r.why else "measure Ki"
        rows.append([str(i + 1), r.OBP, r.compound, action, r.why, r.runs])
        if r.p == 0:
            colors[(i, 4)] = C_DANGER
    _table(fig, [0.04, 0.08, 0.92, 0.80], ["#", "OBP", "compound", "action", "why", "affects"], rows,
           [0.04, 0.2, 0.22, 0.2, 0.16, 0.18], fontsize=7.6, color_cells=colors, row_h=1.3)
    if len(g) > 22:
        fig.text(0.04, 0.05, f"+{len(g) - 22} more in the bets CSV files.", fontsize=8, color=C_MUTE)
    _save(pdf, fig)


METHOD_T1 = """TOOL 1 - SPECIFICITY
1. Ki -> signal: s = 1/Ki (1/uM). Missing Ki = 0 ('does not bind'). '>X' = 1/(1.1 X) (an interferent is assumed to
   bind a little: conservative). Predicted mode: target from *_imputed_pred, interferents from *_imputed_lower.
2. Residual of a panel P: min over concentrations c >= 0 of || A c - b ||  (NNLS). b = target signals on P,
   A = interferent signals on P. It is the distance between the target and the best possible fake mixture.
   The panel works when residual > sensor noise (default 0.05 ~ 5 points of occupancy at target 1 uM).
3. Candidates: OBPs with target data, c2 >= 0.5 (c2 = interferent knowledge: measured 1, predicted x confidence,
   missing 0, / number of interferents), and not blind (only '>X' values for every compound).
4. Search: starts from the OBP you give. Mode 1 (experimental) adds, among the OBPs that raise the residual by
   >= 0.001, the one with the highest gain x c2^2 (best-measured useful OBP); mode 2 adds the one with the
   highest residual. Coverage: every interferent must bind (measured, not '>X', or predicted) a panel OBP.
5. Bets: each unmeasured pair is set to every Ki from 0.1 uM to 10 mM; the Ki range that brings the residual under
   the noise is reported and compared with the Tanimoto prediction. Measured Ki that break the panel with a
   change <= 25 % are listed as fragile values."""
METHOD_T2 = """TOOL 2 - DISCRIMINATION
1. Ki -> signal ('anchored', as main.py): s = log10(40/Ki) / log10(40), clipped to 0-1 (Ki <= 1 uM -> 1,
   Ki >= 40 uM -> 0). Missing Ki and '>X' = 0.
2. Input: mean and SE per compound and class; SD = SE x sqrt(n). Reading of an OBP = sum(s x amount).
   Mean reading mu = A @ mean; covariance Sig = A diag(SD^2) A^T (compounds vary independently).
3. J = d^T (Sig_healthy + Sig_diseased + N)^-1 d, d = mu_diseased - mu_healthy, N = (5 % of the reading)^2.
   One OBP: J = d^2 / (var_h + var_d + noise). Misclassified = Phi(-sqrt(J/2)): 5 % -> J >= 5.4.
4. Bets: every unmeasured Ki of the panel (and each compound on all panel OBPs at once) is moved over 0.1 uM -
   10 mM. A hidden compound that varies between plants more than it separates the classes lowers J: a missing Ki
   can make J look too HIGH. A multi-OBP panel can subtract shared noise, so it resists hidden Ki better.
VERDICTS (both tools): SOLID = no single unverified value can plausibly break it; CONDITIONAL = it rests on
unverified values but the predictions do not contradict them; WEAK = a bet goes against the prediction, or the
panel fails if every prediction is true at once. Full explanation: docs/MISSING_DATA_ES.md."""


def page_method(pdf):
    fig = _fig()
    _title(fig, "Method · how every number is computed and how missing Ki are handled")
    _bubble(fig, 0.04, 0.50, 0.92, 0.40, pad=0.02)
    fig.text(0.06, 0.88, METHOD_T1, fontsize=7.9, family="monospace", color=C_INK, va="top", linespacing=1.45)
    _bubble(fig, 0.04, 0.06, 0.92, 0.41, pad=0.02)
    fig.text(0.06, 0.45, METHOD_T2, fontsize=7.9, family="monospace", color=C_INK, va="top", linespacing=1.45)
    _save(pdf, fig)


# -- options ----------------------------------------------------------------------------
def tool1_opts(source, a):
    return SimpleNamespace(min_confidence=a.min_confidence if source == "predicted" else 0.0, only_confident=False,
                           min_c2=a.min_c2, priority="auto", noise=a.noise, min_gain=1e-3, max_sensors=8,
                           lookahead=3, cover=1, mc=0)


def tool2_opts(source, a):
    return SimpleNamespace(min_confidence=a.min_confidence if source == "predicted" else 0.0, only_confident=False,
                           censored="zero", scale="anchored", ki_min=1.0, ki_max=40.0, sensor_noise=a.sensor_noise,
                           n=a.n, min_gain=None, target_score=None, max_sensors=8, metric="fisher",
                           max_error=a.max_error, weak_frac=None)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="folder with the SCENTINEL CSVs (default: ./data)")
    p.add_argument("--target", default=None, help="target VOC, name or CAS")
    p.add_argument("--interferents", default=None, help="interferent .txt file")
    p.add_argument("--start-with", nargs="+", default=None, metavar="OBP", help="starting OBP(s) for Tool 1")
    p.add_argument("--profiles", default=None,
                   help="vineyard input table for Tool 2 (CAS | name | mean healthy | SE healthy | mean diseased | "
                        "SE diseased). Leave out to skip Tool 2")
    p.add_argument("--min-confidence", type=float, default=0.3, help="predicted mode: minimum Tanimoto confidence "
                                                                     "(default 0.3)")
    p.add_argument("--min-c2", type=float, default=0.5, help="Tool 1: minimum c2 of a candidate OBP (default 0.5)")
    p.add_argument("--noise", type=float, default=0.05, help="Tool 1 sensor noise floor (default 0.05)")
    p.add_argument("--max-error", type=float, default=0.05, help="Tool 2: max share of plants misclassified "
                                                                 "(default 0.05)")
    p.add_argument("--sensor-noise", type=float, default=0.05, help="Tool 2 sensor noise, fraction of the reading")
    p.add_argument("--n", type=int, default=3, help="Tool 2: plants per group behind each SE (default 3)")
    p.add_argument("--out", default=".", help="folder for the PDF (default: current folder)")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    a = parse_args()
    print("SCENTINEL - Panel report (PDF) | iGEM URV 2026\n")
    D = ScentinelData.load(t1.pick_data_dir(a.data_dir))

    target_q = a.target
    while True:
        target_q = target_q or input("Name or CAS of the target VOC: ").strip()
        try:
            target = D.resolve(target_q)
            break
        except CompoundAmbiguous as e:
            if a.target:
                sys.exit(f"  x {e}")
            target = t1.choose_compound_match(target_q, e.matches, e.display_matches)
            break
        except CompoundNotFound as e:
            print(f"  x {e}")
            if a.target:
                sys.exit(1)
            target_q = None
    cas = D.meta.set_index("label").loc[target, "CAS"]

    path = a.interferents or input("Interferent file: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: interferent file '{path}' not found.")
    try:
        interferents, _ = D.resolve_many(read_interferent_file(path), strict=True)
    except CompoundNotFound as e:
        sys.exit("  x Some interferents are not in the database:\n    " + str(e))
    interferents = [i for i in interferents if i != target]

    queries = a.start_with or input("Starting OBP(s) for Tool 1, space-separated: ").split()
    start = []
    for q in queries:
        o = t1.resolve_obp(q, D.obps)
        if o not in start:
            start.append(o)

    profiles = a.profiles
    if profiles is None and a.target is None:
        profiles = input("Vineyard input table for Tool 2 (leave empty to skip): ").strip().strip('"').strip("'") \
            or None
    if profiles and not (os.path.isfile(profiles) and t2.is_input_table(profiles)):
        print(f"  ! '{profiles}' is not an input table (CAS | name | mean healthy | ...): Tool 2 skipped.")
        profiles = None

    runs1, runs2 = {}, {}
    for source in ("experimental", "predicted"):
        print(f"  Tool 1 - {source} ...")
        try:
            run = t1.analyse_tool1(D, target, interferents, source, start, tool1_opts(source, a))
            run["target_short"] = D.short(target)
            runs1[source] = run
        except t1.NotEnoughCandidates as e:
            runs1[source] = str(e)
    if profiles:
        A, B, _ = D.load_input(profiles, n=a.n)
        for source in ("experimental", "predicted"):
            print(f"  Tool 2 - {source} ...")
            try:
                runs2[source] = t2.analyse_tool2_summary(D, source, A, B, tool2_opts(source, a))
            except RuntimeError as e:
                runs2[source] = str(e)

    now = datetime.datetime.now()
    safe = re.sub(r"[^\w]+", "_", D.short(target))[:40]
    os.makedirs(a.out, exist_ok=True)
    pdf_path = os.path.join(a.out, f"SCENTINEL_panel_report_{safe}_{now:%Y-%m-%d_%H-%M-%S}.pdf")
    ctx = dict(target_short=D.short(target), cas=cas, start=start, interf_short=[D.short(i) for i in interferents],
               profiles_name=os.path.basename(profiles) if profiles else None, noise=a.noise,
               min_conf=a.min_confidence, max_error=a.max_error, sensor_noise=a.sensor_noise)
    labels = {"experimental": "Experimental data", "predicted": f"Predicted data (conf >= {a.min_confidence})"}
    with PdfPages(pdf_path) as pdf:
        page_cover(pdf, ctx, runs1, runs2, now.strftime("%d/%m/%Y %H:%M"))
        for source in ("experimental", "predicted"):
            run = runs1[source]
            if isinstance(run, str):
                _page_unavailable(pdf, f"Tool 1 · {labels[source]} · not available", run)
                continue
            page_t1_panel(pdf, run, labels[source], D)
            if run["bets"] is not None:
                page_t1_bets(pdf, run, labels[source])
        page_t1_compare(pdf, runs1)
        for source in ("experimental", "predicted"):
            run = runs2.get(source)
            if run is None:
                continue
            if isinstance(run, str):
                _page_unavailable(pdf, f"Tool 2 · {labels[source]} · not available", run)
                continue
            page_t2_panel(pdf, run, labels[source], D)
            page_t2_bets(pdf, run, labels[source])
        page_lab_priorities(pdf, runs1, runs2)
        page_method(pdf)
        info = pdf.infodict()
        info["Title"] = f"SCENTINEL panel report - {D.short(target)}"
        info["Author"] = "SCENTINEL - iGEM URV 2026"
    print(f"\nOK Report saved: {pdf_path}")


if __name__ == "__main__":
    main()
