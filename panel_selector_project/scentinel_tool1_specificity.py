"""
SCENTINEL - Tool 1 (specificity) on the real SCENTINEL CSVs.

Smallest OBP panel such that no non-negative mixture of interferents can
imitate the target VOC's response (NNLS residual > sensor noise).

Uses the SAME files as main.py / generate_report.py:
    Compound_OBP_binding.csv                (experimental)
    Compound_OBP_binding_imputed_pred.csv   (predicted: target)
    Compound_OBP_binding_imputed_lower.csv  (predicted: interferents, conservative)
    tanimoto_imputation_diagnostic.csv      (confidence of each imputed cell)
    interferents_*.txt                      (names or CAS, one per line)

The starting OBP (--start-with) is required: the search starts from it (e.g.
the protein chosen by SCENTINEL v3) and only looks for complementary OBPs.

Examples
    python scentinel_tool1_specificity.py --data-dir data --target "methyl salicylate" \
        --interferents data/interferents_example.txt --source experimental --start-with DcitOBP3
    python scentinel_tool1_specificity.py --data-dir data --target 119-36-8 \
        --interferents data/interferents_matching.txt --source predicted --min-confidence 0.3 \
        --start-with DcitOBP3
Leave --target / --interferents / --source / --start-with out to be asked interactively.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import os
import re
import sys

import numpy as np
import pandas as pd

from panel_selector import (
    ScentinelData,
    CompoundNotFound,
    CompoundAmbiguous,
    read_interferent_file,
    greedy_panel_for_specificity,
    best_imitation_recipe,
    monte_carlo_collision_probability,
    residual_with_cell,
    breaking_ki_range,
    measured_tolerance,
    complete_coverage,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STOP_TEXT = {
    "target_crossed": "OK - the residual clears the noise floor: no mixture of these interferents "
                      "can imitate the target on this panel.",
    "diminishing_returns": "Stopped: the next OBP barely raises the residual (redundant with the panel). "
                           "The noise floor was NOT crossed.",
    "library_exhausted": "Library exhausted: even with every candidate OBP the target can still be "
                         "imitated within the noise. These proteins are not enough.",
    "max_sensors_reached": "Stopped at --max-sensors without crossing the noise floor.",
}


def ask(prompt, default=None):
    raw = input(prompt).strip()
    return raw if raw else default


def choose_compound_match(query, matches, display=None):
    """Interactive selection for ambiguous compound names; returns the chosen label."""
    opts = list(display) if display is not None else matches
    print(f"  x '{query}' is ambiguous ({len(matches)} matches):")
    for i, name in enumerate(opts, 1):
        print(f"    [{i}] {name}")
    while True:
        raw = ask("  Select a number [1..N]: ")
        if raw and raw.isdigit() and 1 <= int(raw) <= len(matches):
            return matches[int(raw) - 1]
        print(f"  x Please choose a number from 1 to {len(matches)}.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="folder with the SCENTINEL CSVs (default: ./data, ./ or script dir)")
    p.add_argument("--diag", default=None, help="Tanimoto diagnostic CSV (default: auto-detect)")
    p.add_argument("--source", choices=["experimental", "predicted"], default=None)
    p.add_argument("--target", default=None, help="target VOC, name or CAS")
    p.add_argument("--interferents", default=None, help="interferent .txt file")
    p.add_argument("--noise", type=float, default=0.05,
                   help="sensor noise floor on the residual, in 1/uM units at target conc 1 (default 0.05)")
    p.add_argument("--min-gain", type=float, default=1e-3, help="diminishing-returns threshold (default 1e-3)")
    p.add_argument("--max-sensors", type=int, default=8, help="hard cap on panel size (default 8, 0 = none)")
    p.add_argument("--min-c2", "--min-coverage", dest="min_c2", type=float, default=0.5,
                   help="only OBPs with c2 >= this (0-1, default 0.5). c2 = interferent knowledge, as in "
                        "confidence.py: each interferent gives 1 if measured, its Tanimoto confidence if "
                        "imputed, 0 if missing; c2 = sum / number of interferents. Missing interferent data "
                        "counts as 'does not bind', which makes an OBP look falsely specific.")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="predicted mode: drop imputed cells with Tanimoto confidence c1 below this (0-1)")
    p.add_argument("--only-confident", action="store_true",
                   help="predicted mode: keep only imputed cells with decision == 'confident'")
    p.add_argument("--lookahead", type=int, default=3,
                   help="when the residual is stuck at 0, try adding up to this many OBPs at once (default 3, 1 = plain greedy)")
    p.add_argument("--start-with", nargs="+", default=None, metavar="OBP",
                   help="REQUIRED: OBP(s) put into the panel first (e.g. the best protein chosen by SCENTINEL v3); "
                        "the search then only looks for complementary OBPs. If left out, the script asks for it")
    p.add_argument("--mc", type=int, default=0, help="Monte Carlo draws for P(collision) on the final panel (0 = skip)")
    p.add_argument("--priority", choices=["auto", "score", "measured"], default="auto",
                   help="how the next OBP is chosen. score: the one that raises the residual most. "
                        "measured: among the OBPs that raise it by >= --min-gain, the best-measured one "
                        "(gain x c2^2). auto (default): measured in experimental mode, score in predicted mode")
    p.add_argument("--cover", type=int, default=1,
                   help="every interferent must BIND (measured, not '>X', or predicted) at least this many panel OBPs; "
                        "OBPs are added after the search until it does (default 1, 0 = off)")
    p.add_argument("--show-bets", type=int, default=10,
                   help="how many bets (assumptions the panel relies on) to print, default 10")
    p.add_argument("--verbose", action="store_true",
                   help="print every section (lab worklist by OBP, weak points, legends) and save the weak-points CSV")
    p.add_argument("--show-weak", type=int, default=25,
                   help="weak points: how many breaking pairs to print (all go to the CSV, default 25)")
    p.add_argument("--out", default="results", help="folder for the CSV output (default ./results)")
    return p.parse_args()


def resolve_obp(query, obps):
    """OBP name -> exact OBP label: exact match, then case-insensitive, then a unique partial match."""
    if query in obps:
        return query
    low = {o.lower(): o for o in obps}
    if query.lower() in low:
        return low[query.lower()]
    partial = [o for o in obps if query.lower() in o.lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise LookupError(f"'{query}' matches several OBPs: " + ", ".join(partial[:10]))
    close = difflib.get_close_matches(query, obps, n=5, cutoff=0.5)
    raise LookupError(f"OBP '{query}' not found." + (f" Did you mean: {', '.join(close)}?" if close else ""))


def selectivity(S, target, interferents):
    """Per OBP: target signal / signal of the strongest interferent on that OBP.
    > 1 = the OBP sees the target more than any single interferent.
    Missing interferent data counts as 0 (same assumption as the residual);
    no interferent signal at all -> inf (nothing on the list binds it)."""
    tgt = S.S[target].fillna(0.0)
    worst = S.S[interferents].fillna(0.0).max(axis=1) if interferents else pd.Series(0.0, index=S.obps)
    with np.errstate(divide="ignore", invalid="ignore"):
        sel = tgt / worst
    sel[(worst == 0) & (tgt > 0)] = np.inf
    return sel.fillna(0.0)


def binds_matrix(D, S):
    """OBP x compound: True where the OBP really responds to the compound in the
    data being used: a value exists and it is not a measured '>X' (no binding
    up to X uM). '>X' cells still enter the residual as 1/(1.1 X) (conservative
    for interferents), but they are not evidence that the OBP SEES the compound."""
    has = S.S.notna()
    if D.censored is None:
        return has
    cens = (D.censored & D.decision.eq("measured")).T.reindex(index=has.index, columns=has.columns,
                                                              fill_value=False)
    return has & ~cens


def interferent_c2(D, S, interferents):
    """c2 per OBP, same rule as confidence.py: credit per interferent =
    1 measured, c1 (Tanimoto confidence, 0.5 if no diagnostic row) imputed,
    0 missing or dropped; c2 = total credit / number of interferents.
    In experimental mode it equals the fraction of interferents measured."""
    if not interferents:
        return pd.Series(1.0, index=S.obps)
    used = S.S[list(interferents)].notna()                              # OBP x interferent
    credit = D.confidence.loc[list(interferents), S.obps].T.fillna(0.0)
    return credit.where(used, 0.0).sum(axis=1) / len(interferents)


WORK_STATUS_ORDER ={"no data": 0, "not used": 1, "imputed": 2}


def lab_worklist(D, S, panel, target, interferents):
    """Every (panel OBP, compound) pair that was NOT measured in the lab, i.e.
    what still has to be measured before the panel is trusted:
      no data  -> no Ki at all: the residual assumed it does NOT bind (most urgent)
      not used -> a Tanimoto prediction exists but was not used (experimental
                  mode, or dropped by --min-confidence / --only-confident)
      imputed  -> the residual used a Tanimoto prediction (confirm it)
    Ordered by panel OBP (one protein = one lab series), then status, then
    lowest confidence first."""
    rows = []
    for rank, o in enumerate(panel):
        for c in [target] + list(interferents):
            if D.decision.loc[c, o] == "measured":
                continue
            s = S.S.loc[o, c]
            predicted_exists = pd.notna(D.decision.loc[c, o])
            if pd.notna(s) and s > 0:
                status = "imputed"
            else:
                status = "not used" if predicted_exists else "no data"
            conf = D.confidence.loc[c, o]
            rows.append({"panel_rank": rank + 1, "OBP": o, "compound": D.short(c),
                         "role": "target" if c == target else "interferent",
                         "status": status,
                         "Ki_used_uM": 1.0 / s if status == "imputed" else np.nan,
                         "confidence": conf if predicted_exists else np.nan,
                         "tanimoto_decision": D.decision.loc[c, o] if predicted_exists else "-"})
    df = pd.DataFrame(rows, columns=["panel_rank", "OBP", "compound", "role", "status", "Ki_used_uM",
                                     "confidence", "tanimoto_decision"])
    if df.empty:
        return df
    df["_s"] = df["status"].map(WORK_STATUS_ORDER)
    df = df.sort_values(["panel_rank", "_s", "confidence"], na_position="first").drop(columns="_s")
    return df.reset_index(drop=True)


def show_lab_worklist(wl, n_compounds):
    sep = "-" * 104
    print(f"\n{sep}\n  LAB WORKLIST - pairs on the final panel that were NOT measured (measure these)\n"
          f"  Ordered by panel OBP, then: no data (assumed non-binding) > not used > imputed (low confidence first)\n{sep}")
    if wl.empty:
        print("  Nothing to measure: every target/interferent cell on the panel is measured.")
        print(sep)
        return
    print(f"  {'#':>3}  {'OBP':<20} {'compound':<30} {'role':<11} {'status':<9} {'Ki_used':>8} "
          f"{'conf':>5}  {'Tanimoto':<11}")
    print(sep)
    for i, r in enumerate(wl.itertuples(), 1):
        ki = f"{r.Ki_used_uM:.2f}" if pd.notna(r.Ki_used_uM) else "-"
        cf = f"{r.confidence:.2f}" if pd.notna(r.confidence) else "-"
        print(f"  {i:>3}. {r.OBP:<20} {r.compound[:30]:<30} {r.role:<11} {r.status:<9} {ki:>8} "
              f"{cf:>5}  {r.tanimoto_decision:<11}")
    print(sep)
    for o, g in wl.groupby("OBP", sort=False):
        counts = g["status"].value_counts()
        print(f"  {o:<20} {len(g):>3}/{n_compounds} to measure  "
              f"(no data {counts.get('no data', 0)}, not used {counts.get('not used', 0)}, "
              f"imputed {counts.get('imputed', 0)})")
    print(f"  Total pairs to measure: {len(wl)}")


# ----------------------------------------------------------------------------- weak points
WEAK_BINDING_KI = 30.0      # uM: above this the assays usually report "no binding" (>X)
BET_MAX_FOLD = 10.0         # bets that need a guess >10x wrong to be lost are listed only as a count
FRAGILE_CHANGE = 0.25       # a measured Ki is "fragile" if a change of <= 25 % breaks the panel

VERDICT_ORDER = {"BREAKS (prediction)": 0, "BREAKS if it binds": 1, "risk": 2, "safe": 3}


def _predicted_ki(D, compound, obp, is_target):
    """Tanimoto Ki for a cell, from the same table predicted mode would use
    (target: *_imputed_pred, interferent: *_imputed_lower). NaN if none."""
    t = D.ki_pred if is_target else D.ki_lower
    if t is None:
        return np.nan
    v = t.loc[compound, obp]
    return float(v) if pd.notna(v) and v > 0 else np.nan


def _ki_range_text(lo, hi, grid):
    if lo is None:
        return "never"
    if lo <= grid[0] * 1.0001 and hi >= grid[-1] * 0.9999:
        return "any Ki"
    if lo <= grid[0] * 1.0001:
        return f"Ki <= {hi:.3g}"
    if hi >= grid[-1] * 0.9999:
        return f"Ki >= {lo:.3g}"
    return f"{lo:.3g} <= Ki <= {hi:.3g}"


def weak_points(D, S, panel, target, interferents, noise, wl):
    """One row per NOT-measured pair of the final panel (the lab worklist):
    the Ki range that would make the panel fail, the Tanimoto prediction for
    that cell and the residual if the prediction were true, and a verdict:
      BREAKS (prediction) -> the prediction alone already brings the residual under noise
      BREAKS if it binds  -> even weak binding (Ki up to >= 30 uM) breaks it
      risk                -> breaks only if the Ki is in the reported range
      safe                -> no Ki between 0.1 uM and 10 mM breaks it
    fold_to_break: how many times the best guess of the Ki (prediction, or 30 uM
    if none) must be off to reach that range (1 = already inside). Rows are
    sorted by verdict, then by this number (smallest = most urgent).
    Each pair is changed ALONE; several unknowns together can do more harm
    (see all_predictions_residual)."""
    from panel_selector.specificity import DEFAULT_KI_GRID as grid
    full = {D.short(c): c for c in [target] + list(interferents)}
    rows = []
    for r in wl.itertuples():
        c = full[r.compound]
        is_t = c == target
        lo, hi = breaking_ki_range(S, panel, target, interferents, r.OBP, c, noise)
        ki_pred = r.Ki_used_uM if r.status == "imputed" else _predicted_ki(D, c, r.OBP, is_t)
        res_pred = (residual_with_cell(S, panel, target, interferents, r.OBP, c, ki_pred)
                    if pd.notna(ki_pred) else np.nan)
        others = [o for o in panel if o != r.OBP]
        col = S.S.loc[others, c].fillna(0.0) if others else pd.Series(dtype=float)
        invisible = (not is_t) and bool((col <= 0).all())
        if pd.notna(res_pred) and res_pred < noise:
            verdict = "BREAKS (prediction)"
        elif lo is not None and lo <= grid[0] * 1.0001 and hi >= WEAK_BINDING_KI:
            verdict = "BREAKS if it binds"
        elif lo is not None:
            verdict = "risk"
        else:
            verdict = "safe"
        # how far is the best guess of this Ki from the range that breaks the panel?
        # best guess = Tanimoto prediction, or WEAK_BINDING_KI when there is none
        ref = ki_pred if pd.notna(ki_pred) else WEAK_BINDING_KI
        if lo is None:
            fold = np.nan
        elif lo <= ref <= hi:
            fold = 1.0
        else:
            fold = float(min(abs(np.log10(ref / lo)), abs(np.log10(ref / hi))))
            fold = 10 ** fold
        rows.append({"OBP": r.OBP, "compound": r.compound, "role": r.role, "status": r.status,
                     "breaks_if": _ki_range_text(lo, hi, grid), "fold_to_break": fold,
                     "invisible_elsewhere": invisible,
                     "Ki_break_low_uM": lo, "Ki_break_high_uM": hi,
                     "Ki_predicted_uM": ki_pred, "confidence": r.confidence,
                     "residual_if_predicted": res_pred, "verdict": verdict})
    cols = ["OBP", "compound", "role", "status", "breaks_if", "fold_to_break", "invisible_elsewhere", "Ki_break_low_uM", "Ki_break_high_uM",
            "Ki_predicted_uM", "confidence", "residual_if_predicted", "verdict"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["_v"] = df["verdict"].map(VERDICT_ORDER)
        df = df.sort_values(["_v", "fold_to_break"]).drop(columns="_v")
    return df.reset_index(drop=True)


def all_predictions_residual(D, S, panel, target, interferents):
    """Residual if EVERY not-measured cell of the panel that has a Tanimoto
    prediction took that value at the same time (cells with no prediction stay 0)."""
    A = np.nan_to_num(S.S.loc[panel, interferents].values.astype(float)).copy()
    b = np.nan_to_num(S.S.loc[panel, target].values.astype(float)).copy()
    for i, o in enumerate(panel):
        if pd.isna(S.S.loc[o, target]) or D.decision.loc[target, o] != "measured":
            k = _predicted_ki(D, target, o, True)
            if pd.isna(S.S.loc[o, target]) and pd.notna(k):
                b[i] = 1.0 / k
        for j, c in enumerate(interferents):
            if pd.isna(S.S.loc[o, c]):
                k = _predicted_ki(D, c, o, False)
                if pd.notna(k):
                    A[i, j] = 1.0 / k
    from scipy.optimize import nnls
    return float(nnls(A, b)[1]) if A.shape[1] else float(np.linalg.norm(b))


def measured_fragility(D, S, panel, target, interferents, noise):
    """Cells of the panel that DO have a value (measured, or imputed and used):
    how much their Ki may change before the panel fails."""
    rows = []
    for o in panel:
        for c in [target] + list(interferents):
            s = S.S.loc[o, c]
            if pd.isna(s) or s <= 0:
                continue
            down, up = measured_tolerance(S, panel, target, interferents, o, c, noise)
            cens = bool(D.censored is not None and D.censored.loc[c, o]) and \
                D.decision.loc[c, o] == "measured"
            if cens and down is not None and (1.0 + down) < 1.0 / 1.1:
                # '>X' stored as 1.1 X: the true Ki is above X, it cannot drop below it
                down = None
            rows.append({"OBP": o, "compound": D.short(c), "role": "target" if c == target else "interferent",
                         "measured": D.decision.loc[c, o] == "measured", "censored": cens,
                         "Ki_uM": 1.0 / s, "breaks_if_lower_by": down, "breaks_if_higher_by": up})
    df = pd.DataFrame(rows, columns=["OBP", "compound", "role", "measured", "censored", "Ki_uM",
                                     "breaks_if_lower_by", "breaks_if_higher_by"])
    if df.empty:
        return df
    for col in ("breaks_if_lower_by", "breaks_if_higher_by"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["min_change"] = df[["breaks_if_lower_by", "breaks_if_higher_by"]].abs().min(axis=1)
    return df.sort_values("min_change", na_position="last").reset_index(drop=True)


def show_weak_points(D, S, panel, target, interferents, noise, wp, frag, resid, show_max=25):
    sep = "-" * 104
    print(f"\n{sep}\n  WEAK POINTS - what could still break this panel "
          f"(residual {resid:.4f}, noise {noise}, margin {resid - noise:+.4f} "
          f"= {100 * (resid - noise) / noise:+.0f} %)\n{sep}")

    # 1) per interferent: how well is it known on the panel, and its worst verdict
    print("  Per interferent (measured = OBPs of the panel where its Ki was measured in the lab):")
    worst_by = {}
    for r in wp.itertuples():          # wp is sorted most urgent first
        if r.verdict != "safe" and r.compound not in worst_by:
            worst_by[r.compound] = (r.verdict, r.OBP, r.breaks_if, r.fold_to_break)
    lines = []
    for c in [target] + list(interferents):
        n_meas = int(sum(D.decision.loc[c, o] == "measured" for o in panel))
        v, o, txt, fold = worst_by.get(D.short(c), ("ok", "", "", np.nan))
        lines.append((VERDICT_ORDER.get(v, 4), fold if pd.notna(fold) else 0, n_meas, D.short(c),
                      c == target, v, o, txt))
    ok = []
    for order, fold, n_meas, name, is_t, v, o, txt in sorted(lines, key=lambda x: (x[0], x[1], x[2])):
        if not o:
            ok.append(name + (" (target)" if is_t else ""))
            continue
        tag = " (target)" if is_t else ""
        flag = "!!" if order <= 1 else " !"
        far = f", prediction off x{fold:.2g}" if order == 2 else ""
        print(f"   {flag} {name[:28] + tag:<38} measured {n_meas}/{len(panel)}   {v} on {o} ({txt}{far})")
    if ok:
        print(f"      ok ({len(ok)}, no single missing/imputed cell can break it): " + ", ".join(ok))

    # 2) the pairs themselves
    bad = wp[wp["verdict"] != "safe"]
    print(f"\n  Not-measured pairs that can break it on their own: {len(bad)} of {len(wp)}")
    if len(bad):
        print(f"    {'OBP':<18} {'compound':<26} {'verdict':<20} {'breaks if':<18} "
              f"{'Ki_pred':>8} {'conf':>5} {'resid_pred':>10} {'off by':>7}")
        for r in bad.head(show_max).itertuples():
            kp = f"{r.Ki_predicted_uM:.3g}" if pd.notna(r.Ki_predicted_uM) else "-"
            cf = f"{r.confidence:.2f}" if pd.notna(r.confidence) else "-"
            rp = f"{r.residual_if_predicted:.4f}" if pd.notna(r.residual_if_predicted) else "-"
            bi = r.breaks_if + (" *" if r.invisible_elsewhere else "")
            fd = f"x{r.fold_to_break:.2g}" if pd.notna(r.fold_to_break) else "-"
            print(f"    {r.OBP:<18} {r.compound[:26]:<26} {r.verdict:<20} {bi:<18} "
                  f"{kp:>8} {cf:>5} {rp:>10} {fd:>7}")
        if len(bad) > show_max:
            print(f"    ... {len(bad) - show_max} more in the weak-points CSV (use --show-weak N to print more)")
        print("    off by = how many times the predicted Ki (30 uM if none) must be wrong to break the panel")
        print("    -> measure these first. Ki in uM; 'breaks if' = Ki range that brings the residual under noise.")
        inv = bad[bad["invisible_elsewhere"]]
        if len(inv):
            print("    * " + ", ".join(sorted(set(inv["compound"]))) + ": no signal on any other OBP of the panel, so "
                  "even weak binding imitates the target\n      if its concentration is high enough (the linear model "
                  "has no upper limit). Measure it on the other OBPs too.")
    n_safe = int((wp["verdict"] == "safe").sum())
    if n_safe:
        print(f"    The other {n_safe} not-measured pairs cannot break the panel alone (any Ki 0.1 uM - 10 mM).")

    dropped = bad[(bad["verdict"] == "BREAKS (prediction)") & (bad["status"] == "not used")]
    if len(dropped):
        print(f"    ! {len(dropped)} of the BREAKS pairs have a Tanimoto prediction that was NOT used "
              "(experimental mode, or below --min-confidence):\n      the panel passes only because "
              "those predictions were left out and counted as 'does not bind'.")

    # 3) all predictions at once
    if len(wp):
        r_all = all_predictions_residual(D, S, panel, target, interferents)
        verdict = "still passes" if r_all >= noise else "FAILS"
        print(f"\n  If every not-measured pair took its Tanimoto prediction at once: residual {r_all:.4f} "
              f"-> {verdict}")

    # 4) fragile values that do exist
    if not frag.empty:
        f = frag[frag["min_change"] <= FRAGILE_CHANGE]
        print(f"\n  Values the residual depends on (a change of <= {FRAGILE_CHANGE:.0%} in Ki breaks the panel): "
              f"{len(f)}")
        for r in f.itertuples():
            parts = []
            if pd.notna(r.breaks_if_lower_by):
                parts.append(f"{r.breaks_if_lower_by:+.0%}")
            if pd.notna(r.breaks_if_higher_by):
                parts.append(f"{r.breaks_if_higher_by:+.0%}")
            ki = (f">{r.Ki_uM / 1.1:.3g}" if r.censored else f"{r.Ki_uM:.3g}")
            src = "measured" if r.measured else "imputed"
            print(f"    {r.OBP:<18} {r.compound[:26]:<26} Ki {ki:>7} uM ({src})  breaks at "
                  + " or ".join(parts))
        if len(f):
            print("    -> repeat these measurements (replicates): a normal assay error is ~10-20 %.")
    print("  Each pair is changed alone; occupancy view: residual 0.05 ~ 5 points of occupancy "
          "at target 1 uM.")
    print(sep)


# ----------------------------------------------------------------------------- bets
def _bet_text(compound, obp, lo, hi, grid):
    """What must be true about ONE cell for the panel to keep passing, given the
    Ki range [lo, hi] that would break it."""
    at_min = lo <= grid[0] * 1.0001
    at_max = hi >= grid[-1] * 0.9999
    if at_min and at_max:
        return f"{compound} does NOT bind {obp} at all"
    if at_min:
        return f"{compound} binds {obp} weakly or not at all (Ki > {hi:.3g} uM)"
    if at_max:
        return f"{compound} binds {obp} at least this strongly: Ki < {lo:.3g} uM"
    return f"Ki of {compound} on {obp} is below {lo:.3g} or above {hi:.3g} uM (not in between)"


def panel_bets(wp, frag, panel, interferents, D, target):
    """Turn the weak-point analysis into plain statements: every unverified
    assumption the 'pass' depends on, with the data behind it.

    One row per bet:
      bet       what must be true for the panel to keep passing
      evidence  what the database says about that cell
      against   True if the Tanimoto prediction says the bet is WRONG
      off_by    how many times the best guess must be wrong to lose the bet
    """
    from panel_selector.specificity import DEFAULT_KI_GRID as grid
    rows = []
    for r in wp[wp["verdict"] != "safe"].itertuples():
        bet = _bet_text(r.compound, r.OBP, r.Ki_break_low_uM, r.Ki_break_high_uM, grid)
        if r.status == "no data":
            ev = "no measurement, no prediction (counted as 'does not bind')"
        elif r.status == "not used":
            ev = (f"not measured; prediction Ki {r.Ki_predicted_uM:.3g} uM, conf {r.confidence:.2f}, "
                  f"left out (counted as 'does not bind')")
        else:
            ev = f"not measured; prediction Ki {r.Ki_predicted_uM:.3g} uM, conf {r.confidence:.2f}, used"
        if r.invisible_elsewhere:
            ev += "; no signal on the other panel OBPs, so any binding can be scaled up to imitate"
        rows.append({"bet": bet, "evidence": ev, "kind": "unmeasured",
                     "against": r.verdict == "BREAKS (prediction)",
                     "off_by": 1.0 if r.verdict == "BREAKS (prediction)" else r.fold_to_break,
                     "OBP": r.OBP, "compound": r.compound, "verdict": r.verdict})
    if frag is not None and not frag.empty:
        for r in frag[frag["min_change"] <= FRAGILE_CHANGE].itertuples():
            parts = []
            if pd.notna(r.breaks_if_lower_by):
                parts.append(f"{abs(r.breaks_if_lower_by):.0%} lower")
            if pd.notna(r.breaks_if_higher_by):
                parts.append(f"{r.breaks_if_higher_by:.0%} higher")
            ki = f">{r.Ki_uM / 1.1:.3g}" if r.censored else f"{r.Ki_uM:.3g}"
            src = "measured" if r.measured else "predicted (used)"
            rows.append({"bet": f"Ki of {r.compound} on {r.OBP} = {ki} uM is accurate "
                                f"(fails if it is {' or '.join(parts)})",
                         "evidence": f"{src}, single value", "kind": "fragile value",
                         "against": False, "off_by": 1.0 + float(r.min_change),
                         "OBP": r.OBP, "compound": r.compound, "verdict": "fragile"})
    df = pd.DataFrame(rows, columns=["bet", "evidence", "kind", "against", "off_by", "OBP", "compound",
                                     "verdict"])
    if not df.empty:
        df = df.sort_values(["against", "off_by"], ascending=[False, True]).reset_index(drop=True)
    return df


def show_panel_bets(bets, panel, noise, resid, r_all, show_max=10):
    sep = "=" * 104
    far = bets[(~bets["against"]) & (bets["off_by"] > BET_MAX_FOLD)] if len(bets) else bets
    bets = bets.drop(far.index) if len(bets) else bets
    n = len(bets)
    n_against = int(bets["against"].sum()) if n else 0
    n_close = int((bets["off_by"] < 2).sum()) if n else 0
    if n == 0 and r_all >= noise:
        grade, why = "SOLID", "no single unverified or fragile value can plausibly break it"
    elif n_against or r_all < noise:
        grade = "WEAK"
        why = (f"{n_against} bet(s) go AGAINST the Tanimoto prediction"
               if n_against else "it fails if every prediction is true at once")
    elif n_close:
        grade, why = "CONDITIONAL", f"{n_close} bet(s) are lost if a guess is off by less than 2x"
    else:
        grade, why = "CONDITIONAL", "it rests on unverified values, but each has a 2x margin or more"
    print(f"\n{sep}\n  VERDICT: {grade}  -  panel {' + '.join(panel)}\n"
          f"  Passes on the data used (residual {resid:.4f} vs noise {noise}); {why}.\n"
          f"  If every not-measured pair took its Tanimoto prediction at once: residual {r_all:.4f} "
          f"-> {'still passes' if r_all >= noise else 'FAILS'}")
    if n:
        print(f"\n  WHAT THIS PANEL BETS ON ({n} unverified assumptions; most fragile first):")
        for i, r in enumerate(bets.head(show_max).itertuples(), 1):
            tag = "  AGAINST prediction" if r.against else f"  (lost if off by x{r.off_by:.3g})"
            print(f"   {i:>2}. {r.bet}{tag}\n       data: {r.evidence}")
        if n > show_max:
            print(f"   ... {n - show_max} more in the bets CSV (--show-bets N to print more)")
        print("  Measuring the first ones settles whether this panel can be trusted.")
    if len(far):
        print(f"  (+ {len(far)} far-fetched bet(s): lost only if a guess is more than {BET_MAX_FOLD:.0f}x wrong; "
              "listed in the bets CSV)")
    print("  How missing data is handled: missing Ki = 'does not bind' during the search; this section")
    print("  then tests each missing / predicted / fragile value one by one (docs/MISSING_DATA_ES.md).")
    print(sep)


def pick_data_dir(arg):
    if arg:
        return arg
    for d in ("data", ".", os.path.join(SCRIPT_DIR, "data"), SCRIPT_DIR):
        if os.path.isfile(os.path.join(d, "Compound_OBP_binding.csv")):
            return d
    return "."


def main():
    # compound names contain Greek letters (beta-cyclocitral...): a Windows console in cp1252 can't print them
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    a = parse_args()
    print("SCENTINEL - Tool 1: minimum OBP panel so nothing imitates the target\n")

    data_dir = pick_data_dir(a.data_dir)
    D = ScentinelData.load(data_dir, diagnostic=a.diag)
    print(D.summary())

    # -- source ---------------------------------------------------------------
    source = a.source
    if source is None:
        print("\nBinding data type:\n  [1] Experimental only - picks the best-measured OBPs that help (trustworthy panel)"
              "\n  [2] With predicted (Tanimoto) - predictions take part in the choice (explores further)")
        source = "predicted" if ask("Select [1/2, default 1]: ", "1") in ("2", "p", "predicted") else "experimental"
    if source == "predicted" and not D.has_predicted:
        sys.exit("ERROR: predicted mode needs the two *_imputed_*.csv files.")

    # -- target ---------------------------------------------------------------
    target_q = a.target
    while True:
        if target_q is None:
            target_q = ask("\nName or CAS of the target VOC: ")
            if not target_q:
                continue
        try:
            target = D.resolve(target_q)
            break
        except CompoundAmbiguous as e:
            if a.target is not None:
                print(f"  x {e}")
                sys.exit(1)
            target = choose_compound_match(target_q, e.matches, e.display_matches)
            break
        except CompoundNotFound as e:
            print(f"  x {e}")
            if a.target is not None:
                sys.exit(1)
            target_q = None
    print(f"  Target: {target}  (CAS {D.meta.set_index('label').loc[target, 'CAS']})")

    # -- interferents -----------------------------------------------------------
    path = a.interferents or ask("Interferent file: ")
    if not path or not os.path.isfile(path):
        sys.exit(f"ERROR: interferent file '{path}' not found.")
    try:
        interferents, _ = D.resolve_many(read_interferent_file(path), strict=True)
    except CompoundNotFound as e:
        print("  x Some interferents are not in the database:\n    " + str(e).replace("\n", "\n    "))
        sys.exit("  == Fix the interferent file and run again. ==")
    if target in interferents:
        interferents.remove(target)
        print("  (the target itself was in the interferent list - removed)")
    print(f"  {len(interferents)} interferents: " + ", ".join(D.short(i) for i in interferents))

    # -- starting OBP(s) ---------------------------------------------------------------
    # mandatory: the search always starts from the OBP(s) the user gives
    start = []
    queries = a.start_with
    while not start:
        if not queries:
            queries = (ask("\nStarting OBP(s), space-separated (e.g. the protein chosen by SCENTINEL v3): ")
                       or "").split()
            if not queries:
                print("  x A starting OBP is required.")
                continue
        try:
            for q in queries:
                o = resolve_obp(q, D.obps)
                if o not in start:
                    start.append(o)
        except LookupError as e:
            if a.start_with:
                sys.exit(f"  x --start-with: {e}")
            print(f"  x {e}")
            start, queries = [], None

    try:
        run = analyse_tool1(D, target, interferents, source, start, a)
    except NotEnoughCandidates as e:
        sys.exit(f"  x {e}")
    print_tool1(D, run, a)
    save_tool1(D, run, a)


# ----------------------------------------------------------------------------- analysis (no printing)
class NotEnoughCandidates(RuntimeError):
    pass


def analyse_tool1(D, target, interferents, source, start, a):
    """The whole Tool 1 pipeline, without printing. Used by main() and by
    generate_panel_report.py. `a` holds the options (argparse namespace or
    anything with the same attributes). Returns a dict with every result."""
    S = D.specificity_matrix(target, interferents, source, a.min_confidence, a.only_confident)
    tgt_s = S.S[target]
    has_target = tgt_s.notna() & (tgt_s > 0)
    c2 = interferent_c2(D, S, interferents)
    binds = binds_matrix(D, S)
    blind = ~binds[[target] + interferents].any(axis=1)     # no real binding to any compound of the run
    candidates = [o for o in S.obps if has_target[o] and c2[o] >= a.min_c2 and not blind[o]]
    n_blind = int((has_target & (c2 >= a.min_c2) & blind).sum())
    if len(candidates) < 2:
        best = c2[has_target].sort_values(ascending=False).head(3)
        raise NotEnoughCandidates(
            f"Fewer than 2 candidate OBPs (c2 >= {a.min_c2:.2f}). Highest c2 among OBPs with target data: "
            + ", ".join(f"{o} {v:.2f}" for o, v in best.items())
            + ". Lower --min-c2 or use --source predicted.")
    warnings = []
    for o in start:
        if not has_target[o]:
            warnings.append(f"{o} has no Ki data for the target: it cannot see it (kept because you asked for it).")
        elif o not in candidates:
            warnings.append(f"{o} has c2 = {c2[o]:.2f} (< {a.min_c2:.2f}): little is known about the "
                            "interferents on it, it may look more specific than it is.")

    sel = selectivity(S, target, interferents)
    priority = a.priority if a.priority != "auto" else ("measured" if source == "experimental" else "score")
    pick_key = (lambda o, gain, score: gain * float(c2[o]) ** 2) if priority == "measured" else None
    res = greedy_panel_for_specificity(
        S, target=target, interferents=interferents, noise=a.noise,
        candidates=candidates, min_gain=a.min_gain,
        max_sensors=a.max_sensors or None,
        tie_break=lambda o: (float(sel[o]), float(tgt_s[o])),
        lookahead=a.lookahead, initial=start, pick_key=pick_key,
    )

    added, uncovered = [], []
    if a.cover > 0 and interferents:
        added, uncovered = complete_coverage(S, res.panel, target, interferents, candidates, a.cover,
                                             known=binds)
        if uncovered:   # no candidate (c2 filter) binds them: look among all non-blind OBPs with target data
            pool = [o for o in S.obps if has_target[o] and not blind[o]]
            more, uncovered = complete_coverage(S, res.panel + [o for o, _, _ in added], target,
                                                interferents, pool, a.cover, known=binds)
            added += more
    panel = res.panel + [o for o, _, _ in added]
    final_resid = added[-1][2] if added else (res.result.trace[-1].score if res.result.trace else 0.0)
    passed = final_resid >= a.noise

    ev = D.cell_evidence(panel, interferents + [target], source, a.min_confidence, a.only_confident)
    ki_t = 1.0 / tgt_s
    rows = []
    steps = [(st.added, st.score, st.gain, None) for st in res.result.trace] + \
            [(o, sc, np.nan, cov) for o, cov, sc in added]
    for k, (o, score, gain, cov) in enumerate(steps, 1):
        e = ev.loc[o]
        if k <= len(start):
            how = "start"
        elif cov is not None:
            how = "cover"
        elif score == 0 and k < len(res.result.trace):
            how = "set"
        else:
            how = "search"
        prev = rows[-1]["residual"] if rows else 0.0
        rows.append({"step": k, "OBP": o, "added_by": how, "residual": score,
                     "gain": (score - prev) if k > len(start) else np.nan,
                     "Ki_target_uM": ki_t[o], "selectivity": sel[o],
                     "interferents_with_data": int(S.S.loc[o, interferents].notna().sum()),
                     "c2": c2[o], "cells_measured": int(e.measured), "cells_imputed": int(e.imputed),
                     "mean_confidence": e.mean_conf,
                     "covers": ", ".join(D.short(i) for i in cov) if cov else ""})
    steps_df = pd.DataFrame(rows)

    recipe, missing = [], []
    if panel and interferents:
        c_star, _ = best_imitation_recipe(S, panel, target, interferents)
        recipe = sorted(((float(c), D.short(i)) for c, i in zip(c_star, interferents) if c > 1e-6), reverse=True)
        missing = [D.short(i) for i in interferents if S.S.loc[panel, i].isna().any()]

    wl = lab_worklist(D, S, panel, target, interferents)
    wp = frag = bets = None
    r_all = np.nan
    if panel and interferents and passed:
        wp = weak_points(D, S, panel, target, interferents, a.noise, wl)
        frag = measured_fragility(D, S, panel, target, interferents, a.noise)
        bets = panel_bets(wp, frag, panel, interferents, D, target)
        r_all = all_predictions_residual(D, S, panel, target, interferents) if len(wp) else final_resid
        if len(wp):
            v = wp.set_index(["OBP", "compound"])
            wl = wl.join(v[["verdict", "breaks_if", "fold_to_break", "Ki_predicted_uM", "residual_if_predicted"]],
                         on=["OBP", "compound"])
            wl["_v"] = wl["verdict"].map(VERDICT_ORDER)
            wl = wl.sort_values(["_v", "fold_to_break", "panel_rank"], kind="stable") \
                   .drop(columns="_v").reset_index(drop=True)
    mc_p = None
    if getattr(a, "mc", 0) and panel:
        mc_p = monte_carlo_collision_probability(S, panel, target, interferents, a.noise,
                                                 n_draws=a.mc, rng=np.random.default_rng(42))
    grade, why = grade_tool1(bets, a.noise, r_all, passed)
    return dict(target=target, interferents=interferents, source=source, start=start, S=S,
                n_with_target=int(has_target.sum()), n_candidates=len(candidates), n_blind=n_blind,
                priority=priority, warnings=warnings, res=res, stop_reason=res.stop_reason,
                added=added, uncovered=uncovered, panel=panel, final_resid=final_resid, passed=passed,
                steps=steps_df, recipe=recipe, missing=missing, worklist=wl, weak_points=wp,
                fragility=frag, bets=bets, r_all=r_all, mc_p=mc_p, grade=grade, why=why,
                noise=a.noise, c2=c2)


def grade_tool1(bets, noise, r_all, passed):
    """SOLID / CONDITIONAL / WEAK (or FAILS when the panel does not clear the noise)."""
    if not passed:
        return "FAILS", "the panel does not clear the noise floor even with the data used"
    if bets is None or bets.empty:
        b = pd.DataFrame(columns=["against", "off_by"])
    else:
        b = bets[bets["against"] | (bets["off_by"] <= BET_MAX_FOLD)]
    n_against = int(b["against"].sum()) if len(b) else 0
    n_close = int((b["off_by"] < 2).sum()) if len(b) else 0
    if len(b) == 0 and not (pd.notna(r_all) and r_all < noise):
        return "SOLID", "no single unverified or fragile value can plausibly break it"
    if n_against or (pd.notna(r_all) and r_all < noise):
        return "WEAK", (f"{n_against} bet(s) go AGAINST the Tanimoto prediction" if n_against
                        else "it fails if every prediction is true at once")
    if n_close:
        return "CONDITIONAL", f"{n_close} bet(s) are lost if a guess is off by less than 2x"
    return "CONDITIONAL", "it rests on unverified values, but each has a 2x margin or more"


# ----------------------------------------------------------------------------- printing
def print_tool1(D, run, a):
    sep = "-" * 100
    verbose = getattr(a, "verbose", False)
    print(f"\n  Candidates: {run['n_candidates']} of {run['n_with_target']} OBPs with target data "
          f"(c2 >= {a.min_c2:.2f}; {run['n_blind']} blind OBPs removed).  "
          f"Priority: {run['priority']}.")
    for w in run["warnings"]:
        print(f"  ! {w}")

    st = run["steps"]
    print(f"\n{sep}\n  PANEL  (noise floor = {run['noise']})\n{sep}")
    print(f"  {'#':>2}  {'OBP':<20} {'residual':>9} {'added by':>9} {'Ki_tgt':>7} {'c2':>5} {'meas':>5} {'imput':>5}")
    for r in st.itertuples():
        extra = f"   <- binds {r.covers}" if r.covers else ""
        print(f"  {r.step:>2}. {r.OBP:<20} {r.residual:>9.4f} {r.added_by:>9} {r.Ki_target_uM:>7.2f} "
              f"{r.c2:>5.2f} {r.cells_measured:>5} {r.cells_imputed:>5}{extra}")
    print(f"  {STOP_TEXT.get(run['stop_reason'], run['stop_reason'])}" if not run["added"] or
          run["stop_reason"] == "target_crossed" else
          f"  Residual after coverage: {run['final_resid']:.4f} -> "
          f"{'clears' if run['passed'] else 'does NOT clear'} the noise floor.")
    if run["uncovered"]:
        print("  ! No OBP binds: " + ", ".join(D.short(i) for i in run["uncovered"]) + " (invisible to the panel).")
    if verbose:
        print("  added by: start = --start-with; search = greedy; set = added together by the lookahead; "
              "cover = so every interferent binds a panel OBP")
        print("  c2 = interferent knowledge (measured 1, imputed x confidence, missing 0); "
              "meas/imput = target+interferent cells measured / imputed")
    if run["recipe"]:
        print("  Closest fake mixture: " + ", ".join(f"{c:.3f} x {n}" for c, n in run["recipe"]))

    if run["bets"] is not None:
        show_panel_bets(run["bets"], run["panel"], run["noise"], run["final_resid"], run["r_all"], a.show_bets)
        wl = run["worklist"]
        print(f"\n  TO MEASURE ({len(wl)} pair(s) of the panel without a lab Ki; most urgent first)")
        if len(wl):
            print(f"    {'OBP':<18} {'compound':<24} {'status':<9} {'risk':<20} {'breaks if':<14} {'Ki_pred':>8} {'conf':>5}")
            for r in wl.head(a.show_weak if verbose else 15).itertuples():
                risk = getattr(r, "verdict", "") if isinstance(getattr(r, "verdict", ""), str) else ""
                bi = getattr(r, "breaks_if", "") if isinstance(getattr(r, "breaks_if", ""), str) else ""
                kp = getattr(r, "Ki_predicted_uM", np.nan)
                print(f"    {r.OBP:<18} {r.compound[:24]:<24} {r.status:<9} {risk:<20} {bi:<14} "
                      f"{(f'{kp:.3g}' if pd.notna(kp) else '-'):>8} "
                      f"{(f'{r.confidence:.2f}' if pd.notna(r.confidence) else '-'):>5}")
            if len(wl) > 15 and not verbose:
                print(f"    ... {len(wl) - 15} more in the lab CSV")
        if verbose:
            show_lab_worklist(run["worklist"], len(run["interferents"]) + 1)
            show_weak_points(D, run["S"], run["panel"], run["target"], run["interferents"], run["noise"],
                             run["weak_points"], run["fragility"], run["final_resid"], a.show_weak)
    elif run["panel"]:
        print(f"\n  VERDICT: {run['grade']} - {run['why']}.")
        show_lab_worklist(run["worklist"], len(run["interferents"]) + 1)
    if run["mc_p"] is not None:
        print(f"\n  Monte Carlo P(collision) ({a.mc:,} random mixtures): {run['mc_p']:.4f}")


def save_tool1(D, run, a):
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe = re.sub(r"[^\w]+", "_", D.short(run["target"]))[:40]
    base = os.path.join(a.out, f"tool1_{{}}_{safe}_{run['source']}_{stamp}.csv")
    run["steps"].assign(stop_reason=run["stop_reason"], passes=run["passed"], final_residual=run["final_resid"],
                        verdict=run["grade"], noise=run["noise"], source=run["source"]) \
        .to_csv(base.format("panel"), index=False)
    run["worklist"].to_csv(base.format("lab"), index=False)
    saved = ["panel", "lab"]
    if run["bets"] is not None:
        run["bets"].to_csv(base.format("bets"), index=False)
        saved.append("bets")
        if getattr(a, "verbose", False):
            pd.concat([run["weak_points"].assign(kind="not measured"),
                       run["fragility"].assign(kind="has value")], ignore_index=True) \
                .to_csv(base.format("weak_points"), index=False)
            saved.append("weak_points")
    print(f"\n  Saved: {base.format('{' + ','.join(saved) + '}')}")


if __name__ == "__main__":
    main()
