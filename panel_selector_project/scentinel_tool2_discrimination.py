"""
SCENTINEL - Tool 2 (discrimination) on the real SCENTINEL CSVs.

Smallest OBP panel that best separates two classes of GC-MS VOC profiles
(e.g. healthy vs diseased vines), projected through the forward model
response_j = sum_i (1/Ki_ji) * c_i built from the SCENTINEL Ki tables.

Input table (.xlsx or .csv), one row per compound:
    CAS | name | mean healthy | SE healthy | mean diseased | SE diseased
CAS is mandatory, name optional, 'n.d.' is read as 0 (docs/INPUT_FORMAT.md).
(Per-sample profiles with a 'class' column are also accepted.)

Examples
    python scentinel_tool2_discrimination.py --data-dir data \
        --profiles examples/example_vineyard_profiles_SYNTHETIC.csv --source experimental
    python scentinel_tool2_discrimination.py --data-dir data \
        --profiles my_gcms.csv --classes healthy diseased --source predicted --min-confidence 0.3
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

from panel_selector import (ScentinelData, greedy_panel_for_discrimination, project_profiles, fisher_ratio,
                            greedy_panel_from_summary, summary_moments, fisher_from_summary,
                            summary_j_function, replicate_j_function, j_cell_scan,
                            error_from_j, j_for_error)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STOP_TEXT = {
    "target_crossed": "OK - separation reached --target-score.",
    "diminishing_returns": "Stopped: the next OBP adds almost no separation (redundant with the panel).",
    "library_exhausted": "Library exhausted: every candidate OBP was added.",
    "max_sensors_reached": "Stopped at --max-sensors.",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="folder with the SCENTINEL CSVs (default: ./data, ./ or script dir)")
    p.add_argument("--diag", default=None, help="Tanimoto diagnostic CSV (default: auto-detect)")
    p.add_argument("--profiles", default=None, help="GC-MS profile CSV (rows = samples)")
    p.add_argument("--class-col", default="class", help="name of the class column (default 'class')")
    p.add_argument("--classes", nargs=2, default=None, metavar=("A", "B"),
                   help="the two class labels to separate (default: the first two found)")
    p.add_argument("--source", choices=["experimental", "predicted"], default=None,
                   help="predicted = experimental values + Tanimoto central estimates (*_imputed_pred.csv)")
    p.add_argument("--metric", choices=["fisher", "cv_accuracy"], default="fisher")
    p.add_argument("--target-score", type=float, default=None,
                   help="stop once separation exceeds this (e.g. 0.95 for cv_accuracy)")
    p.add_argument("--min-gain", type=float, default=None,
                   help="diminishing-returns threshold (default 0.01 x first Fisher J, or 0.005 for cv_accuracy)")
    p.add_argument("--max-sensors", type=int, default=8, help="hard cap on panel size (default 8, 0 = none)")
    p.add_argument("--censored", choices=["zero", "keep"], default="zero",
                   help="'>X uM' cells (no measurable binding): zero = sensitivity 0 (default), keep = 1/(1.1 X) as in main.py")
    p.add_argument("--scale", choices=["anchored", "inverse"], default="anchored",
                   help="sensitivity scale: anchored = same as s1 in main.py (Ki 1 uM -> 1, Ki >= 40 uM -> 0, log), "
                        "inverse = 1/Ki (default anchored)")
    p.add_argument("--ki-max", type=float, default=40.0, help="weak-binding anchor in uM (default 40, as main.py)")
    p.add_argument("--ki-min", type=float, default=1.0, help="excellent-binding anchor in uM (default 1, as main.py)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="predicted mode: drop imputed cells with Tanimoto confidence c1 below this (0-1)")
    p.add_argument("--only-confident", action="store_true",
                   help="predicted mode: keep only imputed cells with decision == 'confident'")
    p.add_argument("--sensor-noise", type=float, default=0.05,
                   help="summary mode: sensor measurement noise as a fraction of the mean reading (default 0.05)")
    p.add_argument("--n", type=int, default=3, help="plants per group behind each SE (default 3)")
    p.add_argument("--max-error", type=float, default=0.05,
                   help="largest acceptable share of plants misclassified (default 0.05 = 5 %%). It sets the J limit "
                        "J = 2 * z(1-error)^2 (5 %% -> J 5.4) used by the uncertainty check and the verdict")
    p.add_argument("--weak-frac", type=float, default=None,
                   help="alternative limit: a Ki 'breaks' the panel if it can pull J below this fraction of the "
                        "panel's J (e.g. 0.5). Overrides --max-error; --target-score overrides both")
    p.add_argument("--show-bets", type=int, default=10, help="how many bets to print (all go to the CSV)")
    p.add_argument("--verbose", action="store_true", help="print every detail (scale, noise, 8 best single OBPs)")
    p.add_argument("--out", default="results", help="folder for the CSV output (default ./results)")
    return p.parse_args()


# ----------------------------------------------------------------------------- uncertainty
KI_GRID = np.logspace(-1, 4, 81)        # 0.1 uM .. 10 mM
WEAK_BINDING_KI = 30.0
VERDICT_ORDER = {"BREAKS (prediction)": 0, "BREAKS if it binds": 1, "risk": 2, "safe": 3}


def signal_of_ki(ki, a):
    """Same Ki -> sensitivity rule as the matrix used in the run (--scale)."""
    ki = np.asarray(ki, float)
    if a.scale == "anchored":
        return np.clip(np.log10(a.ki_max / ki) / np.log10(a.ki_max / a.ki_min), 0.0, 1.0)
    return 1.0 / ki


def bet_text(c, o, lo, hi):
    at_min, at_max = lo <= KI_GRID[0] * 1.0001, hi >= KI_GRID[-1] * 0.9999
    if at_min and at_max:
        return f"{c} does NOT bind {o} at all"
    if at_min:
        return f"{c} binds {o} weakly or not at all (Ki > {hi:.3g} uM)"
    if at_max:
        return f"{c} binds {o} at least this strongly: Ki < {lo:.3g} uM"
    return f"Ki of {c} on {o} is below {lo:.3g} or above {hi:.3g} uM (not in between)"


def uncertainty_check(a, D, S, kind, panel, comps, J, shift):
    """Every panel cell whose Ki was NOT measured is moved over 0.1 uM - 10 mM,
    one at a time, and J is recomputed. A cell 'breaks' the panel if some Ki
    pulls J below the threshold (half the panel's J by default)."""
    A0 = S.S.loc[panel, comps].fillna(0.0).values.astype(float)
    J0 = J(A0)
    if a.target_score is not None and a.metric == "fisher":
        thr, thr_txt = a.target_score, f"--target-score {a.target_score:g}"
    elif a.weak_frac is not None:
        thr, thr_txt = a.weak_frac * J0, f"{a.weak_frac:.0%} of the panel's J"
    else:
        thr, thr_txt = j_for_error(a.max_error), f"max {a.max_error:.0%} of plants misclassified"
    sig = signal_of_ki(KI_GRID, a)
    ki_used = D.ki_table(kind, a.min_confidence, a.only_confident)
    rows = []
    for r, o in enumerate(panel):
        for j, c in enumerate(comps):
            if D.decision.loc[c, o] == "measured":
                continue
            used = pd.notna(S.S.loc[o, c])
            pred = ki_used.loc[c, o] if used else (D.ki_pred.loc[c, o] if D.ki_pred is not None else np.nan)
            pred = float(pred) if pd.notna(pred) and pred > 0 else np.nan
            status = "imputed" if used else ("not used" if pd.notna(pred) else "no data")
            Js = j_cell_scan(J, A0, r, j, sig)
            bad = KI_GRID[Js < thr]
            lo, hi = (float(bad.min()), float(bad.max())) if len(bad) else (None, None)
            J_pred = float(j_cell_scan(J, A0, r, j, signal_of_ki([pred], a))[0]) if pd.notna(pred) else np.nan
            ref = pred if pd.notna(pred) else WEAK_BINDING_KI
            if pd.notna(J_pred) and J_pred < thr:
                verdict, fold = "BREAKS (prediction)", 1.0
            elif lo is not None and lo <= KI_GRID[0] * 1.0001 and hi >= WEAK_BINDING_KI:
                verdict, fold = "BREAKS if it binds", 1.0
            elif lo is not None:
                verdict = "risk"
                fold = 1.0 if lo <= ref <= hi else 10 ** min(abs(np.log10(ref / lo)), abs(np.log10(ref / hi)))
            else:
                verdict, fold = "safe", np.nan
            sh = shift.get(c, np.nan)
            role = ("changes" if abs(sh) >= 1 else "noise") if pd.notna(sh) else "n.d."
            rows.append({"OBP": o, "compound": D.short(c), "compound_role": role, "shift_SD": sh,
                         "status": status, "Ki_predicted_uM": pred, "confidence": D.confidence.loc[c, o],
                         "J_now": J0, "J_if_predicted": J_pred, "J_min": float(Js.min()),
                         "error_now": error_from_j(J0), "error_worst": error_from_j(float(Js.min())),
                         "Ki_at_J_min_uM": float(KI_GRID[int(Js.argmin())]),
                         "Ki_break_low_uM": lo, "Ki_break_high_uM": hi,
                         "breaks_if": "never" if lo is None else bet_text("", "", lo, hi).split("(")[-1].rstrip(")")
                         if lo > KI_GRID[0] * 1.0001 or hi < KI_GRID[-1] * 0.9999 else "any Ki",
                         "verdict": verdict, "off_by": fold,
                         "bet": bet_text(D.short(c), o, lo, hi) if lo is not None else ""})
    # the same compound on EVERY panel OBP where its Ki is unmeasured, at once
    # (one cell at a time misses it: the other OBPs compensate)
    for j, c in enumerate(comps):
        cells = [r for r, o in enumerate(panel) if D.decision.loc[c, o] != "measured"]
        if len(cells) < 2:
            continue
        Js = np.empty(len(sig))
        for k, sv in enumerate(sig):
            A = A0.copy()
            for r in cells:
                A[r, j] = sv
            Js[k] = J(A)
        bad = KI_GRID[Js < thr]
        lo, hi = (float(bad.min()), float(bad.max())) if len(bad) else (None, None)
        preds = []
        for r in cells:
            o = panel[r]
            v = ki_used.loc[c, o] if pd.notna(S.S.loc[o, c]) else (D.ki_pred.loc[c, o] if D.ki_pred is not None else np.nan)
            preds.append(float(v) if pd.notna(v) and v > 0 else np.nan)
        A = A0.copy()
        for r, v in zip(cells, preds):
            if pd.notna(v):
                A[r, j] = float(signal_of_ki([v], a)[0])
        J_pred = J(A) if any(pd.notna(v) for v in preds) else np.nan
        if pd.notna(J_pred) and J_pred < thr:
            verdict, fold = "BREAKS (prediction)", 1.0
        elif lo is None:
            verdict, fold = "safe", np.nan
        elif lo <= KI_GRID[0] * 1.0001 and hi >= WEAK_BINDING_KI:
            verdict, fold = "BREAKS if it binds", 1.0
        else:
            verdict = "risk"
            fold = 1.0 if lo <= WEAK_BINDING_KI <= hi else \
                10 ** min(abs(np.log10(WEAK_BINDING_KI / lo)), abs(np.log10(WEAK_BINDING_KI / hi)))
        sh = shift.get(c, np.nan)
        role = ("changes" if abs(sh) >= 1 else "noise") if pd.notna(sh) else "n.d."
        where = "+".join(panel[r] for r in cells)
        rows.append({"OBP": f"ALL {len(cells)}: {where}", "compound": D.short(c), "compound_role": role,
                     "shift_SD": sh, "status": "joint", "Ki_predicted_uM": np.nan, "confidence": np.nan,
                     "J_now": J0, "J_if_predicted": J_pred, "J_min": float(Js.min()),
                     "Ki_at_J_min_uM": float(KI_GRID[int(Js.argmin())]),
                     "error_now": error_from_j(J0), "error_worst": error_from_j(float(Js.min())),
                     "Ki_break_low_uM": lo, "Ki_break_high_uM": hi, "breaks_if": "",
                     "verdict": verdict, "off_by": fold,
                     "bet": bet_text(D.short(c), f"the {len(cells)} panel OBPs where it is unmeasured ({where})",
                                     lo, hi) if lo is not None else ""})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_v"] = df["verdict"].map(VERDICT_ORDER)
        df = df.sort_values(["_v", "off_by"]).drop(columns="_v").reset_index(drop=True)
    # every prediction true at once
    A_all = A0.copy()
    for r, o in enumerate(panel):
        for j, c in enumerate(comps):
            if D.decision.loc[c, o] != "measured" and pd.isna(S.S.loc[o, c]) and D.ki_pred is not None:
                k = D.ki_pred.loc[c, o]
                if pd.notna(k) and k > 0:
                    A_all[r, j] = float(signal_of_ki([k], a)[0])
    return df, J0, (thr, thr_txt), J(A_all)


def err_txt(J):
    e = error_from_j(J)
    return f"{100 * e:.1f} %" if e >= 0.001 else "<0.1 %"


def show_uncertainty(df, panel, J0, thr, J_all, show_max, D, comps, S_used):
    thr, thr_txt = thr
    sep = "=" * 104
    bad = df[df["verdict"] != "safe"] if len(df) else df
    near = bad[(bad["verdict"] != "risk") | (bad["off_by"] <= 10)] if len(bad) else bad
    far = len(bad) - len(near)
    n_against = int((near["verdict"] == "BREAKS (prediction)").sum()) if len(near) else 0
    if J0 < thr:
        grade, why = "NOT ENOUGH", "even with the data used the panel does not reach the limit"
    elif len(near) == 0 and J_all >= thr:
        grade, why = "SOLID", "no single unmeasured Ki can plausibly pull J under the threshold"
    elif n_against or J_all < thr:
        grade = "WEAK"
        why = (f"{n_against} bet(s) go AGAINST the Tanimoto prediction" if n_against
               else "J falls under the threshold if every prediction is true at once")
    else:
        grade, why = "CONDITIONAL", "J rests on unmeasured Ki, but the predictions do not contradict it"
    print(f"\n{sep}\n  VERDICT: {grade}  -  panel {' + '.join(panel)}\n"
          f"  J = {J0:.2f} on the data used ({err_txt(J0)} of plants misclassified); limit J >= {thr:.2f} "
          f"({thr_txt}); {why}.\n"
          f"  If every not-measured Ki took its Tanimoto prediction at once: J = {J_all:.2f} ({err_txt(J_all)} misclassified) "
          f"-> {'still within the limit' if J_all >= thr else 'BEYOND the limit'}\n"
          f"  misclassified = Phi(-sqrt(J/2)): Gaussian readings, cut half-way between the class means (idealised)")
    print("  Ki known per panel OBP (measured / used prediction / missing, of "
          f"{len(comps)} compounds):")
    for o in panel:
        m = int(sum(D.decision.loc[c, o] == "measured" for c in comps))
        u = int(sum(D.decision.loc[c, o] != "measured" and pd.notna(S_used.loc[o, c]) for c in comps))
        print(f"    {o:<20} {m:>3} / {u:>3} / {len(comps) - m - u:>3}")
    if len(near):
        print(f"\n  WHAT THIS PANEL BETS ON ({len(near)} unverified Ki; most fragile first):")
        for i, r in enumerate(near.head(show_max).itertuples(), 1):
            tag = "  AGAINST prediction" if r.verdict == "BREAKS (prediction)" else \
                ("" if r.verdict == "BREAKS if it binds" else f"  (lost if off by x{r.off_by:.3g})")
            kind = {"noise": "does not change with disease, varies between plants",
                    "changes": "changes with disease", "n.d.": "not detected"}[r.compound_role]
            data = {"no data": "not measured, no prediction (counted as 'does not bind')",
                    "not used": f"not measured; prediction Ki {r.Ki_predicted_uM:.3g} uM, conf {r.confidence:.2f}, "
                                "left out (counted as 'does not bind')",
                    "imputed": f"not measured; prediction Ki {r.Ki_predicted_uM:.3g} uM, conf {r.confidence:.2f}, used",
                    "joint": "not measured on any of them; the same Ki applied to all at once"}[r.status]
            print(f"   {i:>2}. {r.bet}{tag}\n       J {r.J_now:.2f} -> {r.J_min:.2f} at Ki {r.Ki_at_J_min_uM:.3g} uM "
                  f"(misclassified {err_txt(r.J_now)} -> {err_txt(r.J_min)});"
                  f" compound {kind} ({r.shift_SD:+.1f} SD)\n       data: {data}")
        if len(near) > show_max:
            print(f"   ... {len(near) - show_max} more in the bets CSV (--show-bets N to print more)")
        print("  Measuring the first ones settles whether this panel can be trusted.")
    if far:
        print(f"  (+ {far} far-fetched bet(s): lost only if a prediction is more than 10x wrong)")
    print("  Missing Ki = 'does not bind'; each one (and each compound on all panel OBPs at once) is moved over")
    print("  0.1 uM - 10 mM and J recomputed. See docs/MISSING_DATA_ES.md.")
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
    print("SCENTINEL - Tool 2: OBP panel that best separates two classes of VOC profiles\n")
    D = ScentinelData.load(pick_data_dir(a.data_dir), diagnostic=a.diag)
    print(D.summary())

    source = a.source
    if source is None:
        print("\nBinding data type:\n  [1] Experimental only\n  [2] With predicted (Tanimoto) information")
        source = "predicted" if input("Select [1/2, default 1]: ").strip() in ("2", "p", "predicted") else "experimental"
    if source == "predicted" and not D.has_predicted:
        sys.exit("ERROR: predicted mode needs the *_imputed_pred.csv file.")

    path = a.profiles or input("GC-MS profile CSV: ").strip().strip('"').strip("'")
    if not path or not os.path.isfile(path):
        sys.exit(f"ERROR: profile file '{path}' not found.")
    if is_input_table(path):
        return run_input_table(a, D, source, path)
    classes, errors = D.load_profiles(path, class_col=a.class_col)
    if errors:
        print("  x Some profile columns are not in the database:\n    " + "\n    ".join(errors))
        sys.exit("  == Rename those columns (exact name or CAS) and run again. ==")
    names = a.classes or list(classes)[:2]
    if len(names) < 2 or any(n not in classes for n in names):
        sys.exit(f"ERROR: need two classes; found {list(classes)}")
    A, B = classes[names[0]], classes[names[1]]
    compounds = list(A.columns)
    print(f"\n  Classes: {names[0]} (n={len(A)})  vs  {names[1]} (n={len(B)})")
    print(f"  {len(compounds)} compounds: " + ", ".join(D.short(c) for c in compounds))
    shift = (B.mean() - A.mean()) / np.sqrt((A.var(ddof=1) + B.var(ddof=1)).replace(0, np.nan))
    print("  Per-compound shift (B - A, in pooled SDs): " +
          ", ".join(f"{D.short(c)} {shift[c]:+.1f}" for c in compounds))

    kind = "experimental" if source == "experimental" else "pred"
    S = D.sensitivity(kind, a.min_confidence, a.only_confident, censored=a.censored,
                      scale=a.scale, ki_min=a.ki_min, ki_max=a.ki_max)
    print(f"  Sensitivity scale: {a.scale}" + (f" (Ki {a.ki_min:g} uM -> 1, Ki >= {a.ki_max:g} uM -> 0)" if a.scale=='anchored' else ' (1/Ki)'))
    n_have = (S.S[compounds].fillna(0) > 0).sum(axis=1)   # compounds the OBP actually binds
    candidates = [o for o in S.obps if n_have[o] > 0]
    print(f"\n  Candidate OBPs (bind >= 1 profile compound): {len(candidates)}")
    if not candidates:
        sys.exit("  x No OBP has data for these compounds.")

    min_gain = a.min_gain
    if min_gain is None:
        if a.metric == "fisher":
            first = max(fisher_ratio(project_profiles(S, A, [o]), project_profiles(S, B, [o])) for o in candidates)
            min_gain = 0.01 * first if np.isfinite(first) and first > 0 else 1e-3
        else:
            min_gain = 0.005

    res = greedy_panel_for_discrimination(
        S, class_a_profiles=A, class_b_profiles=B, metric=a.metric,
        target_score=a.target_score, candidates=candidates, min_gain=min_gain,
        max_sensors=a.max_sensors or None,
        tie_break=lambda o: float(n_have[o]),
    )

    ev = D.cell_evidence(res.panel, compounds, source, a.min_confidence, a.only_confident)
    label = "Fisher J" if a.metric == "fisher" else "CV acc."
    sep = "-" * 96
    print(f"\n{sep}\n  GREEDY PANEL  (metric = {a.metric}, min_gain = {min_gain:.4g})\n{sep}")
    print(f"  {'#':>2}  {'OBP':<20} {label:>9} {'gain':>9}  {'cpd cov':>8} {'meas':>5} {'imput':>5} {'conf':>5}  "
          f"main compounds seen")
    print(sep)
    rows = []
    for k, st in enumerate(res.result.trace, 1):
        o = st.added
        e = ev.loc[o]
        gain = "first" if not np.isfinite(st.gain) else f"{st.gain:.4f}"
        s_row = S.S.loc[o, compounds].dropna(); s_row = s_row[s_row > 0].sort_values(ascending=False)
        ki_tab = D.ki_table(kind, a.min_confidence, a.only_confident)
        seen = ", ".join(f"{D.short(c)}({ki_tab.loc[c, o]:.1f})" for c, v in s_row.head(3).items())
        et = f" ({err_txt(st.score)})" if a.metric == "fisher" else ""
        print(f"  {k:>2}. {o:<20} {st.score:>9.4f}{et} {gain:>9}  {int(e.with_data):>3}/{len(compounds):<4} "
              f"{int(e.measured):>5} {int(e.imputed):>5} {e.mean_conf:>5.2f}  {seen}")
        rows.append({"step": k, "OBP": o, "score": st.score,
                     "gain": st.gain if np.isfinite(st.gain) else np.nan,
                     "compounds_with_data": int(e.with_data), "cells_measured": int(e.measured),
                     "cells_imputed": int(e.imputed), "mean_confidence": e.mean_conf,
                     "top_compounds_Ki_uM": seen})
    print(sep)
    print(f"  {STOP_TEXT.get(res.stop_reason, res.stop_reason)}")
    print("  cpd cov = profile compounds with Ki data on that OBP (missing ones are assumed NOT to bind)")
    print("  main compounds seen = strongest binders among the profile compounds, Ki in uM")

    J = replicate_j_function(A, B, compounds)
    if a.metric != "fisher":
        print("  (uncertainty check uses the Fisher J of the panel, not the CV accuracy)")
    unc, J0, thr, J_all = uncertainty_check(a, D, S, kind, res.panel, compounds, J, shift.to_dict())
    show_uncertainty(unc, res.panel, J0, thr, J_all, a.show_bets, D, compounds, S.S)
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(a.out, f"tool2_panel_{names[0]}_vs_{names[1]}_{source}_{stamp}.csv")
    pd.DataFrame(rows).assign(stop_reason=res.stop_reason, metric=a.metric, source=source).to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    out_b = os.path.join(a.out, f"tool2_bets_{names[0]}_vs_{names[1]}_{source}_{stamp}.csv")
    unc.to_csv(out_b, index=False)
    print(f"  Saved: {out_b}")


def is_input_table(path):
    """The VineSplit input table has the columns 'CAS' and 'mean healthy'."""
    try:
        cols = (pd.read_excel(path, nrows=0) if path.lower().endswith((".xlsx", ".xls"))
                else pd.read_csv(path, nrows=0)).columns
    except Exception:
        return False
    cols = {str(c).strip().lower().replace("_", " ") for c in cols}
    return {"cas", "mean healthy"}.issubset(cols)


def run_input_table(a, D, source, path):
    """CAS | name | mean healthy | SE healthy | mean diseased | SE diseased  ('n.d.' = 0)."""
    A, B, rep = D.load_input(path, n=a.n)
    print(f"\n  INPUT: {os.path.basename(path)}   (n = {a.n} plants per group; change with --n)")
    if rep["skipped"]:
        print(f"  skipped ({len(rep['skipped'])}): " + "; ".join(f"{x[0]} [{x[1]}]" for x in rep["skipped"]))
    if rep["nd"]:
        print("  'n.d.' read as 0: " + "; ".join(f"{x[0]} {x[1]}" for x in rep["nd"]))
    analyse_summary(a, D, source, A, B, ["healthy", "diseased"], os.path.splitext(os.path.basename(path))[0])


def analyse_tool2_summary(D, source, A, B, a):
    """Whole Tool 2 pipeline in summary mode (mean +/- SD per class), no printing.
    Used by analyse_summary() and generate_panel_report.py."""
    comps = list(A.index)
    shift = (B["mean"] - A["mean"]) / np.sqrt(A["sd"] ** 2 + B["sd"] ** 2).replace(0, np.nan)
    kind = "experimental" if source == "experimental" else "pred"
    S = D.sensitivity(kind, a.min_confidence, a.only_confident, censored=a.censored,
                      scale=a.scale, ki_min=a.ki_min, ki_max=a.ki_max)
    n_have = (S.S[comps].fillna(0) > 0).sum(axis=1)
    candidates = [o for o in S.obps if n_have[o] > 0]
    if not candidates:
        raise RuntimeError("No OBP binds any of these compounds.")

    def single(o):
        ma, ca = summary_moments(S, [o], A["mean"], A["sd"]); mb, cb = summary_moments(S, [o], B["mean"], B["sd"])
        return fisher_from_summary(ma, ca, mb, cb, a.sensor_noise), ma[0], mb[0], np.sqrt(ca[0, 0]), np.sqrt(cb[0, 0])
    ki_tab = D.ki_table(kind, a.min_confidence, a.only_confident)
    srows = []
    for o in candidates:
        J1, ma, mb, sa, sb = single(o)
        s_row = S.S.loc[o, comps].fillna(0); s_row = s_row[s_row > 0].sort_values(ascending=False)
        srows.append({"OBP": o, "J": J1, "error": error_from_j(J1), "reading_A": ma, "sd_A": sa,
                      "reading_B": mb, "sd_B": sb,
                      "n_measured": int(sum(D.decision.loc[c, o] == "measured" for c in comps)),
                      "binds": ", ".join(f"{D.short(c)}({ki_tab.loc[c, o]:.1f})" for c in s_row.index[:3])})
    singles = pd.DataFrame(srows).sort_values("J", ascending=False).reset_index(drop=True)
    min_gain = a.min_gain if a.min_gain is not None else max(0.01 * singles["J"].iloc[0], 1e-3)
    res = greedy_panel_from_summary(S, A, B, sensor_noise=a.sensor_noise, target_score=a.target_score,
                                    candidates=candidates, min_gain=min_gain,
                                    max_sensors=a.max_sensors or None, tie_break=lambda o: float(n_have[o]))
    steps = pd.DataFrame([{"step": k, "OBP": st.added, "J_panel": st.score, "misclassified": error_from_j(st.score),
                           "gain": st.gain if np.isfinite(st.gain) else np.nan,
                           "n_measured": int(sum(D.decision.loc[c, st.added] == "measured" for c in comps))}
                          for k, st in enumerate(res.result.trace, 1)])
    J = summary_j_function(A, B, comps, a.sensor_noise)
    unc, J0, thr, J_all = uncertainty_check(a, D, S, kind, res.panel, comps, J, shift.to_dict())
    grade, why = grade_tool2(unc, J0, thr[0], J_all)
    known = pd.DataFrame([{"OBP": o,
                           "measured": int(sum(D.decision.loc[c, o] == "measured" for c in comps)),
                           "predicted_used": int(sum(D.decision.loc[c, o] != "measured" and pd.notna(S.S.loc[o, c])
                                                     for c in comps))} for o in res.panel])
    known["missing"] = len(comps) - known["measured"] - known["predicted_used"]
    return dict(source=source, comps=comps, shift=shift, S=S, n_candidates=len(candidates), singles=singles,
                min_gain=min_gain, res=res, stop_reason=res.stop_reason, panel=res.panel, steps=steps,
                bets=unc, J0=J0, thr=thr, J_all=J_all, grade=grade, why=why, known=known,
                n=int(A["n"].iloc[0]), A=A, B=B)


def grade_tool2(df, J0, thr, J_all):
    bad = df[df["verdict"] != "safe"] if len(df) else df
    near = bad[(bad["verdict"] != "risk") | (bad["off_by"] <= 10)] if len(bad) else bad
    n_against = int((near["verdict"] == "BREAKS (prediction)").sum()) if len(near) else 0
    if J0 < thr:
        return "NOT ENOUGH", "even with the data used the panel does not reach the limit"
    if len(near) == 0 and J_all >= thr:
        return "SOLID", "no single unmeasured Ki can plausibly pull J under the limit"
    if n_against or J_all < thr:
        return "WEAK", (f"{n_against} bet(s) go AGAINST the Tanimoto prediction" if n_against
                        else "J falls under the limit if every prediction is true at once")
    return "CONDITIONAL", "J rests on unmeasured Ki, but the predictions do not contradict it"


def analyse_summary(a, D, source, A, B, names, tag):
    run = analyse_tool2_summary(D, source, A, B, a)
    comps, shift = run["comps"], run["shift"]
    verbose = getattr(a, "verbose", False)
    print(f"\n  SUMMARY MODE: mean +/- SE, n = {run['n']} per group.  {names[0]} vs {names[1]}.  "
          f"{len(comps)} compounds, shift ({names[1]} - {names[0]}, in SDs):")
    print("    " + ", ".join(f"{D.short(c)} " + ("n.d." if np.isnan(shift[c]) else f"{shift[c]:+.1f}") for c in comps))
    if verbose:
        print(f"  Sensitivity scale: {a.scale}" + (f" (Ki {a.ki_min:g} uM -> 1, Ki >= {a.ki_max:g} uM -> 0)"
                                                  if a.scale == 'anchored' else ' (1/Ki)'))
        print(f"  Sensor noise: {a.sensor_noise:.0%} of the mean reading. Compounds assumed to vary independently.")
    print(f"  Candidate OBPs (bind >= 1 compound): {run['n_candidates']}")
    sep = "-" * 104
    nshow = 8 if verbose else 5
    print(f"\n{sep}\n  BEST SINGLE OBPs  (J alone; err = plants misclassified; meas = Ki measured)\n{sep}")
    print(f"  {'OBP':<20} {'J':>7} {'err':>7} {'meas':>5}  {'reading '+names[0]:>20} {'reading '+names[1]:>20}  binds (Ki uM)")
    for r in run["singles"].head(nshow).itertuples():
        print(f"  {r.OBP:<20} {r.J:>7.2f} {err_txt(r.J):>7} {r.n_measured:>3}/{len(comps):<2} "
              f"{r.reading_A:>10.1f} +/- {r.sd_A:<6.1f} {r.reading_B:>10.1f} +/- {r.sd_B:<6.1f}  {r.binds}")
    print(f"{sep}\n  PANEL  (multivariate Fisher)\n{sep}")
    for r in run["steps"].itertuples():
        gain = "first" if not np.isfinite(r.gain) else f"+{r.gain:.2f}"
        print(f"  {r.step:>2}. {r.OBP:<20} J = {r.J_panel:>7.2f} (err {err_txt(r.J_panel):>6})   gain {gain:<8}  "
              f"Ki measured {r.n_measured}/{len(comps)}")
    print(f"  {STOP_TEXT.get(run['stop_reason'], run['stop_reason'])}")
    show_uncertainty(run["bets"], run["panel"], run["J0"], run["thr"], run["J_all"], a.show_bets, D, comps,
                     run["S"].S)
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(a.out, f"tool2_{{}}_{tag}_{source}_{stamp}.csv")
    run["steps"].assign(stop_reason=run["stop_reason"], verdict=run["grade"], source=source, mode="summary") \
        .to_csv(base.format("panel"), index=False)
    run["bets"].to_csv(base.format("bets"), index=False)
    print(f"\n  Saved: {base.format('{panel,bets}')}")
    return run


if __name__ == "__main__":
    main()
