"""
confidence.py - Confidence system for the OBP Finder (V3)
------------------------------------------------------------
Separates two questions the program used to conflate:

    "Is the candidate good?"      -> Score        (what it already computed)
    "How much evidence do I have?" -> Confidence   (new)

No candidate is ever excluded for having incomplete data: what drops is its confidence.

Three numbers per candidate, never merged into a single one:
    Conservative = Score x Confidence   -> what I can claim today
    Potential    = Score                -> how far it could go if everything were measured
    Gain         = Potential - Conservative  -> how much ignorance is at stake

Table 3 (pending work) is sorted by Gain x Potential: spend the experimental
effort where you cut the most uncertainty on a candidate that is worth it.
"""

import os
import numpy as np
import pandas as pd


# ── CONSTANTS ────────────────────────────────────────────────────────────────

# Sigma normalization. sigma_log = 1.0 means the donors disagree by a factor
# of 10x in Ki -> zero confidence.
SIGMA_MAX = 1.0

# Penalty for having few donors. With n_donors=1 sigma is 0 out of sheer
# mathematics (there is nobody to disagree with), not because of agreement.
#   support = 1 - 1/(1+n)   ->  1 donor: 0.50 · 2: 0.67 · 5: 0.83 · 8: 0.89
# The jump from 1 to 2 donors is worth the most; from 7 to 8 it adds almost nothing.
USE_SUPPORT_FACTOR = True

# Weight of an imputed interferent relative to a measured one, in the c2 computation.
IMPUTED_WEIGHT = 0.5

# Default confidence of an imputed cell when diag_table cannot be found.
C1_FALLBACK = 0.5

# Color thresholds for the overall confidence
CONF_HIGH = 0.80
CONF_MID = 0.50


# -- LOADING THE DIAGNOSTIC --------------------------------------------------

def load_diag_c1(diag_path, sigma_max=SIGMA_MAX, use_support=USE_SUPPORT_FACTOR):
    """
    Read results/tanimoto_diag_table.csv and return:
        {(row_idx, OBP): c1}   for every IMPUTED cell

    The presence of a key means "this cell is predicted".
    Its absence (while there is a value in the CSV) means "measured" -> c1 = 1.0

    c1 = T_max x max(0, 1 - sigma/sigma_max) x support(n_donors)

    They are multiplied (not added) because all three conditions are NECESSARY:
    a close neighbour, donors that agree, enough donors. If one fails, all falls.
    """
    if not diag_path or not os.path.isfile(diag_path):
        return None

    d = pd.read_csv(diag_path)
    needed = {"row_idx", "OBP", "T_max", "sigma_log", "n_donors"}
    if not needed.issubset(d.columns):
        raise ValueError(
            f"{diag_path} does not have the expected columns {sorted(needed)}.\n"
            f"Found: {list(d.columns)}"
        )

    dispersion = (1.0 - d["sigma_log"] / sigma_max).clip(lower=0.0)
    if use_support:
        support = 1.0 - 1.0 / (1.0 + d["n_donors"])
    else:
        support = 1.0

    c1 = (d["T_max"] * dispersion * support).clip(0.0, 1.0)
    return {(int(r), str(o)): float(v)
            for r, o, v in zip(d["row_idx"], d["OBP"], c1)}


def check_files_match(df_selectivity, df_affinity, path_sel, path_af):
    """
    Both CSVs come out of the SAME call to fill_gaps(): same rows, same
    columns. If they do not match, something serious has happened (files from
    different runs, mixed-up folders...).

    Better to blow up here than to silently use the conservative Ki as the
    affinity - which is exactly the bug we want to avoid.
    """
    cols_sel = set(df_selectivity.columns[2:])
    cols_af = set(df_affinity.columns[2:])
    if cols_sel != cols_af:
        only_sel = sorted(cols_sel - cols_af)[:5]
        only_af = sorted(cols_af - cols_sel)[:5]
        raise ValueError(
            "The selectivity and affinity files do not match.\n"
            "They must come from the same run of generate_imputed_csvs.py.\n"
            f"  selectivity: {path_sel}\n"
            f"  affinity:    {path_af}\n"
            f"  Only in selectivity: {only_sel}\n"
            f"  Only in affinity:    {only_af}"
        )
    if len(df_selectivity) != len(df_affinity):
        raise ValueError(
            f"Different number of rows: {len(df_selectivity)} vs {len(df_affinity)}"
        )


# -- COMPUTING THE CONFIDENCES -----------------------------------------------

def _c1_cell(row_idx, obp, diag_c1, is_predicted):
    """Affinity confidence of one specific cell."""
    if not is_predicted:
        return 1.0                       # purely experimental source
    if diag_c1 is None:
        return C1_FALLBACK               # no diag available: flat value
    key = (int(row_idx), str(obp))
    if key in diag_c1:
        return diag_c1[key]              # imputed cell
    return 1.0                           # it was in the original CSV -> measured


def compute_confidences(result_table, weights, target_row_idx,
                        interferent_row_idx, diag_c1, is_predicted):
    """
    Adds to result_table:
        c1, c2, c4, c5, Confidence, Conservative, Potential, Gain, Priority

    interferent_row_idx : {interferent_name: row_idx}  (or None if there are none)
    diag_c1             : {(row_idx, OBP): c1}  from tanimoto_diag_table.csv
    is_predicted        : True if the data source is the imputed matrix
    """
    df = result_table.copy()
    n_interf = int(df["N_interferents_total"].iloc[0]) if len(df) else 0

    c1_list, c2_list, c4_list, c5_list = [], [], [], []

    for _, row in df.iterrows():
        obp = row["OBP"]

        # -- c1: affinity of the target VOC against this OBP ---------------
        c1_list.append(_c1_cell(target_row_idx, obp, diag_c1, is_predicted))

        # -- c2: selectivity. measured=1 · imputed=C1_i · missing=0 --------
        if n_interf == 0:
            c2 = 0.0
        else:
            missing = set(row["Missing_interferents"])
            c2_total = 0.0
            for name, r_idx in (interferent_row_idx or {}).items():
                if name in missing:
                    continue                      # missing -> adds 0
                if not is_predicted:
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

        # -- c4: stability. Depends only on whether we know the OBP type ---
        c4_list.append(1.0 if row["has_s4_data"] else 0.0)

        # -- c5: promiscuity. Support based on how many competitors exist --
        # A boolean (has_s5_data) treats 2 competitors the same as 100. We use
        # the same "support" as c1 (1 - 1/(1+n)): confidence grows with
        # n_competitors regardless of whether s5 turns out good or bad - an
        # s5=0.99 with 2 competitors is far less reliable than with 100.
        n_comp = int(row["N_competitors_s5"])
        c5_list.append(1.0 - 1.0 / (1.0 + n_comp))

    df["c1"] = c1_list
    df["c2"] = c2_list
    df["c4"] = c4_list
    df["c5"] = c5_list

    # -- Overall confidence: same weights as the score --------------------
    w = weights
    df["Confidence"] = (
        w["w_affinity"] * df["c1"] +
        w["w_selectivity"] * df["c2"] +
        w["w_stability"] * df["c4"] +
        w["w_promiscuity"] * df["c5"]
    ).clip(0.0, 1.0)

    # -- The three numbers. NEVER merged into a single one ----------------
    # Conservative: discounts the score by how much you can trust it.
    # It is a product because it is an expected value: outcome x probability.
    df["Conservative"] = df["Score"] * df["Confidence"]
    df["Potential"] = df["Score"]
    df["Gain"] = df["Potential"] - df["Conservative"]

    # Priority of the pending work: is the candidate worth it (Potential) AND
    # would I learn something by measuring it (Gain)?
    df["Priority"] = df["Gain"] * df["Potential"]

    return df.sort_values("Conservative", ascending=False).reset_index(drop=True)


# -- OUTPUT ------------------------------------------------------------------

def _color(conf):
    if conf >= CONF_HIGH:
        return "green"
    if conf >= CONF_MID:
        return "yellow"
    return "red"


def _fmt_sub(value, width=5, nd=3):
    """Sub-score cell: a missing value prints as '-', never as 'nan'.
    s4 is NaN whenever the OBP type is unknown."""
    if pd.isna(value):
        return f"{'-':>{width}}"
    return f"{value:>{width}.{nd}f}"


def show_unified_table(df, how_many, is_predicted=False, min_conf=None):
    """
    Replaces Tables 1 and 2. A single list: nobody is hidden.

    The old tables split the candidates on a boolean criterion (do I have ALL
    the interferents, yes/no). With a continuous confidence that split is far
    too coarse - and it left out the best measured binder just because one
    interferent was missing.
    """
    if df.empty:
        print("\n  No candidate with data.")
        return df

    view = df if min_conf is None else df[df["Confidence"] >= min_conf]
    if view.empty:
        print(f"\n  No candidate exceeds confidence {min_conf:.2f}.")
        return view


    # Every per-criterion confidence is shown, so the reader can see WHICH
    # criterion the global Confidence is losing evidence on. c1 only appears
    # for predicted data, where it is the one that moves.
    if is_predicted:
        hdr = (f"  {'#':>3}  {'OBP':<18} {'Conserv.':>9} {'Potent.':>8} {'Gain':>7} "
               f"{'Conf.':>6} {'c1':>5} {'c2':>5} {'c4':>5} {'c5':>5}  "
               f"{'s1':>5} {'s2':>5} {'s4':>5} {'s5':>5} {'Ki(uM)':>8} "
               f"{'Interf':>7}  {'Type':<12}")
    else:
        hdr = (f"  {'#':>3}  {'OBP':<18} {'Conserv.':>9} {'Potent.':>8} {'Gain':>7} "
               f"{'Conf.':>6} {'c2':>5} {'c4':>5} {'c5':>5}  "
               f"{'s1':>5} {'s2':>5} {'s4':>5} {'s5':>5} {'Ki(uM)':>8} "
               f"{'Interf':>7}  {'Type':<12}")
    # One width for every rule on this block, taken from the widest line.
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(f"  CANDIDATES - single list, sorted by conservative Score - top {how_many}")
    print(f"  Conservative = Score x Confidence   ·   Potential = Score if everything were measured")
    print(sep)
    print(hdr)
    print(sep)

    for pos, (_, r) in enumerate(view.head(how_many).iterrows()):
        n_tot = int(r["N_interferents_total"])
        n_miss = int(r["N_interferents_missing"])
        interf = f"{n_tot - n_miss}/{n_tot}" if n_tot else "  -"
        mark = {"green": "*", "yellow": "~", "red": "!"}[_color(r["Confidence"])]
        subs = (f"{r['s1_affinity']:>5.3f} {_fmt_sub(r['s2_selectivity'])} "
                f"{_fmt_sub(r['s4_stability'])} {_fmt_sub(r['s5_promiscuity'])}")
        if is_predicted:
            print(f"  {pos+1:>3}. {str(r['OBP'])[:18]:<18} {r['Conservative']:>9.4f} "
                  f"{r['Potential']:>8.4f} {r['Gain']:>7.4f} "
                  f"{r['Confidence']:>5.2f}{mark} {r['c1']:>5.2f} {r['c2']:>5.2f} "
                  f"{r['c4']:>5.2f} {r['c5']:>5.2f}  {subs} "
                  f"{r['Ki_target_uM']:>8.2f} {interf:>7}  {str(r['Type']):<12}")
        else:
            print(f"  {pos+1:>3}. {str(r['OBP'])[:18]:<18} {r['Conservative']:>9.4f} "
                  f"{r['Potential']:>8.4f} {r['Gain']:>7.4f} "
                  f"{r['Confidence']:>5.2f}{mark} {r['c2']:>5.2f} "
                  f"{r['c4']:>5.2f} {r['c5']:>5.2f}  {subs} "
                  f"{r['Ki_target_uM']:>8.2f} {interf:>7}  {str(r['Type']):<12}")

    print(sep)
    print(f"  * conf >= {CONF_HIGH:.2f} (reliable)   "
          f"~ {CONF_MID:.2f}-{CONF_HIGH:.2f} (partial)   "
          f"! < {CONF_MID:.2f} (little evidence)")
    print(f"  c1 affinity   c2 selectivity   c4 stability   c5 promiscuity "
          f"(evidence behind each sub-score, not its value)")
    if is_predicted:
        print(f"  c1 < 1.00 -> the affinity is PREDICTED (Tanimoto), not measured.")
    print(f"  Total candidates: {len(df)}")
    return view


def show_worklist(df, how_many=25):
    """
    Table 3 - pending work, sorted by Gain x Potential.

    Sorting by score would be suboptimal: a candidate that already has 4/5
    interferents has little to gain from the last measurement. A candidate in
    the dark (1/5) with good affinity, on the other hand, can jump a long way
    up - or sink. Every measurement there is hugely informative.

    Potential keeps you from chasing candidates with a low ceiling: even if you
    measured them all, they would never be good.
    """
    pending = df[df["N_interferents_missing"] > 0].copy()
    if pending.empty:
        sep = "-" * 78
        print(f"\n{sep}")
        print(f"  TABLE 3 - Pending work")
        print(f"  No OBP/interferent pair left to compute.")
        print(sep)
        return

    pending = pending.sort_values("Priority", ascending=False)

    sep = "-" * 90
    print(f"\n{sep}")
    print(f"  TABLE 3 - Pending work, sorted by PRIORITY = Gain x Potential")
    print(f"  Spend the effort where you cut the most ignorance (with a good ceiling)")
    print(sep)
    print(f"  {'#':>3}  {'OBP':<20} {'Prior.':>7} {'Gain':>7} {'Potent.':>8}  "
          f"{'Missing interferent':<28}")
    print(sep)

    i = 0
    for _, r in pending.iterrows():
        for interf in sorted(r["Missing_interferents"]):
            i += 1
            if i > how_many:
                break
            print(f"  {i:>3}. {r['OBP']:<20} {r['Priority']:>7.4f} "
                  f"{r['Gain']:>7.4f} {r['Potential']:>8.4f}  {interf:<28}")
        if i > how_many:
            break

    total = int(pending["N_interferents_missing"].sum())
    print(sep)
    print(f"  Total OBP/interferent pairs to compute: {total}")

def show_dangerous_vineyard_interferents(df, how_many, margin, max_names):
    """s2 alert - one line per candidate with the measured vineyard interferents
    that bind as strongly, more strongly, or a bit more weakly than the target."""
    if 'Danger_interferents' not in df.columns:
        return

    top = list(df.head(how_many).iterrows())

    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  s2 ALERT - vineyard interferents that hurt the selectivity")
    print(f"  (Ki_interf <= Ki_target x {margin:.2f})   v = binds more strongly than the target")
    print(sep)

    none_found = True
    for pos, (_, r) in enumerate(top):
        danger = r['Danger_interferents']
        if not danger:
            continue
        none_found = False
        ki_d = r['Ki_target_uM']

        shown = danger if max_names is None else danger[:max_names]
        labels = []
        for name, ki_i in shown:
            fl = "v" if ki_i < ki_d and not np.isclose(ki_i, ki_d) else ""
            labels.append(f"{name}({ki_i:.1f}{fl})")
        extra = (f" +{len(danger) - max_names}"
                 if max_names is not None and len(danger) > max_names else "")

        print(f"  {pos+1:>3}. {r['OBP']:<18} Ki={ki_d:>6.2f} [{r['N_danger_interferents']}]  "
              + ", ".join(labels) + extra)

    if none_found:
        print(f"  OK No candidate in the top list has dangerous vineyard interferents.")
    print(sep)
    print(f"  v = stronger than the target (worse)   [n] = total number of dangerous ones")


def show_superbinders(df, how_many, ki_max, max_names):
    """s5 alert - same format as the vineyard alert but over the WHOLE database:
    VOCs (any, not only vineyard) each OBP binds strongly.
    * = Ki <= ki_max (strong superbinder)   v = binds stronger than the target."""
    if 'Superbinders' not in df.columns:
        return

    top = list(df.head(how_many).iterrows())

    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  s5 ALERT - superbinders (VOCs from the whole database bound strongly)")
    print(f"  * = Ki <= {ki_max:.1f} uM (strong)   v = binds more strongly than the target")
    print(sep)

    none = True
    for pos, (_, r) in enumerate(top):
        supers = r['Superbinders']
        if isinstance(supers, float):
            supers = []
        if not supers:
            continue
        none = False
        shown = supers if max_names is None else supers[:max_names]
        labels = []
        for name, ki_v, is_strong, beats in shown:
            mark = ("*" if is_strong else "") + ("v" if beats else "")
            labels.append(f"{name}({ki_v:.1f}{mark})")
        extra = (f"  +{len(supers) - max_names}"
                 if max_names is not None and len(supers) > max_names else "")
        print(f"  {pos+1:>3}. {str(r['OBP'])[:18]:<18} [{len(supers)}]  "
              + ", ".join(labels) + extra)

    if none:
        print(f"  OK  No candidate in the top binds any strong superbinder.")
    print(sep)
    print(f"  * strong (Ki <= {ki_max:.1f})   v stronger than target   [n] = total superbinders")
