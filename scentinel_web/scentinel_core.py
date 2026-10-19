"""
scentinel_core.py - glue between the web pages and the ORIGINAL tool code.

The scientific code in tools/ is copied unchanged from the team repositories:
    tools/obp_searcher/   main.py, confidence.py, SMAAA.py, generate_report.py
    tools/panel_selector/ panel_selector/ package, the two command-line tools and
                          generate_panel_report.py (panel_selector_project, final version)

This module only:
  * loads the shared data once (cached between visitors),
  * replaces the interactive input() prompts with function arguments,
  * captures the console printing so it does not flood the server logs,
  * returns pandas tables the pages can show.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
for sub in ("obp_searcher", "panel_selector"):
    p = os.path.join(ROOT, "tools", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import main as scentinel                       # noqa: E402  (tools/obp_searcher/main.py)
import confidence as cf                        # noqa: E402
from SMAAA import smaa_full                    # noqa: E402
import panel_selector as ps                    # noqa: E402

EXP_FILE = os.path.join(DATA, scentinel.BINDING_FILE_EXPERIMENTAL)
LOWER_FILE = os.path.join(DATA, scentinel.BINDING_FILE_PREDICTED)
PRED_FILE = os.path.join(DATA, scentinel.BINDING_FILE_PREDICTED_AFFINITY)
INFO_FILE = os.path.join(DATA, scentinel.INFO_FILE)
DIAG_FILE = os.path.join(DATA, "results", "tanimoto_diag_table.csv")
SMAA_CHUNK = 1000   # SMAAA.smaa_full batch size: same results, ~60x less memory than the default 20 000
EXAMPLES = os.path.join(DATA, "examples")


@contextlib.contextmanager
def quiet():
    """Swallow the console output of the original scripts."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


# -- Non-interactive compound lookup ----------------------------------------
# main.validate_voc() calls input() when a name has several partial matches.
# A web server has no keyboard, so inside the web app it is replaced by this
# version: same priority (CAS, exact name), plus exact '/' synonym, plus a
# UNIQUE partial match; anything else returns None instead of asking.
_original_validate_voc = scentinel.validate_voc


def validate_voc_web(binding_table, name_col, cas_col, search_text, file_path=None):
    q = str(search_text).strip()
    cas = binding_table[cas_col].astype(str).str.strip()
    hit = binding_table[cas == q]
    if len(hit):
        return hit.iloc[0]
    names = binding_table[name_col].astype(str).str.strip()
    hit = binding_table[names.str.lower() == q.lower()]
    if len(hit):
        return hit.iloc[0]
    syn = names.map(lambda n: q.lower() in [p.strip().lower() for p in n.split("/")])
    if syn.any():
        return binding_table[syn].iloc[0]
    if not re.match(r"^\d+-\d+-\d+$", q):
        part = binding_table[names.str.contains(q, case=False, regex=False, na=False)]
        if len(part) == 1:
            return part.iloc[0]
    return None


scentinel.validate_voc = validate_voc_web


def canonical_name(full_name: str) -> str:
    """Shortest '/' synonym - exactly what main.validate_interferent_list keeps."""
    parts = [p.strip() for p in str(full_name).split("/")]
    return min(parts, key=len)


# -- Shared data (loaded once per server process) ---------------------------

def load_binding(path):
    with quiet():
        return scentinel.load_csv_files(path, INFO_FILE)


class Store:
    """Lazy, process-wide cache of every table the pages need."""
    _cache: dict = {}

    @classmethod
    def get(cls, key, fn):
        if key not in cls._cache:
            cls._cache[key] = fn()
        return cls._cache[key]


def experimental():
    return Store.get("exp", lambda: load_binding(EXP_FILE))


def predicted_lower():
    return Store.get("lower", lambda: load_binding(LOWER_FILE))


def predicted_affinity():
    return Store.get("pred", lambda: load_binding(PRED_FILE))


def diag_c1():
    return Store.get("diag_c1", lambda: cf.load_diag_c1(DIAG_FILE))


def compound_catalogue() -> pd.DataFrame:
    """CAS + name of every compound, with a label for drop-down menus."""
    def build():
        bt, _, cas_col, name_col, obps, _, _ = experimental()
        df = pd.DataFrame({"CAS": bt[cas_col].astype(str).str.strip(),
                           "Name": bt[name_col].astype(str).str.strip()})
        df["Short"] = df["Name"].map(canonical_name)
        df["n_measured"] = bt[obps].notna().sum(axis=1).values
        df["label"] = df["Name"] + " (CAS " + df["CAS"] + ")"
        return df
    return Store.get("catalogue", build)


def obp_list():
    return list(experimental()[4])


# -- Weights (same rules as main.ask_user_for_weights) ----------------------

def default_weights(lock_selectivity_zero: bool):
    dw = scentinel.DEFAULT_WEIGHTS
    if not lock_selectivity_zero:
        return dict(dw)
    rest = dw["w_affinity"] + dw["w_stability"] + dw["w_promiscuity"]
    return {"w_affinity": dw["w_affinity"] / rest, "w_selectivity": 0.0,
            "w_stability": dw["w_stability"] / rest, "w_promiscuity": dw["w_promiscuity"] / rest}


def normalize_weights(custom: dict, lock_selectivity_zero: bool):
    """Copy of the custom-weight branch of main.ask_user_for_weights."""
    custom = dict(custom)
    if lock_selectivity_zero:
        custom["w_selectivity"] = 0.0
    total = sum(custom.values())
    if total == 0:
        return default_weights(lock_selectivity_zero)
    normalized = {k: v / total for k, v in custom.items()}
    mean_sp = (normalized["w_stability"] + normalized["w_promiscuity"]) / 2.0
    normalized["w_stability"] = mean_sp
    normalized["w_promiscuity"] = mean_sp
    return normalized


# -- TOOL: OBP finder (SCENTINEL main.py) -----------------------------------

def resolve_interferents(queries, binding_table, name_col, cas_col):
    """Free-text names/CAS -> (canonical names, errors). Never stops the program."""
    names, errors = [], []
    for q in queries:
        row = validate_voc_web(binding_table, name_col, cas_col, q)
        if row is None:
            errors.append(q)
        else:
            n = canonical_name(row[name_col])
            if n not in names:
                names.append(n)
    return names, errors


def run_obp_ranking(target_cas, interferent_names, weights, tau, source="experimental"):
    """Exactly the computation of main.main() up to the mode selection."""
    bt, info, cas_col, name_col, obps, lit_min, lit_max = experimental()
    chosen = validate_voc_web(bt, name_col, cas_col, target_cas)
    if chosen is None:
        raise ValueError(f"Target {target_cas} not found")
    target_name = chosen[name_col]

    experimental_ki = None
    affinity_ki = None
    if source == "predicted":
        sel_bt, _, _, _, _, _, _ = predicted_lower()
        row_sel = validate_voc_web(sel_bt, name_col, cas_col, chosen[cas_col])
        ki_target = pd.Series({c: row_sel[c] for c in obps})
        experimental_ki = pd.Series({c: chosen[c] for c in obps})
        aff_bt, _, _, _, _, _, _ = predicted_affinity()
        row_aff = validate_voc_web(aff_bt, name_col, cas_col, chosen[cas_col])
        if row_aff is not None:
            affinity_ki = pd.Series({c: row_aff[c] for c in obps})
        table_for_ranking = sel_bt
        target_row_idx = row_sel.name
    else:
        ki_target = pd.Series({c: chosen[c] for c in obps})
        table_for_ranking = bt
        target_row_idx = chosen.name

    with quiet():
        result = scentinel.build_obp_ranking(
            ki_values_target=ki_target, obp_info_table=info, binding_table=table_for_ranking,
            cas_col=cas_col, name_col=name_col, interferent_list=list(interferent_names),
            obp_name_list=obps, weights=weights, ki_min_matrix=lit_min, ki_max_matrix=lit_max,
            tau=tau, experimental_ki_values_target=experimental_ki, binding_source_label=source,
            ki_values_target_affinity=affinity_ki, target_name=target_name)
        if result.empty:
            return result, target_name
        result = cf.compute_confidences(
            result, weights, target_row_idx=target_row_idx,
            interferent_row_idx=result.attrs.get("interferent_row_idx", {}),
            diag_c1=diag_c1(), is_predicted=(source == "predicted"))
    return result, target_name


def smaa_inputs(result, weights):
    """Criterion set for Modes 2/3, as in main.main()."""
    names = ["Affinity", "Stability", "Promisc."]
    base = [weights["w_affinity"], weights["w_stability"], weights["w_promiscuity"]]
    cols = ["s1_affinity", "s4_stability", "s5_promiscuity"]
    has = [np.ones(len(result), bool), np.ones(len(result), bool), result["has_s5_data"].values]
    if weights["w_selectivity"] > 0.0:
        names.insert(1, "Selectiv.")
        base.insert(1, weights["w_selectivity"])
        cols.insert(1, "s2_selectivity")
        has.insert(1, result["has_s2_data"].values)
    return names, base, result[cols].values, np.column_stack(has)


def run_smaa(result, weights, mode, n_iter, c=50.0):
    names, base, sub, has = smaa_inputs(result, weights)
    with quiet():
        if mode == "perturbation":
            ra, _, cw = smaa_full(subscores=sub, has_data=has, n_iter=n_iter,
                                  base_weights=base, c=c, mode="perturbation", seed=42, chunk_size=SMAA_CHUNK)
            cw = None
        else:
            ra, _, cw = smaa_full(subscores=sub, has_data=has, n_iter=n_iter, mode="smaa", seed=42, chunk_size=SMAA_CHUNK)
    return names, ra, cw


def build_pdf_report(target_cas, interferent_names, weights, tau, how_many, n_iter, c_val):
    """The PDF of generate_report.main(), written to memory instead of disk."""
    import generate_report as gr
    from matplotlib.backends.backend_pdf import PdfPages

    bt, info, cas_col, name_col, obps, lit_min, lit_max = experimental()
    chosen = validate_voc_web(bt, name_col, cas_col, target_cas)
    voc_display, voc_cas = chosen[name_col], str(chosen[cas_col])
    with quiet():
        result = scentinel.build_obp_ranking(
            ki_values_target=pd.Series({c: chosen[c] for c in obps}), obp_info_table=info,
            binding_table=bt, cas_col=cas_col, name_col=name_col,
            interferent_list=list(interferent_names), obp_name_list=obps, weights=weights,
            ki_min_matrix=lit_min, ki_max_matrix=lit_max, tau=tau, target_name=voc_display)
        if result.empty:
            raise ValueError("No OBP has experimental data for this VOC.")
        how_many = min(how_many, len(result))
        result = cf.compute_confidences(
            result, weights, target_row_idx=chosen.name,
            interferent_row_idx=result.attrs.get("interferent_row_idx", {}),
            diag_c1=diag_c1(), is_predicted=False)
        base = [weights["w_affinity"], weights["w_selectivity"],
                weights["w_stability"], weights["w_promiscuity"]]
        sub = result[["s1_affinity", "s2_selectivity", "s4_stability", "s5_promiscuity"]].values
        has = np.column_stack([np.ones(len(result), bool), result["has_s2_data"].values,
                               np.ones(len(result), bool), result["has_s5_data"].values])
        ra_d, _, _ = smaa_full(subscores=sub, has_data=has, n_iter=n_iter, base_weights=base,
                               c=c_val, mode="perturbation", seed=42, chunk_size=SMAA_CHUNK)
        ra_s, _, central_w = smaa_full(subscores=sub, has_data=has, n_iter=n_iter, mode="smaa", seed=42, chunk_size=SMAA_CHUNK)

        pred_bt, _, _, _, _, _, _ = predicted_lower()
        aff_bt, _, _, _, _, _, _ = predicted_affinity()
        row_p = validate_voc_web(pred_bt, name_col, cas_col, voc_cas)
        row_a = validate_voc_web(aff_bt, name_col, cas_col, voc_cas)
        pred_result = scentinel.build_obp_ranking(
            ki_values_target=pd.Series({c: (row_p if row_p is not None else chosen)[c] for c in obps}),
            obp_info_table=info, binding_table=pred_bt, cas_col=cas_col, name_col=name_col,
            interferent_list=list(interferent_names), obp_name_list=obps, weights=weights,
            ki_min_matrix=lit_min, ki_max_matrix=lit_max, tau=tau,
            experimental_ki_values_target=pd.Series({c: chosen[c] for c in obps}),
            binding_source_label="predicted",
            ki_values_target_affinity=(pd.Series({c: row_a[c] for c in obps}) if row_a is not None else None),
            target_name=voc_display)

        ts_human = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            gr.page_cover(pdf, voc_display, voc_cas, weights, tau, how_many,
                          list(interferent_names), n_iter, c_val, ts_human)
            gr.page_detailed_table(pdf, result, voc_display, how_many)
            if not pred_result.empty:
                gr.page_exp_vs_comp(pdf, result, pred_result, voc_display, how_many)
            else:
                gr.page_missing_predicted(pdf, voc_display, "x Could not generate the predicted ranking.")
            gr.page_smaa(pdf, result, ra_d, None, voc_display, how_many, "Method 2 - Weight Perturbation")
            gr.page_smaa(pdf, result, ra_s, central_w, voc_display, how_many, "Method 3 - Pure SMAA")
            gr.page_comparison(pdf, result, ra_d, ra_s, voc_display, how_many)
    safe = re.sub(r"[^\w]+", "_", voc_display)[:40]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return buf.getvalue(), f"SCENTINEL_report_{safe}_{stamp}.pdf"


# -- TOOLS: Proteus (specificity) and Dionysus (discrimination) -------------
# Both come from panel_selector_project (final version). The analysis functions
# analyse_tool1() / analyse_tool2_summary() of the command-line scripts are
# called directly, so the web shows exactly what the terminal and the PDF show.

def panel_data():
    return Store.get("panel_data", lambda: ps.ScentinelData.load(DATA, diagnostic=DIAG_FILE))


import scentinel_tool1_specificity as tool1     # noqa: E402
import scentinel_tool2_discrimination as tool2  # noqa: E402
import generate_panel_report as panel_report    # noqa: E402
from types import SimpleNamespace                # noqa: E402

PROTEUS_DEFAULTS = dict(noise=0.05, min_c2=0.5, max_sensors=8, min_gain=1e-3, lookahead=3, cover=1,
                        min_confidence=0.3, priority="auto", mc=0)
DIONYSUS_DEFAULTS = dict(max_error=0.05, sensor_noise=0.05, n=3, max_sensors=8, scale="anchored",
                         censored="zero", min_confidence=0.3, metric="fisher")


def proteus_opts(source, **kw):
    o = dict(PROTEUS_DEFAULTS, **{k: v for k, v in kw.items() if v is not None})
    return SimpleNamespace(min_confidence=o["min_confidence"] if source == "predicted" else 0.0,
                           only_confident=False, min_c2=o["min_c2"], priority=o["priority"],
                           noise=o["noise"], min_gain=o["min_gain"], max_sensors=o["max_sensors"],
                           lookahead=o["lookahead"], cover=o["cover"], mc=o["mc"])


def dionysus_opts(source, **kw):
    o = dict(DIONYSUS_DEFAULTS, **{k: v for k, v in kw.items() if v is not None})
    return SimpleNamespace(min_confidence=o["min_confidence"] if source == "predicted" else 0.0,
                           only_confident=False, censored=o["censored"], scale=o["scale"], ki_min=1.0,
                           ki_max=40.0, sensor_noise=o["sensor_noise"], n=int(o["n"]), min_gain=None,
                           target_score=None, max_sensors=o["max_sensors"], metric=o["metric"],
                           max_error=o["max_error"], weak_frac=None, show_bets=10)


_HEAVY = ("S", "res", "c2", "A", "B")


def _light(run):
    """Drop the big matrices before a run goes into a visitor's session."""
    return {k: v for k, v in run.items() if k not in _HEAVY}


def run_proteus(target, interferents, source="experimental", start=(), **kw):
    """Tool 1 exactly as scentinel_tool1_specificity.analyse_tool1().
    target / interferents are compound labels of panel_data()."""
    D = panel_data()
    interferents = [i for i in interferents if i != target]
    run = tool1.analyse_tool1(D, target, interferents, source, list(start), proteus_opts(source, **kw))
    run["target_short"] = D.short(target)
    run["interf_short"] = [D.short(i) for i in interferents]
    return _light(run)


def tool2_is_summary(path):
    return tool2.is_input_table(path)


def run_dionysus_summary(path, source="experimental", **kw):
    """Tool 2 (summary table: mean +/- SE per class) as analyse_tool2_summary()."""
    D = panel_data()
    opts = dionysus_opts(source, **kw)
    A, B, rep = D.load_input(path, n=opts.n)
    run = tool2.analyse_tool2_summary(D, source, A, B, opts)
    run["report"] = rep
    run["mode"] = "summary"
    run["comps_short"] = [D.short(c) for c in run["comps"]]
    return _light(run)


def run_dionysus_profiles(path, source="experimental", class_col="class", classes=None, **kw):
    """Tool 2 on per-sample profiles, with the same uncertainty check as the terminal."""
    D = panel_data()
    opts = dionysus_opts(source, **kw)
    groups, errors = D.load_profiles(path, class_col=class_col)
    if errors:
        raise ValueError("Some columns are not in the database:\n" + "\n".join(errors))
    names = list(classes) if classes else list(groups)[:2]
    if len(names) < 2 or any(n not in groups for n in names):
        raise ValueError(f"Need two classes; found {list(groups)}")
    A, B = groups[names[0]], groups[names[1]]
    comps = list(A.columns)
    kind = "experimental" if source == "experimental" else "pred"
    S = D.sensitivity(kind, opts.min_confidence, opts.only_confident, censored=opts.censored,
                      scale=opts.scale, ki_min=opts.ki_min, ki_max=opts.ki_max)
    n_have = (S.S[comps].fillna(0) > 0).sum(axis=1)
    candidates = [o for o in S.obps if n_have[o] > 0]
    if not candidates:
        raise ValueError("No OBP has data for these compounds.")
    first = max(ps.fisher_ratio(ps.project_profiles(S, A, [o]), ps.project_profiles(S, B, [o])) for o in candidates)
    min_gain = 0.01 * first if np.isfinite(first) and first > 0 else 1e-3
    with quiet():
        res = ps.greedy_panel_for_discrimination(
            S, class_a_profiles=A, class_b_profiles=B, metric="fisher", candidates=candidates,
            min_gain=min_gain, max_sensors=opts.max_sensors or None, tie_break=lambda o: float(n_have[o]))
    steps = pd.DataFrame([{"step": k, "OBP": st.added, "J_panel": st.score, "misclassified": ps.error_from_j(st.score),
                           "gain": st.gain if np.isfinite(st.gain) else np.nan,
                           "n_measured": int(sum(D.decision.loc[c, st.added] == "measured" for c in comps))}
                          for k, st in enumerate(res.result.trace, 1)])
    shift = (B.mean() - A.mean()) / np.sqrt((A.var(ddof=1) + B.var(ddof=1)).replace(0, np.nan))
    J = ps.replicate_j_function(A, B, comps)
    unc, J0, thr, J_all = tool2.uncertainty_check(opts, D, S, kind, res.panel, comps, J, shift.to_dict())
    grade, why = tool2.grade_tool2(unc, J0, thr[0], J_all)
    known = pd.DataFrame([{"OBP": o, "measured": int(sum(D.decision.loc[c, o] == "measured" for c in comps)),
                           "predicted_used": int(sum(D.decision.loc[c, o] != "measured" and pd.notna(S.S.loc[o, c])
                                                     for c in comps))} for o in res.panel])
    known["missing"] = len(comps) - known["measured"] - known["predicted_used"]
    return dict(mode="profiles", source=source, comps=comps, comps_short=[D.short(c) for c in comps],
                shift=shift, n_candidates=len(candidates), panel=res.panel, steps=steps,
                stop_reason=res.stop_reason, bets=unc, J0=J0, thr=thr, J_all=J_all, grade=grade, why=why,
                known=known, classes=names, sizes=(len(A), len(B)))


def build_panel_pdf(target, interferents, start, profiles_path=None, proteus=None, dionysus=None):
    """The PDF of generate_panel_report.py (Proteus in both data modes, Dionysus in both
    modes if a summary table is given), written to memory."""
    from matplotlib.backends.backend_pdf import PdfPages
    D = panel_data()
    proteus, dionysus = proteus or {}, dionysus or {}
    interferents = [i for i in interferents if i != target]
    runs1, runs2 = {}, {}
    for src in ("experimental", "predicted"):
        try:
            r = tool1.analyse_tool1(D, target, interferents, src, list(start), proteus_opts(src, **proteus))
            r["target_short"] = D.short(target)
            runs1[src] = r
        except tool1.NotEnoughCandidates as e:
            runs1[src] = str(e)
    if profiles_path and tool2.is_input_table(profiles_path):
        n = int(dionysus.get("n", DIONYSUS_DEFAULTS["n"]))
        A, B, _ = D.load_input(profiles_path, n=n)
        for src in ("experimental", "predicted"):
            try:
                runs2[src] = tool2.analyse_tool2_summary(D, src, A, B, dionysus_opts(src, **dionysus))
            except RuntimeError as e:
                runs2[src] = str(e)
    p_o = proteus_opts("predicted", **proteus)
    d_o = dionysus_opts("predicted", **dionysus)
    now = datetime.datetime.now()
    ctx = dict(target_short=D.short(target), cas=D.meta.set_index("label").loc[target, "CAS"],
               start=list(start) or ["(none: chosen by the search)"],
               interf_short=[D.short(i) for i in interferents],
               profiles_name=os.path.basename(profiles_path) if profiles_path and runs2 else None,
               noise=p_o.noise, min_conf=p_o.min_confidence, max_error=d_o.max_error,
               sensor_noise=d_o.sensor_noise)
    labels = {"experimental": "Experimental data", "predicted": f"Predicted data (conf >= {p_o.min_confidence})"}
    pr = panel_report
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pr.page_cover(pdf, ctx, runs1, runs2, now.strftime("%d/%m/%Y %H:%M"))
        for src in ("experimental", "predicted"):
            r = runs1[src]
            if isinstance(r, str):
                pr._page_unavailable(pdf, f"Tool 1 · {labels[src]} · not available", r)
                continue
            pr.page_t1_panel(pdf, r, labels[src], D)
            if r["bets"] is not None:
                pr.page_t1_bets(pdf, r, labels[src])
        pr.page_t1_compare(pdf, runs1)
        for src in ("experimental", "predicted"):
            r = runs2.get(src)
            if r is None:
                continue
            if isinstance(r, str):
                pr._page_unavailable(pdf, f"Tool 2 · {labels[src]} · not available", r)
                continue
            pr.page_t2_panel(pdf, r, labels[src], D)
            pr.page_t2_bets(pdf, r, labels[src])
        pr.page_lab_priorities(pdf, runs1, runs2)
        pr.page_method(pdf)
    safe = re.sub(r"[^\w]+", "_", D.short(target))[:40]
    return buf.getvalue(), f"SCENTINEL_panel_report_{safe}_{now:%Y-%m-%d_%H-%M-%S}.pdf"
