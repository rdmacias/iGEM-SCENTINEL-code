import numpy as np
import pandas as pd
import streamlit as st

import scentinel_core as sc
import ui

ui.page_header("diana", "Diana", "Rank candidate OBPs by predicted binding to a single target VOC "
                                 "(SCENTINEL v3 scoring: affinity, selectivity, stability, promiscuity).")


@st.cache_resource(show_spinner="Loading the binding database…")
def load():
    sc.experimental(); sc.predicted_lower(); sc.predicted_affinity(); sc.diag_c1()
    return sc.compound_catalogue()


cat = load()
bt, _, cas_col, name_col, _, _, _ = sc.experimental()
labels = cat["label"].tolist()
label_to_cas = dict(zip(cat["label"], cat["CAS"]))
cfg = ui.settings()["diana"]


def labels_from_queries(queries):
    found, bad = [], []
    for q in queries:
        row = sc.validate_voc_web(bt, name_col, cas_col, q)
        if row is None:
            bad.append(q)
        else:
            lab = cat.loc[row.name, "label"]
            if lab not in found:
                found.append(lab)
    return found, bad


left, right = st.container(), st.container()  # parameters on top, results below

# ================================================================ parameters
with left.container(border=True):
    st.markdown('<div class="scn-h3">Parameters</div>', unsafe_allow_html=True)
    default_target = next((l for l in labels if l.startswith("methyl salicylate")), None)
    target_label = st.selectbox("Target VOC", labels,
                                index=labels.index(default_target) if default_target else None,
                                placeholder="Name or CAS, e.g. 119-36-8")
    if "finder_interf" not in st.session_state:
        st.session_state.finder_interf = []
    if "finder_pending" in st.session_state:  # list loaded below; must be set before the widget exists
        st.session_state.finder_interf = st.session_state.pop("finder_pending")
    interf_labels = st.multiselect("Interfering VOCs (optional)", labels, key="finder_interf",
                                   placeholder="Search and add compounds…")
    queries = ui.voc_list_input("finder", sc.EXAMPLES)
    if queries is not None:
        found, bad = labels_from_queries(queries)
        st.session_state.finder_pending = found
        if bad:
            st.session_state.finder_bad = bad
        st.rerun()
    if st.session_state.get("finder_bad"):
        st.warning("Not found (ignored): " + ", ".join(st.session_state.pop("finder_bad")))
    source = st.radio("Binding data", ["experimental", "predicted"],
                      format_func=lambda s: {"experimental": "Experimental only",
                                             "predicted": "Experimental + Tanimoto"}[s])
    has_interf = bool(interf_labels)
    with st.expander("Weights, τ and robustness"):
        dw = sc.default_weights(lock_selectivity_zero=not has_interf)
        use_default = st.checkbox("Default weights", value=True)
        if use_default:
            weights = dw
        else:
            raw = {"w_affinity": st.slider("Affinity s1", 0.0, 1.0, float(round(dw["w_affinity"], 2)), 0.05),
                   "w_selectivity": st.slider("Selectivity s2", 0.0, 1.0, float(round(dw["w_selectivity"], 2)), 0.05,
                                              disabled=not has_interf),
                   "w_stability": st.slider("Stability s4", 0.0, 1.0, float(round(dw["w_stability"], 2)), 0.05),
                   "w_promiscuity": st.slider("Promiscuity s5", 0.0, 1.0, float(round(dw["w_promiscuity"], 2)), 0.05)}
            weights = sc.normalize_weights(raw, lock_selectivity_zero=not has_interf)
        st.caption("Weights: " + ", ".join(f"{k[2:]} {v:.2f}" for k, v in weights.items())
                   + ("  (no interferents: selectivity locked to 0)" if not has_interf else ""))
        tau = st.slider("τ selectivity margin", 1.0, 100.0, float(cfg["tau"]), 1.0,
                        help="s2 = min(1, Ki_interferent / (Ki_target × τ)). 1 = lax, 10 = recommended, 50 = strict.")
        how_many = st.slider("Candidates to show", 5, 50, int(cfg["how_many"]))
        mode = st.radio("Analysis mode", ["deterministic", "perturbation", "smaa"],
                        format_func=lambda m: {"deterministic": "Deterministic",
                                               "perturbation": "Weight perturbation",
                                               "smaa": "Pure SMAA"}[m],
                        help="Perturbation and SMAA test how robust the ranking is when the weights change.")
        n_iter, c_val = int(cfg["n_iter"]), float(cfg["c"])
        if mode != "deterministic":
            n_iter = st.select_slider("Iterations", [10_000, 20_000, 50_000, 100_000, 200_000], value=n_iter)
            if mode == "perturbation":
                c_val = st.slider("Dispersion c%", 1.0, 100.0, c_val, 1.0)
    run = st.button("Rank OBPs", type="primary", disabled=target_label is None, use_container_width=True)

# ================================================================ run
if run:
    target_cas = label_to_cas[target_label]
    names = [sc.canonical_name(cat.loc[cat["label"] == l, "Name"].iloc[0]) for l in interf_labels
             if l != target_label]
    with st.spinner("Scoring every OBP…"):
        result, tname = sc.run_obp_ranking(target_cas, names, weights, tau, source)
        smaa = None
        if not result.empty and mode != "deterministic":
            smaa = sc.run_smaa(result, weights, mode, n_iter, c_val)
    st.session_state.finder = dict(result=result, target=tname, cas=target_cas, interf=names,
                                   interf_cas=[label_to_cas[l] for l in interf_labels if l != target_label],
                                   weights=weights, tau=tau, how_many=how_many, source=source,
                                   mode=mode, smaa=smaa, n_iter=n_iter, c=c_val,
                                   dropped_target=target_label in interf_labels)
    st.session_state.pop("finder_pdf", None)
    if not result.empty:
        top0 = result.iloc[0]
        ui.log_run("Diana", sc.canonical_name(tname),
                   f"best {top0['OBP']}, confidence {top0['Confidence']:.0%}, {len(result)} OBPs",
                   state={"finder": st.session_state.finder, "finder_pdf": None})

res = st.session_state.get("finder")
with right.container(border=True):
    if run:
        ui.scroll_here()
    if not res:
        ui.card_title("Results")
        ui.empty("Choose a target VOC and press Rank OBPs.")
        st.stop()
    result = res["result"]
    ui.card_title("Results", [sc.canonical_name(res['target']),
                             f"{'experimental' if res['source'] == 'experimental' else 'experimental + Tanimoto'}"])
    if result.empty:
        ui.info_box("bad", f"No OBP has {res['source']} binding data for <b>{res['target']}</b>. "
                           "Try <b>Experimental + Tanimoto</b>.")
        st.stop()
    if res["dropped_target"]:
        st.caption("The target was also in the interferent list; it was removed from the interferents.")

    top = result.iloc[0]
    conf = top["Confidence"]
    kind = "ok" if conf >= sc.cf.CONF_HIGH else ("mid" if conf >= sc.cf.CONF_MID else "bad")
    ui.info_box(kind, f"<b>Best candidate: {ui.obp_link(top['OBP'])}</b>"
                      f"<div class='why'><span class='scn-meta'><span>Conservative score {top['Conservative']:.3f}</span>"
                      f"<span>Confidence {conf:.0%}</span><span>Ki {top['Ki_target_uM']:.2f} µM</span>"
                      f"<span>{top['Type']}</span><span><i>{top['Species']}</i></span></span></div>")
    ui.kv([("OBPs with data", len(result)), ("Interferents", len(res["interf"]) or "none"),
           ("τ", f"{res['tau']:g}"), ("Mode", res["mode"])])

    is_pred = res["source"] == "predicted"

    def conf_flag(c):
        return "reliable" if c >= sc.cf.CONF_HIGH else ("partial" if c >= sc.cf.CONF_MID else "little evidence")

    view = result.head(res["how_many"]).copy()
    n_tot = view["N_interferents_total"]
    table = pd.DataFrame({
        "#": range(1, len(view) + 1), "OBP": view["OBP"],
        "Conservative": view["Conservative"], "Potential": view["Potential"],
        "Confidence": view["Confidence"], "Evidence": view["Confidence"].map(conf_flag),
        "Ki target (µM)": view["Ki_target_uM"],
        "s1": view["s1_affinity"], "s2": view["s2_selectivity"].where(view["has_s2_data"]),
        "s4": view["s4_stability"], "s5": view["s5_promiscuity"],
        "Interf. measured": [f"{t - mi}/{t}" if t else "–" for t, mi in zip(n_tot, view["N_interferents_missing"])],
        "Type": view["Type"], "Species": view["Species"]})
    if is_pred:
        table.insert(6, "Affinity source", view["s1_source_label"])
        table.insert(7, "c1", view["c1"])

    tabs = st.tabs(["Candidates", "Interferent alerts", "Superbinders", "What to measure next"]
                   + (["Robustness (SMAA)"] if res["smaa"] else []) + ["Report & downloads"], key="diana_tabs")
    with tabs[0]:
        ui.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "Conservative": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1,
                                                            help="score × confidence: what the data supports today"),
            "Potential": st.column_config.NumberColumn(format="%.3f", help="score if every missing value were measured"),
            "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Ki target (µM)": st.column_config.NumberColumn(format="%.2f"),
            **{c: st.column_config.NumberColumn(format="%.3f") for c in ["s1", "s2", "s4", "s5", "c1"]}})
        st.markdown(f'<div class="scn-note" style="margin-bottom:22px">s1 affinity (Ki 1 µM → 1, 40 µM → 0); s2 selectivity '
                    f'min(1, Ki interferent / (Ki target × τ)), τ = {res["tau"]:g}; s4 stability by OBP family; '
                    f's5 promiscuity. Weights: ' + ", ".join(f"{k[2:]} {v:.2f}" for k, v in res["weights"].items())
                    + '. Missing criteria are dropped and the weights renormalised.</div>', unsafe_allow_html=True)
        if st.button(f"Build a specific panel from {top['OBP']} in Proteus →"):
            st.session_state.proteus_prefill = dict(target_cas=res["cas"], interf_cas=res["interf_cas"],
                                                    start=[top["OBP"]])
            st.switch_page("views/proteus.py")

    with tabs[1]:
        st.caption(f"Interferents that bind as strongly as the target, or at most "
                   f"{(sc.scentinel.DANGER_INTERFERENT_MARGIN - 1):.0%} weaker (measured data only).")
        rows = []
        for _, r in view.iterrows():
            for n, ki in r["Danger_interferents"]:
                rows.append({"OBP": r["OBP"], "Ki target (µM)": r["Ki_target_uM"], "Interferent": n,
                             "Ki interferent (µM)": ki,
                             "Binds stronger": "yes" if ki < r["Ki_target_uM"] and not np.isclose(ki, r["Ki_target_uM"]) else ""})
        if rows:
            ui.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            ui.info_box("ok", "No candidate in the list has a dangerous interferent." if res["interf"]
                        else "Add interferents to see these alerts.")

    with tabs[2]:
        st.caption(f"Compounds from the whole database each OBP binds strongly (Ki ≤ {sc.scentinel.SUPERBINDER_KI_MAX:g} µM) "
                   "or more strongly than the target. Informative only; it does not change the score.")
        rows = []
        for _, r in view.iterrows():
            s = r["Superbinders"] if isinstance(r["Superbinders"], list) else []
            if s:
                rows.append({"OBP": r["OBP"], "Count": len(s),
                             "Strongest (Ki µM)": ", ".join(f"{n} ({k:.1f}{'*' if st_ else ''}{'↑' if b else ''})"
                                                            for n, k, st_, b in s[:10])})
        if rows:
            ui.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("* strong binder; ↑ binds stronger than the target")
        else:
            ui.info_box("ok", "No superbinders among these candidates.")

    with tabs[3]:
        pending = result[result["N_interferents_missing"] > 0].sort_values("Priority", ascending=False)
        if pending.empty:
            ui.info_box("ok", "Nothing pending: every OBP/interferent pair has data." if res["interf"]
                        else "Add interferents to get a measurement worklist.")
        else:
            st.caption("OBP/interferent pairs with no data, sorted by priority = gain × potential.")
            rows = [{"OBP": r["OBP"], "Missing interferent": i, "Priority": r["Priority"],
                     "Gain": r["Gain"], "Potential": r["Potential"]}
                    for _, r in pending.iterrows() for i in sorted(r["Missing_interferents"])]
            ui.dataframe(pd.DataFrame(rows).head(100), hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ["Priority", "Gain", "Potential"]})
            st.caption(f"{len(rows)} pairs in total.")

    if res["smaa"]:
        with tabs[4]:
            names, ra, cw = res["smaa"]
            order = np.argsort(-ra[:, 0])[:res["how_many"]]
            k = min(len(result), 5)
            df = pd.DataFrame({"OBP": result.iloc[order]["OBP"].values, "P(best)": ra[order, 0],
                               **{f"P(rank {i + 1})": ra[order, i] for i in range(1, k)}})
            st.caption(f"{'Weight perturbation' if res['mode'] == 'perturbation' else 'Pure SMAA'}, "
                       f"{res['n_iter']:,} random weight sets. P(best) = how often the OBP comes first.")
            ui.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "P(best)": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                **{f"P(rank {i + 1})": st.column_config.NumberColumn(format="%.3f") for i in range(1, k)}})
            if cw is not None:
                st.markdown('<div class="scn-h3">Typical weights under which each candidate wins</div>',
                            unsafe_allow_html=True)
                cwt = pd.DataFrame([dict(OBP=result.iloc[i]["OBP"], **dict(zip(names, cw[i])))
                                    for i in order if cw.get(i) is not None])
                ui.dataframe(cwt, hide_index=True, use_container_width=True,
                             column_config={n: st.column_config.NumberColumn(format="%.3f") for n in names})

    with tabs[-1]:
        export = result.copy()
        for c in ["Missing_interferents", "Danger_interferents", "Superbinders"]:
            export[c] = export[c].map(lambda v: "; ".join(map(str, v)) if isinstance(v, list) else v)
        safe = sc.re.sub(r"[^\w]+", "_", sc.canonical_name(res["target"]))
        st.download_button("Full ranking (CSV)", export.to_csv(index=False).encode(),
                           f"SCENTINEL_{safe}_{res['source']}.csv", "text/csv", icon=":material/download:")
        st.markdown('<div class="scn-h3" style="margin-top:14px">PDF report</div>', unsafe_allow_html=True)
        st.caption("Same report as generate_report.py: cover, detailed table, experimental vs predicted, "
                   "both SMAA methods and the comparison. Experimental data, 10–40 s.")
        if st.button("Generate PDF report"):
            try:
                with st.spinner("Running both SMAA methods and drawing the report…"):
                    st.session_state.finder_pdf = sc.build_pdf_report(
                        res["cas"], res["interf"], res["weights"], res["tau"], res["how_many"], 20_000, res["c"])
            except Exception as e:
                st.error(f"Could not build the report: {e}")
        if st.session_state.get("finder_pdf"):
            data, fname = st.session_state.finder_pdf
            st.download_button("Download the PDF report", data, fname, "application/pdf", type="primary",
                               icon=":material/picture_as_pdf:")
