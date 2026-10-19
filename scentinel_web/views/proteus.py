import altair as alt
import pandas as pd
import streamlit as st

import scentinel_core as sc
import ui

ui.page_header("proteus", "Proteus", "Test whether a mix of background VOCs can mimic the signal of your target, "
                                     "and build the smallest OBP panel that cannot be fooled.")


@st.cache_resource(show_spinner="Loading the binding database…")
def load():
    return sc.panel_data()


D = load()
meta = D.meta.set_index("label")
compound_labels = D.meta["label"].tolist()
fmt = lambda lab: f"{meta.loc[lab, 'Name']} (CAS {meta.loc[lab, 'CAS']})"
cfg = ui.settings()["proteus"]


def resolve(queries):
    labs, errs = D.resolve_many(queries, strict=False)
    return labs, errs


# ---------------------------------------------------------------- prefill from Diana
pre = st.session_state.pop("proteus_prefill", None)
if pre:
    try:
        st.session_state.p_target = D.resolve(pre["target_cas"])
    except Exception:
        pass
    labs, _ = resolve(pre.get("interf_cas", []))
    if labs:
        st.session_state.p_interf = labs
    st.session_state.p_start = pre["start"][0] if pre.get("start") else None

if "p_target" not in st.session_state:
    st.session_state.p_target = next((l for l in compound_labels if l.lower().startswith("methyl salicylate")), None)
if "p_interf" not in st.session_state:
    st.session_state.p_interf = []
if "p_start" not in st.session_state:
    st.session_state.p_start = "DcitOBP3" if "DcitOBP3" in D.obps else None

left, right = st.container(), st.container()  # parameters on top, results below

with left.container(border=True):
    st.markdown('<div class="scn-h3">Parameters</div>', unsafe_allow_html=True)
    if pre:
        st.caption("Filled in from Diana.")
    target = st.selectbox("Target VOC", compound_labels, format_func=fmt, key="p_target",
                          placeholder="Name or CAS")
    start = st.selectbox("OBP to test (starting OBP)", D.obps, key="p_start",
                         placeholder="Leave empty: the search picks it",
                         help="Usually Diana's best OBP. Proteus first tests whether it can be fooled on its own, "
                              "then adds the OBPs that complement it.")
    if "p_pending" in st.session_state:  # list loaded below; must be set before the widget exists
        st.session_state.p_interf = st.session_state.pop("p_pending")
    interf = st.multiselect("Interfering VOCs", compound_labels, format_func=fmt, key="p_interf",
                            placeholder="Search and add compounds…")
    queries = ui.voc_list_input("p", sc.EXAMPLES)
    if queries is not None:
        labs, errs = resolve(queries)
        st.session_state.p_pending = labs
        st.session_state.p_errs = errs
        st.rerun()
    if st.session_state.get("p_errs"):
        st.warning("Not found (skipped): " + ", ".join(st.session_state.pop("p_errs")))
    source = st.radio("Binding data", ["experimental", "predicted"], key="p_source",
                      format_func={"experimental": "Experimental only",
                                   "predicted": "Experimental + Tanimoto"}.get,
                      help="Experimental: picks among the best-measured OBPs. "
                           "With Tanimoto: predicted Ki also count, so it can find panels the experimental mode misses.")
    with st.expander("Advanced"):
        noise = st.number_input("Sensor noise floor", 0.0001, 10.0, float(cfg["noise"]), 0.01, format="%.4f",
                                help="Residual the panel must exceed. 0.05 ≈ 5 points of occupancy at target 1 µM.")
        min_c2 = st.slider("Min c2 of a candidate OBP", 0.0, 1.0, float(cfg["min_c2"]), 0.05,
                           help="c2 = how much is known about the interferents on that OBP "
                                "(measured 1, predicted × confidence, missing 0).")
        max_sensors = st.slider("Max panel size", 1, 20, int(cfg["max_sensors"]))
        priority = st.radio("How the next OBP is chosen", ["auto", "measured", "score"], horizontal=True,
                            format_func={"auto": "Auto", "measured": "Measured first", "score": "Highest residual"}.get,
                            help="Auto = measured first with experimental data, highest residual with Tanimoto.")
        cover = st.slider("Every interferent must bind ≥ n panel OBPs", 0, 3, int(cfg["cover"]))
        min_conf = st.slider("Tanimoto: drop predictions with confidence below", 0.0, 1.0,
                             float(cfg["min_confidence"]), 0.05, disabled=source != "predicted")
    go = st.button("Run interference test", type="primary", use_container_width=True,
                   disabled=not (target and interf))
    if not interf:
        st.caption("Add at least one interfering VOC.")

if go:
    try:
        with st.spinner("Searching for the smallest panel that cannot be fooled…"):
            run = sc.run_proteus(target, interf, source, [start] if start else [], noise=noise, min_c2=min_c2,
                                 max_sensors=max_sensors, priority=priority, cover=cover, min_confidence=min_conf)
        run["inputs"] = dict(target=target, interf=list(interf), start=[start] if start else [],
                             proteus=dict(noise=noise, min_c2=min_c2, max_sensors=max_sensors, cover=cover,
                                          min_confidence=min_conf, priority=priority))
        st.session_state.p_res = run
        st.session_state.pop("p_pdf", None)
        ui.log_run("Proteus", run["target_short"],
                   f"{len(run['panel'])} OBPs, residual {run['final_resid']:.4f}, {len(interf)} interferents",
                   run["grade"], state={"p_res": run, "p_pdf": None})
    except sc.tool1.NotEnoughCandidates as e:
        st.session_state.p_res = None
        st.session_state.p_err = str(e)

r = st.session_state.get("p_res")
with right.container(border=True):
    if go:
        ui.scroll_here()
    if st.session_state.get("p_err"):
        ui.card_title("Results")
        ui.info_box("bad", st.session_state.pop("p_err"))
        st.stop()
    if not r:
        ui.card_title("Results")
        ui.empty("Choose the target, the OBP to test and the interfering VOCs, then run the test.")
        st.stop()
    ui.card_title("Results", [r['target_short'],
                             f"{'experimental' if r['source'] == 'experimental' else 'experimental + Tanimoto'}"])
    ui.verdict(r["grade"], "panel", r["why"][:1].upper() + r["why"][1:] + ".", obps=r["panel"])
    pairs = [("Residual", f"{r['final_resid']:.4f}"), ("Noise floor", r["noise"]),
             ("OBPs", len(r["panel"])), ("Candidates", f"{r['n_candidates']} of {r['n_with_target']}")]
    if pd.notna(r["r_all"]):
        pairs.append(("All predictions at once", f"{r['r_all']:.4f} → " +
                      ("passes" if r["r_all"] >= r["noise"] else "fails")))
    ui.kv(pairs)
    for w in r["warnings"]:
        st.caption(w)
    if r["uncovered"]:
        st.caption("No OBP binds: " + ", ".join(r["uncovered"]) + " (invisible to the panel).")

    bets = r["bets"]
    near = bets[bets["against"] | (bets["off_by"] <= sc.tool1.BET_MAX_FOLD)] if bets is not None and len(bets) else None
    tabs = st.tabs(["Panel", f"What it bets on ({0 if near is None else len(near)})",
                    f"To measure ({len(r['worklist'])})", "Report & downloads"], key="proteus_tabs")

    with tabs[0]:
        st_ = r["steps"]
        show = pd.DataFrame({"#": st_["step"], "OBP": st_["OBP"], "Added by": st_["added_by"],
                             "Residual": st_["residual"], "Ki target (µM)": st_["Ki_target_uM"],
                             "c2": st_["c2"], "Measured / imputed": [f"{a} / {b}" for a, b in
                                                                       zip(st_["cells_measured"], st_["cells_imputed"])],
                             "Binds (coverage)": st_["covers"]})
        ui.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "Residual": st.column_config.NumberColumn(format="%.4f"),
            "Ki target (µM)": st.column_config.NumberColumn(format="%.2f"),
            "c2": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        chart_df = st_[["OBP", "residual", "added_by"]].copy()
        chart_df["order"] = range(len(chart_df))
        bars = alt.Chart(chart_df).mark_bar(size=34, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("OBP:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("residual:Q", title="Residual (best fake mixture)"),
            color=alt.Color("added_by:N", title="Added by",
                            scale=alt.Scale(domain=["start", "search", "set", "cover"],
                                            range=["#8a96a3", "#1e7a5c", "#5fae8f", "#b7791f"])))
        rule = alt.Chart(pd.DataFrame({"y": [r["noise"]]})).mark_rule(color="#b42318", strokeDash=[5, 4]).encode(y="y:Q")
        st.altair_chart((bars + rule).properties(height=230), use_container_width=True)
        if r["recipe"]:
            st.markdown('<div class="scn-note"><b>Closest fake mixture:</b> ' +
                        ", ".join(f"{c:.3f} × {n}" for c, n in r["recipe"]) + "</div>", unsafe_allow_html=True)
        st.markdown('<div class="scn-note">Residual = distance between the target and the best possible '
                    'non-negative mixture of interferents (NNLS). The panel works when it is above the noise floor '
                    '(red line).</div>', unsafe_allow_html=True)

    with tabs[1]:
        if near is None or near.empty:
            ui.info_box("ok", "No single unverified or fragile value can plausibly break this panel."
                        if r["passed"] else "The panel does not clear the noise floor, so there is nothing to bet on.")
        else:
            st.caption("For the panel to keep working, each of these must be true. Most fragile first.")
            tb = pd.DataFrame({"#": range(1, len(near) + 1), "This must be true": near["bet"].values,
                               "Data": near["evidence"].values,
                               "Risk": ["AGAINST prediction" if a else f"lost if off by ×{o:.3g}"
                                        for a, o in zip(near["against"], near["off_by"])]})
            ui.table(tb, ["4%", "42%", "38%", "16%"], danger_col="Risk")
            far = len(bets) - len(near)
            if far:
                st.caption(f"+ {far} far-fetched bet(s): lost only if a guess is more than 10× wrong (in the CSV).")

    with tabs[2]:
        wl = r["worklist"]
        if wl.empty:
            ui.info_box("ok", "Every target/interferent pair of the panel has a lab Ki.")
        else:
            cols = ["OBP", "compound", "status"] + [c for c in ["verdict", "breaks_if", "Ki_predicted_uM"] if c in wl] \
                   + ["confidence"]
            show = wl[cols].rename(columns={"compound": "Compound", "status": "Status", "verdict": "Risk",
                                            "breaks_if": "Breaks the panel if", "Ki_predicted_uM": "Ki predicted (µM)",
                                            "confidence": "Confidence"})
            st.caption("Panel pairs without a lab Ki, most urgent first. 'no data' = assumed not to bind; "
                       "'not used' = a prediction exists but was left out.")
            ui.dataframe(show, hide_index=True, use_container_width=True, column_config={
                "Ki predicted (µM)": st.column_config.NumberColumn(format="%.3g"),
                "Confidence": st.column_config.NumberColumn(format="%.2f")})

    with tabs[3]:
        safe = sc.re.sub(r"[^\w]+", "_", r["target_short"])[:40]
        c = st.columns(3)
        c[0].download_button("Panel (CSV)", r["steps"].to_csv(index=False).encode(),
                             f"proteus_panel_{safe}_{r['source']}.csv", "text/csv", icon=":material/download:")
        c[1].download_button("To measure (CSV)", r["worklist"].to_csv(index=False).encode(),
                             f"proteus_lab_{safe}_{r['source']}.csv", "text/csv", icon=":material/download:")
        if bets is not None:
            c[2].download_button("Bets (CSV)", bets.to_csv(index=False).encode(),
                                 f"proteus_bets_{safe}_{r['source']}.csv", "text/csv", icon=":material/download:")
        st.markdown('<div class="scn-h3" style="margin-top:14px">PDF report</div>', unsafe_allow_html=True)
        dpath = st.session_state.get("d_summary_path")
        st.caption("generate_panel_report.py: Proteus with experimental and Tanimoto data, verdicts, bets, lab "
                   "priorities and the method page" + (". Includes Dionysus with the table chosen there." if dpath
                                                        else ". Choose a summary table in Dionysus to include it.")
                   + " About 20–40 s.")
        if st.button("Generate PDF report", key="p_pdf_btn"):
            i = r["inputs"]
            try:
                with st.spinner("Running both data modes and drawing the report…"):
                    st.session_state.p_pdf = sc.build_panel_pdf(i["target"], i["interf"], i["start"], dpath,
                                                                proteus=i["proteus"],
                                                                dionysus=st.session_state.get("d_opts"))
            except Exception as e:
                st.error(f"Could not build the report: {e}")
        if st.session_state.get("p_pdf"):
            data, fname = st.session_state.p_pdf
            st.download_button("Download the PDF report", data, fname, "application/pdf", type="primary",
                               icon=":material/picture_as_pdf:")
