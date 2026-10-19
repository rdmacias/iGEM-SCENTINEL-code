import os
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import scentinel_core as sc
import ui

ui.page_header("dionysus", "Dionysus", "Check how well a panel of OBPs separates healthy from diseased vines, "
                                       "starting from their measured VOC profiles.")


@st.cache_resource(show_spinner="Loading the binding database…")
def load():
    return sc.panel_data()


D = load()
cfg = ui.settings()["dionysus"]
EXAMPLES = {
    "MeSA + vineyard interferents (Bianca, 48 h)": "mesa_interferentes_real_48hpi.csv",
    "Bianca grapevine, 48 h post-infection (13 VOCs)": "bianca_2017_48hpi.csv",
    "Bianca grapevine, 12 h": "bianca_2017_12hpi.csv",
    "Bianca grapevine, 96 h": "bianca_2017_96hpi.csv",
    "Bianca grapevine, 0 h": "bianca_2017_0hpi.csv",
    "Synthetic per-sample profiles (invented data)": "example_vineyard_profiles_SYNTHETIC.csv",
}


def save_upload(up):
    suffix = os.path.splitext(up.name)[1] or ".csv"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(up.getvalue())
    f.close()
    return f.name


def err(J):
    e = sc.ps.error_from_j(J)
    return f"{100 * e:.1f} %" if e >= 0.001 else "<0.1 %"


left, right = st.container(), st.container()  # parameters on top, results below

with left.container(border=True):
    st.markdown('<div class="scn-h3">Parameters</div>', unsafe_allow_html=True)
    choice = st.radio("VOC profiles", ["example", "upload"], horizontal=True,
                      format_func={"example": "Example data", "upload": "My file"}.get)
    path = None
    label = None
    if choice == "example":
        label = st.selectbox("Example", list(EXAMPLES))
        path = os.path.join(sc.EXAMPLES, EXAMPLES[label])
    else:
        up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], key="d_up")
        if up is not None:
            path, label = save_upload(up), up.name
        with open(os.path.join(sc.EXAMPLES, "vinesplit_input.xlsx"), "rb") as f:
            st.download_button("Excel template", f.read(), "vinesplit_input.xlsx", icon=":material/download:",
                               use_container_width=True)
        with st.popover("Accepted formats", use_container_width=True):
            st.markdown("""
**A. Summary table** (recommended): one row per compound
`CAS | name | mean healthy | SE healthy | mean diseased | SE diseased`
CAS is mandatory, `n.d.` counts as 0.

**B. Per-sample profiles**: one row per plant, a `class` column, then one column per compound
(compound name or CAS).""")
    source = st.radio("Binding data", ["experimental", "predicted"],
                      format_func={"experimental": "Experimental only", "predicted": "Experimental + Tanimoto"}.get)
    summary_mode = path is not None and sc.tool2_is_summary(path)
    with st.expander("Advanced"):
        max_error = st.slider("Max share of plants misclassified", 0.01, 0.30, float(cfg["max_error"]), 0.01,
                              format="%.2f", help="Sets the limit J = 2·z(1−error)². 5 % → J ≥ 5.4.")
        s_noise = st.number_input("Sensor noise (fraction of the reading)", 0.0, 1.0, float(cfg["sensor_noise"]), 0.01)
        n_plants = st.number_input("Plants per group behind each SE (n)", 1, 100, int(cfg["n"]),
                                   disabled=not summary_mode)
        max_s = st.slider("Max panel size", 1, 20, int(cfg["max_sensors"]))
        scale = st.radio("Sensitivity scale", ["anchored", "inverse"],
                         format_func={"anchored": "Anchored (1 µM→1, ≥40 µM→0)", "inverse": "1/Ki"}.get)
        censored = st.radio("'>X µM' cells", ["zero", "keep"],
                            format_func={"zero": "No binding (0)", "keep": "1/(1.1·X)"}.get)
        min_conf = st.slider("Tanimoto: drop predictions with confidence below", 0.0, 1.0,
                             float(cfg["min_confidence"]), 0.05, disabled=source != "predicted")
    classes = None
    if path is not None and not summary_mode:
        try:
            prev = pd.read_csv(path)
            if "class" in prev.columns:
                found = prev["class"].astype(str).unique().tolist()
                if len(found) > 2:
                    classes = st.multiselect("Two classes to separate", found, default=found[:2], max_selections=2)
        except Exception:
            pass
    go = st.button("Evaluate panel", type="primary", use_container_width=True, disabled=path is None)

if go:
    opts = dict(max_error=max_error, sensor_noise=s_noise, n=int(n_plants), max_sensors=max_s, scale=scale,
                censored=censored, min_confidence=min_conf)
    try:
        with st.spinner("Searching for the panel that best separates the two groups…"):
            if summary_mode:
                out = sc.run_dionysus_summary(path, source, **opts)
                st.session_state.d_summary_path = path
                st.session_state.d_opts = opts
            else:
                out = sc.run_dionysus_profiles(path, source, classes=classes, **opts)
        out["label"] = label
        out["max_error"] = max_error
        st.session_state.d_res = out
        ui.log_run("Dionysus", label, f"{len(out['panel'])} OBPs, J {out['J0']:.2f}, "
                                      f"{err(out['J0'])} misclassified", out["grade"],
                   state={"d_res": out, **({"d_summary_path": path, "d_opts": opts} if summary_mode else {})})
    except Exception as e:
        st.session_state.d_res = None
        st.session_state.d_err = f"Could not read or analyse this file: {e}"

r = st.session_state.get("d_res")
with right.container(border=True):
    if go:
        ui.scroll_here()
    if st.session_state.get("d_err"):
        ui.card_title("Results")
        ui.info_box("bad", st.session_state.pop("d_err"))
        st.stop()
    if not r:
        ui.card_title("Results")
        ui.empty("Choose the VOC profiles and press Evaluate panel.")
        st.stop()
    ui.card_title("Results", [r['label'],
                             f"{'experimental' if r['source'] == 'experimental' else 'experimental + Tanimoto'}"])
    thr, thr_txt = r["thr"]
    ui.verdict(r["grade"], "panel", r["why"][:1].upper() + r["why"][1:] + ".", obps=r["panel"])
    ui.kv([("Separation J", f"{r['J0']:.2f}"), ("Plants misclassified", err(r["J0"])),
           ("Limit", f"J ≥ {thr:.2f} ({thr_txt})"),
           ("All predictions at once", f"J {r['J_all']:.2f} ({err(r['J_all'])})"),
           ("Compounds", len(r["comps"]))])
    if r["mode"] == "profiles":
        st.caption(f"Classes: {r['classes'][0]} (n={r['sizes'][0]}) vs {r['classes'][1]} (n={r['sizes'][1]})")

    b = r["bets"]
    bad = b[b["verdict"] != "safe"] if len(b) else b
    near = bad[(bad["verdict"] != "risk") | (bad["off_by"] <= 10)] if len(bad) else bad
    names = ["Panel", "Ki known per OBP", f"What it bets on ({len(near)})", "Compounds"]
    if r["mode"] == "summary":
        names.insert(1, "Best single OBPs")
    names.append("Downloads")
    tabs = dict(zip(names, st.tabs(names, key="dionysus_tabs")))

    with tabs["Panel"]:
        s = r["steps"]
        show = pd.DataFrame({"#": s["step"], "OBP": s["OBP"], "J (panel)": s["J_panel"],
                             "Plants misclassified": [err(j) for j in s["J_panel"]],
                             "Gain": ["first" if pd.isna(g) else f"+{g:.2f}" for g in s["gain"]], "Ki measured": [f"{m}/{len(r['comps'])}" for m in s["n_measured"]]})
        ui.dataframe(show, hide_index=True, use_container_width=True, column_config={
            "J (panel)": st.column_config.NumberColumn(format="%.2f")})
        cd = s[["OBP", "J_panel"]].copy()
        cd["order"] = range(len(cd))
        bars = alt.Chart(cd).mark_bar(size=34, color="#1e7a5c", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("OBP:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("J_panel:Q", title="Separation J of the panel"))
        rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(color="#b42318", strokeDash=[5, 4]).encode(y="y:Q")
        st.altair_chart((bars + rule).properties(height=230), use_container_width=True)
        st.markdown(f'<div class="scn-note">J = (difference of the mean readings)² ÷ (spread healthy + spread diseased '
                    f'+ sensor noise). Plants misclassified = Φ(−√(J/2)). Red line: the limit for '
                    f'{r["max_error"]:.0%} misclassified.</div>', unsafe_allow_html=True)

    if "Best single OBPs" in tabs:
        with tabs["Best single OBPs"]:
            sg = r["singles"].head(10)
            ui.dataframe(pd.DataFrame({
                "OBP": sg["OBP"], "J alone": sg["J"], "Misclassified": [err(j) for j in sg["J"]],
                "Ki measured": [f"{m}/{len(r['comps'])}" for m in sg["n_measured"]],
                "Reading healthy": [f"{a:.1f} ± {b_:.1f}" for a, b_ in zip(sg["reading_A"], sg["sd_A"])],
                "Reading diseased": [f"{a:.1f} ± {b_:.1f}" for a, b_ in zip(sg["reading_B"], sg["sd_B"])],
                "Binds (Ki µM)": sg["binds"]}), hide_index=True, use_container_width=True,
                column_config={"J alone": st.column_config.NumberColumn(format="%.2f")})
            st.caption("An OBP with few measured Ki can look better than it is: a hidden compound that varies "
                       "between plants would add noise. See the bets tab.")

    with tabs["Ki known per OBP"]:
        kn = r["known"].melt(id_vars="OBP", var_name="Ki", value_name="compounds")
        kn["Ki"] = kn["Ki"].map({"measured": "measured", "predicted_used": "predicted (used)", "missing": "missing"})
        ch = alt.Chart(kn).mark_bar().encode(
            y=alt.Y("OBP:N", sort=list(r["known"]["OBP"]), title=None),
            x=alt.X("compounds:Q", title=f"compounds (of {len(r['comps'])})", stack=True),
            color=alt.Color("Ki:N", scale=alt.Scale(domain=["measured", "predicted (used)", "missing"],
                                                    range=["#1e7a5c", "#8fc9b2", "#f2c4bf"])))
        st.altair_chart(ch.properties(height=max(120, 34 * len(r["known"]))), use_container_width=True)

    with tabs[names[2 if r["mode"] == "profiles" else 3]]:
        if near.empty:
            ui.info_box("ok", "No unmeasured Ki can pull J under the limit on its own, nor any compound on all panel "
                              "OBPs at once.")
        else:
            st.caption("For the panel to keep its separation, each of these must be true. Most fragile first.")
            ui.table(pd.DataFrame({
                "This must be true": near["bet"].values,
                "J now → worst": [f"{a:.2f} → {b_:.2f}" for a, b_ in zip(near["J_now"], near["J_min"])],
                "Misclassified": [f"{err(a)} → {err(b_)}" for a, b_ in zip(near["J_now"], near["J_min"])],
                "Compound": [f"{ro} ({sh:+.1f} SD)" for ro, sh in zip(near["compound_role"], near["shift_SD"])],
                "Risk": ["AGAINST prediction" if v == "BREAKS (prediction)" else
                         (v if v != "risk" else f"lost if off by ×{o:.3g}") for v, o in zip(near["verdict"], near["off_by"])]}),
                ["44%", "13%", "15%", "14%", "14%"], danger_col="Risk")
        worst = b.sort_values("J_min").head(8) if len(b) else b
        if len(worst):
            st.markdown('<div class="scn-h3" style="margin-top:10px">Largest possible drops of J</div>',
                        unsafe_allow_html=True)
            st.caption("Every unmeasured Ki of the panel moved over 0.1 µM – 10 mM; 'all n OBPs' = the compound "
                       "binding every panel OBP where it is unmeasured, at once.")
            ui.table(pd.DataFrame({
                "OBP": [("all " + o.split(":")[0].split()[1] + " panel OBPs") if st_ == "joint" else o
                        for o, st_ in zip(worst["OBP"], worst["status"])],
                "Compound": worst["compound"].values,
                "Status": ["–" if v == "joint" else v for v in worst["status"]],
                "Ki predicted (µM)": ["–" if pd.isna(k) else f"{k:.3g}" for k in worst["Ki_predicted_uM"]],
                "Worst J": [f"{j:.2f}" for j in worst["J_min"]],
                "Misclassified": [err(j) for j in worst["J_min"]],
                "Verdict": worst["verdict"].values}),
                ["19%", "24%", "11%", "13%", "10%", "12%", "11%"], danger_col="Verdict")

    with tabs["Compounds"]:
        sh = r["shift"]
        ui.dataframe(pd.DataFrame({"Compound": r["comps_short"],
                                   "Shift diseased − healthy (SD)": [sh[c] for c in r["comps"]]}),
                     hide_index=True, use_container_width=True,
                     column_config={"Shift diseased − healthy (SD)": st.column_config.NumberColumn(format="%+.2f")})
        if r["mode"] == "summary":
            rep = r["report"]
            if rep["skipped"]:
                st.warning("Skipped rows: " + "; ".join(f"{a} [{b_}]" for a, b_ in rep["skipped"]))
            if rep["nd"]:
                st.caption("'n.d.' read as 0: " + "; ".join(f"{a} {b_}" for a, b_ in rep["nd"][:20]))

    with tabs["Downloads"]:
        c = st.columns(2)
        c[0].download_button("Panel (CSV)", r["steps"].to_csv(index=False).encode(), "dionysus_panel.csv",
                             "text/csv", icon=":material/download:")
        c[1].download_button("Bets (CSV)", b.to_csv(index=False).encode(), "dionysus_bets.csv", "text/csv",
                             icon=":material/download:")
        st.caption("The PDF report (Proteus + Dionysus) is generated from Proteus, which adds this table to it."
                   if r["mode"] == "summary" else "")
