import copy
import json

import streamlit as st

import ui

ui.page_header(None, "Personalize settings",
               "Adjust the default values each application starts with. They apply to the next run and are kept "
               "for this browser session; export them to reuse later.")

s = ui.settings()
new = copy.deepcopy(s)

c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown('<div class="scn-h3"><img src="data:image/png;base64,' + ui.ICONS["diana"] +
                    '" style="width:18px;vertical-align:-3px;margin-right:6px">Diana <span class="scn-muted" style="font-weight:400;margin-left:6px">OBP ranking</span></div>',
                    unsafe_allow_html=True)
        new["diana"]["tau"] = st.number_input("τ selectivity margin", 1.0, 100.0, float(s["diana"]["tau"]), 1.0,
                                              help="s2 = min(1, Ki interferent / (Ki target × τ))")
        new["diana"]["how_many"] = st.number_input("Candidates shown", 5, 50, int(s["diana"]["how_many"]))
        new["diana"]["n_iter"] = st.select_slider("SMAA iterations", [10_000, 20_000, 50_000, 100_000, 200_000],
                                                  value=int(s["diana"]["n_iter"]))
        new["diana"]["c"] = st.number_input("Perturbation dispersion c%", 1.0, 100.0, float(s["diana"]["c"]), 1.0)
    with st.container(border=True):
        st.markdown('<div class="scn-h3"><img src="data:image/png;base64,' + ui.ICONS["dionysus"] +
                    '" style="width:18px;vertical-align:-3px;margin-right:6px">Dionysus <span class="scn-muted" style="font-weight:400;margin-left:6px">healthy vs diseased</span></div>',
                    unsafe_allow_html=True)
        new["dionysus"]["max_error"] = st.number_input("Max share of plants misclassified", 0.01, 0.30,
                                                       float(s["dionysus"]["max_error"]), 0.01, format="%.2f",
                                                       help="5 % → the panel needs J ≥ 5.4")
        new["dionysus"]["sensor_noise"] = st.number_input("Sensor noise (fraction of the reading)", 0.0, 1.0,
                                                          float(s["dionysus"]["sensor_noise"]), 0.01)
        new["dionysus"]["n"] = st.number_input("Plants per group behind each SE", 1, 100, int(s["dionysus"]["n"]))
        new["dionysus"]["max_sensors"] = st.number_input("Max panel size", 1, 20, int(s["dionysus"]["max_sensors"]),
                                                         key="d_ms")
        new["dionysus"]["min_confidence"] = st.number_input("Tanimoto: min. confidence", 0.0, 1.0,
                                                            float(s["dionysus"]["min_confidence"]), 0.05, key="d_mc")
with c2:
    with st.container(border=True):
        st.markdown('<div class="scn-h3"><img src="data:image/png;base64,' + ui.ICONS["proteus"] +
                    '" style="width:18px;vertical-align:-3px;margin-right:6px">Proteus <span class="scn-muted" style="font-weight:400;margin-left:6px">interference</span></div>',
                    unsafe_allow_html=True)
        new["proteus"]["noise"] = st.number_input("Sensor noise floor", 0.0001, 10.0, float(s["proteus"]["noise"]),
                                                  0.01, format="%.4f",
                                                  help="0.05 ≈ 5 points of occupancy at target 1 µM")
        new["proteus"]["min_c2"] = st.number_input("Min c2 of a candidate OBP", 0.0, 1.0,
                                                   float(s["proteus"]["min_c2"]), 0.05)
        new["proteus"]["max_sensors"] = st.number_input("Max panel size", 1, 20, int(s["proteus"]["max_sensors"]),
                                                        key="p_ms")
        new["proteus"]["cover"] = st.number_input("Every interferent must bind ≥ n panel OBPs", 0, 3,
                                                  int(s["proteus"]["cover"]))
        new["proteus"]["min_confidence"] = st.number_input("Tanimoto: min. confidence", 0.0, 1.0,
                                                           float(s["proteus"]["min_confidence"]), 0.05, key="p_mc")
    with st.container(border=True):
        st.markdown('<div class="scn-h3">Save, restore, share</div>', unsafe_allow_html=True)
        b = st.columns(2)
        if b[0].button("Save settings", type="primary", use_container_width=True):
            st.session_state.settings = new
            st.session_state.settings_custom = new != ui.DEFAULT_SETTINGS
            st.toast("Settings saved. They apply to the next run.")
            st.rerun()
        if b[1].button("Restore defaults", use_container_width=True):
            st.session_state.settings = copy.deepcopy(ui.DEFAULT_SETTINGS)
            st.session_state.settings_custom = False
            st.toast("Default settings restored.")
            st.rerun()
        st.download_button("Export settings (JSON)", json.dumps(s, indent=2), "scentinel_settings.json",
                           "application/json", use_container_width=True, icon=":material/download:")
        up = st.file_uploader("Import settings", type=["json"], label_visibility="collapsed")
        if up is not None and st.button("Load imported settings", use_container_width=True):
            try:
                data = json.loads(up.getvalue().decode())
                merged = copy.deepcopy(ui.DEFAULT_SETTINGS)
                for app, vals in data.items():
                    if app in merged:
                        merged[app].update({k: v for k, v in vals.items() if k in merged[app]})
                st.session_state.settings = merged
                st.session_state.settings_custom = merged != ui.DEFAULT_SETTINGS
                st.toast("Settings imported.")
                st.rerun()
            except Exception as e:
                st.error(f"Not a valid settings file: {e}")
