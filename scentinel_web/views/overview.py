import html

import streamlit as st

import ui

ui.page_header(None, "Overview",
               "Three independent applications for odorant-binding proteins (OBPs) and volatile organic "
               "compounds (VOCs), all running on the same binding database.")

with st.container(border=True):
    w = [1.6, 3.0, 2.0, 2.2, 1.0]
    h = st.columns(w, vertical_alignment="center")
    for c, t in zip(h, ["Application", "Purpose", "Input", "Output", ""]):
        c.markdown(f'<span class="scn-muted" style="font-size:12px;font-weight:500">{t}</span>', unsafe_allow_html=True)
    for key, app in ui.APPS.items():
        st.markdown('<hr style="margin:4px 0;border:0;border-top:1px solid #e2e5e9">', unsafe_allow_html=True)
        c = st.columns(w, vertical_alignment="center")
        c[0].markdown(f'<span class="scn-app"><img src="data:image/png;base64,{ui.ICONS[key]}" alt="">'
                      f'{app["title"]}</span>', unsafe_allow_html=True)
        c[1].markdown(app["purpose"])
        c[2].markdown(f'<span class="scn-io">{html.escape(app["input"])}</span>', unsafe_allow_html=True)
        c[3].markdown(f'<span class="scn-io">{html.escape(app["output"])}</span>', unsafe_allow_html=True)
        c[4].page_link(app["page"], label="Open", icon=":material/arrow_forward:")

ui.section("How they fit together")
c = st.columns(3)
steps = [("1. Diana", "Pick the target VOC. Diana ranks every OBP and says how much evidence backs each score."),
         ("2. Proteus", "Start from Diana's best OBP. Proteus adds the OBPs needed so that no mixture of "
                         "vineyard interferents can imitate the target, and lists the measurements the panel relies on."),
         ("3. Dionysus", "Give real healthy / diseased VOC profiles. Dionysus picks the panel that tells them "
                          "apart and estimates the share of plants it would misclassify.")]
for col, (t, d) in zip(c, steps):
    with col.container(border=True):
        st.markdown(f'<div class="scn-h3">{t}</div><div class="scn-muted">{d}</div>', unsafe_allow_html=True)

ui.section("Recent runs")
runs = st.session_state.get("runs", [])
with st.container(border=True):
    if not runs:
        ui.empty("No runs in this session yet.")
    else:
        w = [0.8, 1.0, 1.6, 3.0, 0.8, 0.9]
        h = st.columns(w, vertical_alignment="center")
        for c, t in zip(h, ["Time", "Application", "Target / input", "Result", "Verdict", ""]):
            c.markdown(f'<span class="scn-muted" style="font-size:12px;font-weight:500">{t}</span>', unsafe_allow_html=True)
        for i, r in enumerate(runs):
            st.markdown('<hr style="margin:4px 0;border:0;border-top:1px solid #e2e5e9">', unsafe_allow_html=True)
            c = st.columns(w, vertical_alignment="center")
            c[0].markdown(f"<span class='scn-num'>{r['time']}</span>", unsafe_allow_html=True)
            c[1].markdown(f"**{r['app']}**")
            c[2].markdown(html.escape(r["target"]), unsafe_allow_html=True)
            c[3].markdown(f"<span class='scn-io'>{html.escape(r['result'])}</span>", unsafe_allow_html=True)
            c[4].markdown(ui.tag(r["grade"]), unsafe_allow_html=True)
            if r.get("state") and c[5].button("Open", key=f"run_open_{i}", icon=":material/arrow_forward:",
                                              type="tertiary"):
                ui.reopen_run(r)
