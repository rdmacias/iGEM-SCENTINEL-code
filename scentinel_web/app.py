"""SCENTINEL web app - entry point.  Run locally with:  streamlit run app.py"""
import importlib
import os

import streamlit as st

import ui

importlib.reload(ui)  # pick up edits to ui.py without restarting the server

st.set_page_config(page_title="SCENTINEL: OBP–VOC analysis suite",
                   page_icon=os.path.join(ui.ASSETS, "logo_s.png"), layout="wide",
                   initial_sidebar_state="expanded")
st.logo(os.path.join(ui.ASSETS, "logo.png"), size="large",
        icon_image=os.path.join(ui.ASSETS, "logo_s.png"))  # just the S when the sidebar is closed

overview = st.Page("views/overview.py", title="Overview", icon=":material/grid_view:", url_path="overview",
                   default=True)
diana = st.Page("views/diana.py", title="Diana", icon=":material/radio_button_checked:", url_path="diana")
proteus = st.Page("views/proteus.py", title="Proteus", icon=":material/radio_button_checked:", url_path="proteus")
dionysus = st.Page("views/dionysus.py", title="Dionysus", icon=":material/radio_button_checked:", url_path="dionysus")
settings = st.Page("views/settings.py", title="Personalize settings", icon=":material/tune:", url_path="settings")
method = st.Page("views/method.py", title="Method & data", icon=":material/menu_book:", url_path="method")

nav = st.navigation({"Workspace": [overview], "Applications": [diana, proteus, dionysus],
                     "Settings": [settings, method]})

ui.inject_css()
ui.top_bar()
ui.settings()
ui.status_bar("Data: 621 VOCs × 437 OBPs, 7,852 measured Ki + Tanimoto predictions")

nav.run()
