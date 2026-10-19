"""
ui.py - look & feel of the SCENTINEL suite (same design as scentinel-tools.html):
white top bar and sidebar, Rubik, brand green #1e7a5c, flat cards with 1 px borders,
verdict boxes with a coloured left edge, and a status bar at the bottom.
Also keeps the per-session settings and the 'Recent runs' log.
"""
from __future__ import annotations

import base64
import datetime
import html
import os

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
VERSION = "v2.0"


def _b64(name):
    with open(os.path.join(ASSETS, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


ICONS = {k: _b64(f"{k}.png") for k in ("diana", "proteus", "dionysus")}

APPS = {
    "diana": dict(title="Diana", page="views/diana.py",
                  purpose="Rank candidate OBPs by predicted binding to one target VOC",
                  input="Target VOC, interferents (optional)",
                  output="Ranked OBPs with confidence, alerts, PDF report"),
    "proteus": dict(title="Proteus", page="views/proteus.py",
                    purpose="Test whether a mix of background VOCs can imitate the target signal, "
                            "and build the smallest panel that cannot be fooled",
                    input="Target VOC, starting OBP, interferents",
                    output="Specific OBP panel, verdict, what to measure"),
    "dionysus": dict(title="Dionysus", page="views/dionysus.py",
                     purpose="Estimate how well an OBP panel separates healthy from diseased vines",
                     input="VOC profiles of healthy and diseased plants",
                     output="OBP panel, separation J, % plants misclassified"),
}

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;600&display=swap");
:root{--bg:#f6f7f8;--card:#fff;--ink:#1c2228;--muted:#69737d;--line:#e2e5e9;--line2:#d3d8de;
--brand:#1e7a5c;--brand-dk:#145c45;--hover:#f2f4f6;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}

/* ---------- top bar ---------- */
header[data-testid="stHeader"]{background:rgba(255,255,255,.72);-webkit-backdrop-filter:saturate(180%) blur(18px);
  backdrop-filter:saturate(180%) blur(18px);border-bottom:1px solid rgba(0,0,0,.06);height:56px}
/* collapsed sidebar: small mark + round glass button instead of the long wordmark */
[data-testid="stHeaderLogo"]{height:28px !important;width:auto !important;margin-left:4px}
/* open/close buttons: flat rounded squares, same look as the nav items */
[data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapseButton"] button{width:28px;height:28px;
  border-radius:50% !important;background:rgba(255,255,255,.65) !important;border:.5px solid rgba(0,0,0,.08) !important;
  box-shadow:0 1px 2px rgba(0,0,0,.06),0 2px 8px rgba(0,0,0,.06) !important;
  -webkit-backdrop-filter:saturate(180%) blur(12px);backdrop-filter:saturate(180%) blur(12px);
  color:var(--muted) !important;transition:background .15s,color .15s,box-shadow .15s,transform .15s}
[data-testid="stExpandSidebarButton"]:hover,[data-testid="stSidebarCollapseButton"] button:hover{
  background:#fff !important;color:var(--ink) !important;box-shadow:0 2px 10px rgba(0,0,0,.12) !important}
[data-testid="stExpandSidebarButton"]:active,[data-testid="stSidebarCollapseButton"] button:active{transform:scale(.94)}
[data-testid="stSidebarCollapseButton"]{position:static !important;margin-left:auto;display:flex !important;
  visibility:visible !important;opacity:.7}
[data-testid="stSidebarCollapseButton"]:hover{opacity:1}
.scn-top{position:fixed;top:0;right:0;height:56px;display:flex;align-items:center;gap:12px;
  padding:0 64px 0 20px;z-index:999991;font-size:12px;color:var(--muted);pointer-events:none}
.scn-top .suite{font-size:13px;color:var(--muted)}
.scn-top .pill{border:1px solid var(--line2);border-radius:4px;padding:2px 8px;font-weight:500;color:var(--ink)}
.scn-top .live{border-color:#bfe0d3;background:#eaf3ef;color:var(--brand-dk)}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{border-right:1px solid var(--line);width:236px !important;min-width:236px !important}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{padding:14px 10px 6px 16px;display:flex;
  align-items:center;gap:6px}
[data-testid="stSidebarHeader"] img{height:26px !important;width:auto !important;max-width:160px}
[data-testid="stSidebarNav"] header, [data-testid="stNavSectionHeader"]{font-size:11px !important;font-weight:500;
  color:var(--muted) !important;text-transform:uppercase;letter-spacing:.06em}
[data-testid="stSidebarNavLink"]{border-radius:6px;padding:6px 10px;margin:1px 0}
[data-testid="stSidebarNavLink"] span{font-size:14px;color:var(--ink)}
[data-testid="stSidebarNavLink"]:hover{background:var(--hover)}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:#eaf3ef}
[data-testid="stSidebarNavLink"][aria-current="page"] span{color:var(--brand-dk);font-weight:500}
[data-testid="stSidebarNavLink"][href$="/diana"]::before,
[data-testid="stSidebarNavLink"][href$="/proteus"]::before,
[data-testid="stSidebarNavLink"][href$="/dionysus"]::before{content:"";display:inline-block;width:28px;height:28px;
  background-size:contain;background-repeat:no-repeat;margin-right:4px;flex:none}
[data-testid="stSidebarNavLink"][href$="/diana"]::before{background-image:url(data:image/png;base64,ICON_diana)}
[data-testid="stSidebarNavLink"][href$="/proteus"]::before{background-image:url(data:image/png;base64,ICON_proteus)}
[data-testid="stSidebarNavLink"][href$="/dionysus"]::before{background-image:url(data:image/png;base64,ICON_dionysus)}
/* hide the whole material-icon wrapper (not just the icon) so the PNG icon is the only item and centres */
[data-testid="stSidebarNavLink"][href$="/diana"] > :first-child:has([data-testid="stIconMaterial"]),
[data-testid="stSidebarNavLink"][href$="/proteus"] > :first-child:has([data-testid="stIconMaterial"]),
[data-testid="stSidebarNavLink"][href$="/dionysus"] > :first-child:has([data-testid="stIconMaterial"]){display:none !important}
/* sidebar as a rounded floating glass panel */
section[data-testid="stSidebar"]{background:transparent !important;border-right:0}
[data-testid="stSidebarContent"]{margin:8px;height:calc(100% - 16px);border-radius:16px;
  background:rgba(255,255,255,.78);-webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px);
  border:1px solid rgba(0,0,0,.06);box-shadow:0 4px 24px rgba(0,0,0,.06)}
[data-testid="stSidebarNavLink"]{border-radius:10px}
/* material icons (Overview, Settings, Method) in the same 28px box as the app icons */
[data-testid="stSidebarNavLink"] > span:first-child:has([data-testid="stIconMaterial"]){width:28px;height:28px;
  margin:0 4px 0 0 !important;display:flex;align-items:center;justify-content:center;flex:none}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"]{font-size:24px !important;width:24px;height:24px;line-height:1}
/* closed: a narrow icon rail (like VS Code's activity bar) instead of hiding everything */
section[data-testid="stSidebar"][data-collapsed="true"]{transform:none !important;width:72px !important;
  min-width:72px !important;max-width:72px !important}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarContent"]{overflow:hidden}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stNavSectionHeader"],
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavSeparator"],
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavLink"] > :last-child{display:none !important}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarHeader"]{height:56px;padding:0;
  justify-content:center}
/* closed: show just the S; hovering the rail swaps it for the open button */
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarLogo"]{height:30px !important;
  max-width:none;transition:opacity .15s}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarHeader"]{cursor:pointer}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarHeader"]:hover [data-testid="stSidebarLogo"]{
  transform:scale(1.1)}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarLogo"]{transition:transform .15s}
/* clicking the S opens the sidebar (see SIDEBAR_JS), so the separate open button is not needed */
[data-testid="stExpandSidebarButton"]{display:none !important}
/* hidden helper iframes (scripts only) take no space */
[data-testid="stElementContainer"]:has(> iframe[height="0"]),
[data-testid="element-container"]:has(> iframe[height="0"]){display:none}
/* strip Streamlit's side paddings so the icons sit in the middle of the white rail */
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarContent"]{scrollbar-gutter:auto !important}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNav"],
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavItems"] li,
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavLinkContainer"]{padding-left:0 !important;
  padding-right:0 !important;margin-left:0 !important;margin-right:0 !important;width:100%}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavLink"]{justify-content:center;
  width:40px;height:40px;padding:0;margin:6px auto;border-radius:12px}
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavLink"]::before,
section[data-testid="stSidebar"][data-collapsed="true"] [data-testid="stSidebarNavLink"] > span:first-child{margin:0 !important}
[data-testid="stExpandSidebarButton"]{position:fixed !important;top:22px;left:22px;z-index:1000010}
body:has(section[data-testid="stSidebar"][data-collapsed="true"]) [data-testid="stHeaderLogo"]{display:none}
.scn-side-foot{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:10px;margin-top:6px;line-height:1.5}

/* ---------- main ---------- */
.block-container{padding-top:76px !important;padding-bottom:64px !important;max-width:1240px;transition:max-width .25s ease}
/* sidebar closed: let the page use the extra width */
body:has(section[data-testid="stSidebar"][data-collapsed="true"]) .block-container{max-width:1600px}
h1,h2,h3{letter-spacing:0}
.scn-ph{display:flex;align-items:center;gap:14px;margin:0 0 18px 0}
.scn-ph img{width:40px;height:40px}
.scn-ph h1{font-size:20px !important;font-weight:600;line-height:1.3;padding:0 !important;margin:0 !important}
.scn-ph p{color:var(--muted);margin:2px 0 0 0;font-size:14px}
.scn-sh{font-size:13px;font-weight:600;margin:26px 0 10px 0;color:var(--ink)}
.scn-h3{font-size:13px;font-weight:600;margin:0 0 2px 0}
.scn-muted{color:var(--muted);font-size:13px}
.scn-empty{color:var(--muted);padding:56px 18px;text-align:center;font-size:14px}
.scn-outh{font-size:13px;font-weight:600;padding:0 0 10px 0;border-bottom:1px solid var(--line);margin-bottom:12px}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line) !important;border-radius:6px}

/* verdict + key/value */
.scn-verdict{border:1px solid var(--line);border-left:3px solid;border-radius:4px;padding:10px 14px;margin:2px 0 12px 0}
.scn-verdict b{font-weight:600}
.scn-verdict .scn-grade{font-weight:600;margin-right:12px;padding-right:12px;border-right:1px solid currentColor;opacity:.9}
.v-ok .scn-grade{color:#145c45}.v-mid .scn-grade{color:#8a5a12}.v-bad .scn-grade{color:#9b1c13}
.scn-chip{display:inline-block;margin-left:8px;padding:1px 9px;border-radius:999px;background:#f2f4f6;color:var(--muted);font-size:12px;font-weight:400;vertical-align:1px}
.scn-meta{display:inline-flex;flex-wrap:wrap;gap:4px 16px}
a.scn-obp{color:inherit !important;text-decoration:underline dotted;text-underline-offset:3px}
a.scn-obp:hover{text-decoration:underline solid}
.scn-verdict .why{color:var(--ink);font-size:13px;margin-top:2px}
.v-ok{border-left-color:#1e7a5c;background:#f3f9f6}.v-ok b{color:#145c45}
.v-mid{border-left-color:#b7791f;background:#fdf9f0}.v-mid b{color:#8a5a12}
.v-bad{border-left-color:#b42318;background:#fdf3f2}.v-bad b{color:#9b1c13}
.scn-kv{display:flex;flex-wrap:wrap;gap:6px 26px;color:var(--muted);font-size:13px;margin:4px 0 12px 0}
.scn-kv b{color:var(--ink);margin-left:6px;font-weight:500}
.scn-num{font-family:var(--mono);font-size:13px}
.scn-note{font-size:12px;color:var(--muted);margin-top:8px}

/* overview table */
.scn-tools{width:100%;border-collapse:collapse;font-size:14px}
.scn-tools th{text-align:left;font-size:12px;color:var(--muted);font-weight:500;padding:9px 14px;border-bottom:1px solid var(--line);background:#fafbfc}
.scn-tools td{padding:16px 14px;border-bottom:1px solid var(--line);vertical-align:middle}
.scn-tools tr:last-child td{border-bottom:0}
.scn-app{display:flex;align-items:center;gap:12px;font-weight:600;font-size:15px;white-space:nowrap}
.scn-app img{flex-shrink:0}
[data-testid="stPageLink"] p,[data-testid="stPageLink"] span{white-space:nowrap}
.scn-app img{width:44px;height:44px}
.scn-io{color:var(--muted)}
.scn-runs td,.scn-runs th{padding:9px 14px}
.scn-tw{border:1px solid var(--line);border-radius:6px;overflow:auto;margin:4px 0 8px 0}
.scn-tbl{font-size:13px}.scn-tbl td{padding:9px 14px;vertical-align:top}
.scn-tag{display:inline-block;border-radius:4px;padding:1px 8px;font-size:12px;font-weight:500;border:1px solid}
.t-ok{color:#145c45;background:#f3f9f6;border-color:#bfe0d3}
.t-mid{color:#8a5a12;background:#fdf9f0;border-color:#efd9ad}
.t-bad{color:#9b1c13;background:#fdf3f2;border-color:#f2c4bf}
.t-none{color:var(--muted);background:#fafbfc;border-color:var(--line)}

/* widgets */
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"],.stFormSubmitButton>button{background:var(--brand);border-color:var(--brand);font-weight:500}
.stButton>button[kind="primary"]:disabled{background:#e7ebee;border-color:#e7ebee;color:#8a939c}
.stButton>button[kind="primary"]:hover,.stDownloadButton>button[kind="primary"]:hover{background:var(--brand-dk);border-color:var(--brand-dk)}
[data-testid="stTabs"] button p{font-size:13px}
[data-testid="stMetricValue"]{font-size:20px}
[data-testid="stExpander"] summary p{font-size:13px}
label p{font-weight:500}

/* status bar */
.scn-status{position:fixed;left:0;right:0;bottom:0;height:28px;background:#fff;border-top:1px solid var(--line);
  display:flex;align-items:center;gap:22px;padding:0 20px;font-size:12px;color:var(--muted);z-index:1000005}
.scn-status{padding-left:92px}
body:has(section[data-testid="stSidebar"][data-collapsed="false"]) .scn-status{padding-left:256px}
.scn-status .led{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--brand);margin-right:6px;vertical-align:1px}
.scn-status>span:nth-last-child(2){margin-left:auto}
footer{visibility:hidden}
@media(max-width:860px){.scn-top .suite,.scn-status span:nth-child(2){display:none}.block-container{padding-top:70px !important}}
</style>
"""


SIDEBAR_JS = """<script>
const d = window.parent.document;
if (!d.__scnToggle) {
  d.__scnToggle = true;
  d.addEventListener("click", e => {
    const sb = d.querySelector('section[data-testid="stSidebar"]');
    if (!sb || sb.dataset.collapsed !== "true" || !e.target.closest('[data-testid="stSidebarHeader"]')) return;
    e.preventDefault(); e.stopPropagation();
    d.querySelector('[data-testid="stExpandSidebarButton"]')?.click();
  }, true);
}
</script>"""


def inject_css():
    css = CSS
    for k, v in ICONS.items():
        css = css.replace(f"ICON_{k}", v)
    st.markdown(css, unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html(SIDEBAR_JS, height=0)


def top_bar():
    st.markdown(f"""<div class="scn-top"><span class="suite">OBP–VOC analysis suite</span>
      <span class="pill">{VERSION}</span></div>""",
                unsafe_allow_html=True)


def status_bar(data_line: str):
    custom = "custom" if st.session_state.get("settings_custom") else "defaults"
    st.markdown(f"""<div class="scn-status"><span><span class="led"></span>Ready</span>
      <span>{html.escape(data_line)}</span><span>Settings: {custom}</span>
      <span>SCENTINEL {VERSION}</span><span>iGEM URV-Tarragona 2026</span></div>""", unsafe_allow_html=True)


def page_header(app_key: str | None, title: str, subtitle: str):
    icon = f'<img src="data:image/png;base64,{ICONS[app_key]}" alt="">' if app_key in ICONS else ""
    st.markdown(f'<div class="scn-ph">{icon}<div><h1>{html.escape(title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div></div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="scn-sh">{html.escape(title)}</div>', unsafe_allow_html=True)


def card_title(title: str, hint=None):
    """hint: a string or a list of strings, each shown as a small grey chip next to the title."""
    hints = [hint] if isinstance(hint, str) else list(hint or [])
    st.markdown(f'<div class="scn-outh">{html.escape(title)}'
                + "".join(f'<span class="scn-chip">{html.escape(h)}</span>' for h in hints)
                + "</div>", unsafe_allow_html=True)


def empty(text: str):
    st.markdown(f'<div class="scn-empty">{html.escape(text)}</div>', unsafe_allow_html=True)


GRADE_CLASS = {"SOLID": "ok", "CONDITIONAL": "mid", "WEAK": "bad", "FAILS": "bad", "NOT ENOUGH": "bad"}
GRADE_TEXT = {"SOLID": "no single unverified value can plausibly break it.",
              "CONDITIONAL": "it rests on unverified values, but the predictions do not contradict them.",
              "WEAK": "a bet goes against the prediction, or all predictions together break it. Measure first.",
              "FAILS": "it does not reach the limit even with the data used.",
              "NOT ENOUGH": "it does not reach the limit even with the data used."}


def obp_link(name: str) -> str:
    """HTML link from a protein name to its iOBPdb page."""
    return (f"<a class='scn-obp' href='{html.escape(obp_url(name))}' target='_blank' rel='noopener' "
            f"title='Open on UniProt'>{html.escape(str(name))}</a>")


def verdict(grade: str, headline: str, why: str = "", obps=None):
    """obps: protein names appended to the headline as iOBPdb links."""
    c = GRADE_CLASS.get(grade, "mid")
    head = html.escape(headline) + (" " + " + ".join(obp_link(o) for o in obps) if obps else "")
    st.markdown(f'<div class="scn-verdict v-{c}"><span class="scn-grade">{html.escape(grade)}</span>{head}'
                + (f'<div class="why">{html.escape(why)}</div>' if why else "") + "</div>",
                unsafe_allow_html=True)


def info_box(kind: str, text: str):
    """kind: ok / mid / bad - a verdict-styled note without a grade."""
    st.markdown(f'<div class="scn-verdict v-{kind}">{text}</div>', unsafe_allow_html=True)


LIST_HELP = ("**Format:** one compound per line, as a name or a CAS number. "
             "Blank lines and lines starting with `#` are ignored.\n\n"
             "Example:\n```\n# vineyard interferents\nhexanal\nL-Linalool\n87-44-5\n```")
VOC_EXAMPLES = {"Copy 5-VOC vineyard example": "interferents_example.txt",
                "Copy 42-VOC vineyard matching list": "interferents_matching.txt"}


def voc_list_input(key: str, examples_dir: str):
    """Paste box (gray example as placeholder) + .txt upload for a VOC list.
    Returns the non-comment lines when the user clicks 'Use this list', else None."""
    with st.expander("Load a list (paste or upload .txt)"):
        texts = {lab: open(os.path.join(examples_dir, f), encoding="utf-8").read() for lab, f in VOC_EXAMPLES.items()}
        for lab, txt in texts.items():
            if st.button(lab, key=f"{key}_cp_{lab}", icon=":material/content_copy:", use_container_width=True):
                st.session_state[f"{key}_txt"] = txt
        text = st.text_area("Names or CAS, one per line", key=f"{key}_txt", height=150,
                            placeholder=next(iter(texts.values()), ""), help=LIST_HELP)
        up = st.file_uploader("…or upload a .txt file", type=["txt"], key=f"{key}_up", help=LIST_HELP)
        if st.button("Use this list", key=f"{key}_use", type="primary", use_container_width=True,
                     disabled=not (text.strip() or up is not None)):
            src = up.getvalue().decode("utf-8", "ignore") if up is not None else text
            return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    return None


def scroll_here():
    """Scroll the page to this point (used to jump to the results after a run)."""
    import streamlit.components.v1 as components
    st.markdown('<div id="scn-results"></div>', unsafe_allow_html=True)
    # the timestamp makes the iframe new on every run, so the script fires again
    components.html(f"<script>/*{datetime.datetime.now().timestamp()}*/ setTimeout(() => window.parent.document"
                    ".getElementById('scn-results')?.scrollIntoView({behavior: 'smooth', block: 'start'}), 300)"
                    "</script>", height=0)


def kv(pairs):
    st.markdown('<div class="scn-kv">' + "".join(f'<span>{html.escape(k)}<b>{html.escape(str(v))}</b></span>'
                                               for k, v in pairs) + "</div>", unsafe_allow_html=True)


# iOBPdb has no per-protein URL (its pages only open a protein from a table click), so links go to
# UniProt, then NCBI Protein, then the iOBPdb search page. "#name" lets tables show the name as the link text.
IOBPDB_SEARCH = "https://www.iobpdb.com/Search"


@st.cache_data(show_spinner=False)
def _obp_ids():
    import csv
    ids = {}
    with open(os.path.join(ROOT, "data", "OBP_info_new.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            uni = (r.get("UniProtID") or "").strip()
            acc = (r.get("Accession number / ID") or "").strip()
            if uni and uni not in ("-", "nan"):
                ids[r["Binding Protein Name"]] = f"https://www.uniprot.org/uniprotkb/{uni}/entry"
            elif acc and acc not in ("-", "nan"):
                ids[r["Binding Protein Name"]] = f"https://www.ncbi.nlm.nih.gov/protein/{acc}"
    return ids


def obp_url(name: str) -> str:
    from urllib.parse import quote
    return _obp_ids().get(str(name), IOBPDB_SEARCH) + "#" + quote(str(name))


def dataframe(df, column_config=None, **kw):
    """st.dataframe where an 'OBP' column links to that protein's page on iOBPdb."""
    column_config = dict(column_config or {})
    if "OBP" in df.columns:
        df = df.copy()
        df["OBP"] = df["OBP"].map(obp_url)
        column_config["OBP"] = st.column_config.LinkColumn(
            "OBP", display_text=r".*#(.*)$", help="Opens the protein on UniProt (or NCBI)")
    st.dataframe(df, column_config=column_config, **kw)


def table(df, widths=None, danger_col=None):
    """Wrapping HTML table in the suite style (st.dataframe cuts long sentences).
    danger_col: column whose cells starting with AGAINST/BREAKS are shown in red."""
    head = "".join((f"<th style='width:{widths[i]}'>" if widths else "<th>") + html.escape(str(c)) + "</th>"
                   for i, c in enumerate(df.columns))
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = "" if r[c] is None or (isinstance(r[c], float) and r[c] != r[c]) else str(r[c])
            style = ""
            if c == danger_col and (v.startswith("AGAINST") or v.startswith("BREAKS")):
                style = " style='color:#b42318;font-weight:500'"
            cell = html.escape(v)
            if c == "OBP" and v and " " not in v:  # single protein name -> link to iOBPdb
                cell = f"<a href='{html.escape(obp_url(v))}' target='_blank' rel='noopener'>{cell}</a>"
            cells.append(f"<td{style}>{cell}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f"<div class='scn-tw'><table class='scn-tools scn-tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>", unsafe_allow_html=True)


def tag(grade: str | None):
    if not grade:
        return '<span class="scn-tag t-none">–</span>'
    return f'<span class="scn-tag t-{GRADE_CLASS.get(grade, "mid")}">{html.escape(grade)}</span>'


# ---------------------------------------------------------------- settings
DEFAULT_SETTINGS = {
    "diana": {"tau": 5.0, "how_many": 10, "n_iter": 50_000, "c": 50.0},
    "proteus": {"noise": 0.05, "min_c2": 0.5, "max_sensors": 8, "cover": 1, "min_confidence": 0.3},
    "dionysus": {"max_error": 0.05, "sensor_noise": 0.05, "n": 3, "max_sensors": 8, "min_confidence": 0.3},
}


def settings():
    if "settings" not in st.session_state:
        import copy
        st.session_state.settings = copy.deepcopy(DEFAULT_SETTINGS)
    return st.session_state.settings


# ---------------------------------------------------------------- recent runs
def log_run(app: str, target: str, result: str, grade: str | None = None, state: dict | None = None):
    """state: session_state keys that bring this run's results back when reopened (None = remove the key)."""
    runs = st.session_state.setdefault("runs", [])
    runs.insert(0, dict(time=datetime.datetime.now().strftime("%H:%M:%S"), app=app, target=target,
                        result=result, grade=grade, state=state or {}))
    del runs[20:]


def reopen_run(run: dict):
    """Restore a logged run's results and go to its page."""
    for k, v in run["state"].items():
        if v is None:
            st.session_state.pop(k, None)
        else:
            st.session_state[k] = v
    st.switch_page(f"views/{run['app'].lower()}.py")
