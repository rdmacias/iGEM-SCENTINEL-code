# SCENTINEL · OBP–VOC analysis suite (web app)

Website of the SCENTINEL tools (iGEM URV-Tarragona 2026), three applications on one binding database
(621 VOCs × 437 OBPs, 7,852 measured Ki + Tanimoto predictions):

| App | What it does | Code it runs |
|---|---|---|
| **Diana** | Ranks every OBP for a target VOC (affinity, selectivity, stability, promiscuity) with confidence, interferent alerts, superbinders, measurement worklist, SMAA robustness and the 7-page PDF report | `obp_searcher_aborrit/main.py`, `confidence.py`, `SMAAA.py`, `generate_report.py` |
| **Proteus** | Tests whether a mixture of background VOCs can imitate the target and builds the smallest OBP panel that cannot be fooled; verdict (SOLID / CONDITIONAL / WEAK), what the panel bets on, pairs to measure, PDF report | `panel_selector_project` Tool 1 + `generate_panel_report.py` |
| **Dionysus** | Picks the OBP panel that best separates healthy from diseased vines (Fisher J, % plants misclassified), with the same uncertainty check on missing Ki | `panel_selector_project` Tool 2 |

Plus **Overview** (apps + recent runs of the session), **Personalize settings** (default values of each
app, export/import as JSON) and **Method & data**.

The scientific code in `tools/` is copied **unchanged** from the team repositories (the panel selector
is the final version: verdicts, bets, coverage, measured-first search, PDF report).
`scentinel_core.py` only connects it to the pages (form fields instead of `input()` prompts).

```
app.py                  entry point: navigation, logo, top bar, status bar
ui.py                   look & feel (same design as scentinel-tools.html), settings, recent runs
views/                  overview, diana, proteus, dionysus, settings, method
assets/                 logo and app icons
scentinel_core.py       glue between pages and tools
tools/obp_searcher/     main.py, confidence.py, SMAAA.py, generate_report.py
tools/panel_selector/   panel_selector package, the two CLI tools, generate_panel_report.py
data/                   binding CSVs, OBP info, Tanimoto diagnostic, example files
.streamlit/config.toml  theme (Rubik, green #1e7a5c)
```

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```
It opens at http://localhost:8501

## Put it online with Render

1. **Create a GitHub repository** (e.g. `scentinel-web`) and upload the *contents* of this folder, so
   that `app.py` is at the top level of the repository:
   ```bash
   git init && git add . && git commit -m "SCENTINEL web app"
   git branch -M main
   git remote add origin https://github.com/<your-team>/scentinel-web.git
   git push -u origin main
   ```
2. Go to **https://dashboard.render.com** and sign in with GitHub.
3. **New → Blueprint**, pick the repository. Render reads `render.yaml` and fills in every setting.
   (Or **New → Web Service** by hand: build `pip install -r requirements.txt`, start
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, environment variable
   `PYTHON_VERSION = 3.12.7`.)
4. **Deploy**. The first build takes 3–5 min and gives a link like `https://scentinel.onrender.com`.
   Every `git push` to `main` redeploys automatically.

### Good to know
- **Free plan sleeps** after 15 min without visitors; the next visit takes ~1 min to wake up.
- **Memory**: the app uses ~350 MB on its own and ~420 MB with a few visitors running analyses
  (free plan limit: 512 MB). For the iGEM judging period use the **Starter** plan (always on, more RAM):
  change `plan: free` to `plan: starter` in `render.yaml`, or in Render → *Settings → Instance type*.
- `scikit-learn` is not installed on purpose (saves ~40 MB): the web uses the Fisher J metric; the
  optional cross-validated accuracy of the command-line Tool 2 is the only part that needs it.
- **Updating the data**: replace the files in `data/` (same names) and push. If you regenerate the imputed
  CSVs, also copy the new `results/tanimoto_diag_table.csv` to `data/results/` (it must come from the
  same run).
- Custom domain: Render → your service → *Settings → Custom Domains*.

## Notes for the code team
- The web uses `results/tanimoto_diag_table.csv` (Oct 2026 run) for every tool. It matches the imputed
  CSVs cell for cell, unlike the older `tanimoto_imputation_diagnostic.csv` (19k imputed cells without a
  row). Prediction confidences in Proteus/Dionysus can therefore differ slightly from a terminal run that
  reads the older file; panels and verdicts for methyl salicylate are the same.
- `SMAAA.smaa_full` is called with `chunk_size=1000` instead of the default 20 000: identical results
  (same random stream), ~60× less memory. Worth changing the default in `SMAAA.py` too.
- `main.validate_voc` asks with `input()` when a name has several partial matches. The web replaces it
  with a version that also accepts exact `/` synonyms and never asks (compounds are picked from lists).
- Settings and recent runs live in the visitor's session (they reset when the page is reloaded).
