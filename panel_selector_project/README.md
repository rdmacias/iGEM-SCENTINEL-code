# panel_selector

A standalone Python module that implements the two panel tools described in
`IDEA_SOFTWARE.pdf`, both built on the same shared core (sensitivity matrix +
forward model + greedy optimiser). It does not modify SCENTINEL v3 and does not
depend on its code, but it reads the same Ki table (OBP x compound, in uM).

## What is new in this version (v2)

- **PDF report: `python generate_panel_report.py`** (or option 3 of `python main.py`). Runs Tool 1 with
  experimental and predicted data and, if a vineyard summary table is given with `--profiles`, Tool 2 in
  both modes; writes `SCENTINEL_panel_report_<target>_<date>.pdf` in the SCENTINEL style of
  `generate_report.py`: cover with the verdict of every run, panel + residual chart, bets and pairs to
  measure, experimental vs predicted, Tool 2 J chart with % misclassified and Ki known per OBP, lab
  priorities across all runs, and a method page.

  ```bash
  python generate_panel_report.py --target "methyl salicylate" --interferents data/interferents_example.txt \
      --start-with DcitOBP3 --profiles examples/tool2_mesa_interferentes/mesa_interferentes_real_48hpi.csv
  ```
- **Shorter console output.** Tool 1 prints the panel, the verdict with its bets and one "TO MEASURE"
  table (pairs sorted by risk); Tool 2 the 5 best single OBPs, the panel and the verdict. `--verbose`
  prints every section as before. CSV files per run: `tool1_{panel,lab,bets}_*`, `tool2_{panel,bets}_*`
  (`tool1_weak_points_*` only with `--verbose`).

How the tools deal with Ki values nobody measured (full explanation in Spanish:
`docs/MISSING_DATA_ES.md`):

- **Tool 1 - verdict and bets.** After the panel: VERDICT (SOLID / CONDITIONAL / WEAK) and
  WHAT THIS PANEL BETS ON - every unmeasured or fragile Ki the "pass" depends on, with the Ki
  range that would break it and what the Tanimoto prediction says.
- **Tool 1 - weak points.** Each unmeasured pair is moved over 0.1 uM - 10 mM; measured Ki that
  break the panel with a <= 25 % change are listed; lab worklist sorted by risk.
- **Tool 1 - `--priority`.** Mode 1 (experimental) picks the best-MEASURED useful OBP
  (gain x c2^2); mode 2 (predicted) picks the highest residual, predictions included.
- **Tool 1 - blind OBPs removed** (only `>X` values: they bind none of the compounds).
- **Tool 1 - `--cover`.** Every interferent must really bind at least one panel OBP.
- **Tool 2 - uncertainty check.** Same verdict and bets for the Fisher J, one Ki at a time and
  each compound on all panel OBPs at once; limit set by `--max-error` (share of plants
  misclassified, error = Phi(-sqrt(J/2)), default 5 % -> J >= 5.4); every J printed with its %.
- Results: `results/tool1_{panel,lab,bets}_*`, `tool2_{panel,bets}_*`.
- Example inputs: `examples/tool2_mesa_interferentes/` (files marked INVENTADO contain invented values).
- Ambiguous target names: the matches are listed and you pick one by number (`CompoundAmbiguous`).
- `main.py`: real data only; files are chosen by typing their name.
- 43 tests (`tests/test_weak_points.py` and `tests/test_report.py` are new).

## Overview

1. **Sensitivity matrix S** (OBPs x compounds): `s_ji = 1/Ki`, or the s1 scale of
   `main.py` (`--scale anchored`: Ki 1 uM -> 1, Ki >= 40 uM -> 0).
2. **Forward model** (linear regime): `response_j = sum_i s_ji * c_i`.
3. Two tools on top of it, same greedy loop, different objective:
   - **Tool 1 (`specificity.py`)**: smallest panel such that no non-negative
     mixture of interferents can imitate the target (e.g. MeSA). Solved with NNLS
     (`scipy.optimize.nnls`); the residual is the distance between the best
     possible imitation and the target. Stop when the residual exceeds the
     sensor noise.
   - **Tool 2 (`discrimination.py`)**: panel that best separates two classes
     (e.g. healthy vs diseased) from GC-MS profiles projected through the same
     forward model. Metric: Fisher ratio (multivariate in summary mode), or
     cross-validated accuracy (`scikit-learn`, needs per-sample data).

The greedy loop (`greedy_core.py`) is shared and implements the three stop
conditions of the document: `target_crossed` (success), `diminishing_returns`
(redundant sensors) and `library_exhausted` (honest verdict: your protein
library is not enough).

## Layout

```
panel_selector/
    model.py            SensitivityMatrix + forward_response (shared core)
    greedy_core.py      shared greedy loop (tie-break + plateau lookahead)
    specificity.py      Tool 1 - NNLS residual, greedy search, optional Monte Carlo
    discrimination.py   Tool 2 - Fisher ratio / CV accuracy, summary-statistics mode
    scentinel_data.py   reads the SCENTINEL CSVs, the Tanimoto diagnostic and vineyard tables
scentinel_tool1_specificity.py     command-line Tool 1 on the SCENTINEL data
scentinel_tool2_discrimination.py  command-line Tool 2 on the SCENTINEL data
generate_panel_report.py           PDF report of both tools (both data modes)
tools/build_vineyard_profiles.py   Ciubotaru et al. 2021 table -> summary files
examples/
    example_sensitivity.csv        synthetic data (reproduces the hand-worked example
                                   of the PDF: DcitOBP3 + OBP-B, residual ~0.171 for
                                   MeSA vs nonanal/hexanol/limonene)
    demo_tool1_specificity.py      reproduces the PDF numbers step by step
    demo_tool2_discrimination.py   synthetic healthy vs diseased; avoids a strong
                                   but non-discriminating "decoy" OBP
tests/
    test_core.py                   checks the key numbers of the document
    test_scentinel_data.py         data adapter, censored values, summary mode, Excel input
```

## Quick start

```bash
pip install numpy scipy pandas scikit-learn openpyxl   # scikit-learn only needed for cv_accuracy
PYTHONPATH=. python main.py
PYTHONPATH=. python -m pytest tests/ -v
```

```python
from panel_selector import ScentinelData, greedy_panel_for_specificity

D = ScentinelData.load("data")                       # SCENTINEL CSVs
target = D.resolve("methyl salicylate")
interf = D.resolve_many(["hexanal", "1-hexanol", "L-Linalool"])[0]
S = D.specificity_matrix(target, interf, source="experimental")
result = greedy_panel_for_specificity(S, target=target, interferents=interf, noise=0.05)
print(result.panel, result.stop_reason)
```

## Menu: `python main.py`

The easiest way in. It works on the real SCENTINEL files in `data/` and asks which tool
(1 = specificity, 2 = discrimination, 3 = PDF report of both), then the questions for that
tool (experimental or predicted, minimum Tanimoto confidence, interferent / profile file,
target, starting OBP), prints the results and offers to run another analysis.

- Files are chosen by typing the exact filename (relative or full path; it is also looked
  up in the project and `data/` folders); Enter takes the first file listed.
- If a target name matches several compounds (e.g. "methyl"), the matches are listed with
  their CAS and you pick one by number.
- The synthetic examples are still available directly:
  `PYTHONPATH=. python examples/demo_tool1_specificity.py` (and `demo_tool2_discrimination.py`).

Tool 1 ends with a **TO MEASURE** table (`--verbose`: the full LAB WORKLIST by OBP): every
target/interferent pair on the final panel that was not measured, so you know what to
measure before trusting the panel:
`no data` (no Ki at all, assumed non-binding - most urgent), `not used` (a Tanimoto
prediction exists but was not used) and `imputed` (the residual used a prediction).
It is also saved as `results/tool1_lab_*.csv`.

### Choosing the next OBP: `--priority` (NEW)

- Mode 1 (experimental) defaults to `measured`: among the OBPs that raise the residual by
  at least `--min-gain`, pick the one with the highest gain x c2^2 (best-measured useful OBP).
- Mode 2 (predicted) defaults to `score`: pick the OBP that raises the residual most,
  predictions included.

Blind OBPs (only `>X` values for the target and every interferent: they bind none of
them) are no longer candidates: the `>X -> 1/(1.1 X)` rule gave them a fake, equal
signal for every compound.

### Coverage: no invisible interferent (NEW, `--cover`, default 1)

After the search, every interferent must BIND at least one panel OBP (a measured Ki that
is not `>X`, or a used prediction); OBPs that bind the missing ones are added (marked
`cover` in the table). Otherwise an
interferent with no data anywhere on the panel could imitate the target for free.

### Verdict and bets (NEW)

Right after the panel, Tool 1 prints a **VERDICT** (`SOLID` / `CONDITIONAL` / `WEAK`) and
**WHAT THIS PANEL BETS ON**: every unverified value the "pass" depends on, written as a
statement with its data, e.g.

```
1. L-Linalool does NOT bind DcitOBP3 at all  AGAINST prediction
   data: not measured; prediction Ki 3.09 uM, conf 0.59, left out (counted as 'does not bind')
```

Saved in `results/tool1_bets_*.csv`; `--show-bets N` sets how many are printed. A full
explanation of how missing data is handled (in Spanish) is in `docs/MISSING_DATA_ES.md`.

### Weak points (NEW)

When the panel clears the noise floor, Tool 1 then asks which single value could
break it. It changes ONE cell of the panel at a time (Ki from 0.1 uM to 10 mM) and
checks whether the residual falls back under the noise:

| verdict | meaning |
|---|---|
| `BREAKS (prediction)` | the Tanimoto prediction for that pair alone already breaks the panel (often a prediction dropped by `--min-confidence`, or not used in experimental mode) |
| `BREAKS if it binds` | the pair has no prediction (or one that does not break), but even weak binding (Ki up to >= 30 uM) breaks it |
| `risk` | breaks only if the Ki falls in the printed range; `off by` = how many times the prediction (30 uM if none) would have to be wrong |
| `safe` | no Ki can break it on its own |

`Ki <= X` = a missing interferent that binds stronger than X breaks the panel.
`Ki >= X` = the panel relies on a predicted binding; if the true binding is weaker it breaks.
`any Ki *` = the interferent gives no signal on any other OBP of the panel, so in the
linear model (no upper limit on concentration) any binding imitates the target: it
needs to be measured on the other OBPs too.

The section also prints one line per interferent (worst pair, how many panel OBPs
have a measured Ki), the residual if every prediction were true at once, and the
values the panel depends on: Ki that break it with a change of <= 25 %
(a normal assay error is ~10-20 %; censored `>X` values are only moved upwards).
The lab worklist is re-sorted with the breaking pairs first, and everything is saved
in `results/tool1_weak_points_*.csv`. `--show-weak N` sets how many pairs are printed.

### Tool 2 uncertainty check (NEW)

After the greedy panel, Tool 2 moves every NOT-measured Ki of the panel over 0.1 uM - 10 mM,
one at a time, and recomputes J. It prints a VERDICT (SOLID / CONDITIONAL / WEAK), how many Ki
each panel OBP has measured, the bets (`J 17.17 -> 0.00 at Ki 31.6 uM`), and J if every
prediction were true at once. A missing Ki can make J look too HIGH: if the hidden compound
varies between plants more than it separates the classes, binding to it adds noise.
The limit is `--max-error` (default 0.05): the largest share of plants the panel may
misclassify, error = Phi(-sqrt(J/2)) (Gaussian readings, cut half-way), so 5 % -> J >= 5.4.
Every J in the output is printed with its misclassification rate. `--weak-frac` (fraction of
the panel's J) and `--target-score` are alternatives. Saved in `results/tool2_bets_*.csv`.

## Running both tools on the SCENTINEL CSVs (NEW)

Both tools now read **the same files as the OBP selector** (`main.py`,
`generate_report.py`) through `panel_selector/scentinel_data.py`:

| file | used for |
|---|---|
| `Compound_OBP_binding.csv` | experimental Ki (µM). Decides which cells are *measured* |
| `Compound_OBP_binding_imputed_pred.csv` | predicted mode: target (Tool 1) and every compound (Tool 2) |
| `Compound_OBP_binding_imputed_lower.csv` | predicted mode: interferents in Tool 1 (conservative, as for s2 in `main.py`) |
| `tanimoto_imputation_diagnostic.csv` | confidence c1 of each imputed cell (same formula as `confidence.py`) |
| `interferents_*.txt` | interferent list, names or CAS, `#` comments |

Put them in `./data/` or in the folder you run from (or pass `--data-dir`).

```bash
pip install numpy scipy pandas scikit-learn
# Tool 1 - minimum panel so nothing imitates the target
python scentinel_tool1_specificity.py --target "methyl salicylate" \
    --interferents data/interferents_example.txt --source experimental --start-with DcitOBP3
python scentinel_tool1_specificity.py --target 119-36-8 \
    --interferents data/interferents_matching.txt --source predicted --min-confidence 0.3 \
    --start-with DcitOBP3
# Tool 2 - panel that separates two classes of GC-MS profiles
python scentinel_tool2_discrimination.py \
    --profiles examples/example_vineyard_profiles_SYNTHETIC.csv --source predicted
```
Leave out `--target`, `--interferents`, `--source`, `--start-with` or `--profiles` and the script asks for them.
Results are also saved as CSV in `./results/`.

What the adapter does:
- Transposes the tables (compounds x OBPs -> OBPs x compounds) and converts to s = 1/Ki.
  `>20` -> 22 (same `x1.1` rule as `main.convert_ki_to_float`).
- Resolves compounds as `main.validate_voc` does: CAS, exact name, `/` synonym,
  then a unique partial match. All errors in an interferent file are listed at once.
- **Measured vs imputed comes from the experimental CSV**, never from the
  diagnostic. Imputed cells that have no diagnostic row get confidence 0.5
  (`C1_FALLBACK`), not 1.0. This matters: the current diagnostic does not match
  the imputed CSVs exactly (~19k imputed cells without a row, 649 rows on empty
  cells). The tools print this at start-up. Regenerate both files in the same run to fix it.
- `--min-confidence c` / `--only-confident` drop weak Tanimoto predictions before
  the search (measured cells are always kept).
- The output tables show, for each chosen OBP, how many cells are measured or
  imputed, their mean confidence, and the coverage (how many interferents or
  profile compounds have data).

Two changes to the shared greedy loop (backwards compatible, old tests pass):
- `tie_break`: with real data a single sensor can always be imitated (residual 0
  for every OBP), so the first pick was alphabetical. Ties now go to the
  strongest target binder.
- `lookahead`: with many interferents (e.g. the 42 in `interferents_matching.txt`)
  every 2-sensor panel still has residual 0, so plain greedy stopped after one
  OBP. When the score is stuck at exactly 0, it now tries adding 2-3 OBPs together.

### Starting OBP and selectivity tie-break (NEW)

- `--start-with OBP [OBP ...]` is **required** (if left out, the script asks for it):
  these OBPs go into the panel first (e.g. the best single protein chosen by SCENTINEL v3)
  and the search only looks for the OBPs that complement them. Names are case-insensitive,
  a unique partial name is accepted, and a typo gets suggestions. They are marked `start`
  in the output table.
- Ties between complementary OBPs go to the **most selective** one: target signal /
  strongest interferent signal on that OBP (column `selec`; missing interferent data counts
  as 0, `inf` = no interferent binds it). Equal selectivity -> strongest target binder.

```bash
python scentinel_tool1_specificity.py --target "methyl salicylate" \
    --interferents data/interferents_example.txt --source experimental --start-with DcitOBP3
```

Data caveat: an interferent with no Ki on an OBP is treated as "does not bind".
That makes the OBP look more specific than it really is. Tool 1 therefore only
considers OBPs with **c2 >= 0.5** (`--min-c2`, 1.0 = strict), where c2 is computed as
in `confidence.py`: each interferent gives 1 if measured, its Tanimoto confidence if
imputed (0.5 if it has no diagnostic row) and 0 if missing; c2 = total / number of
interferents. In experimental mode c2 is just the fraction measured; in predicted mode
weak predictions count less. It lists the interferents missing on the final panel.

### Input table (any paper)

One row per compound: `CAS | name | mean healthy | SE healthy | mean diseased | SE diseased`.
CAS is mandatory, name optional, **n.d. is read as 0**; plants per group with `--n` (default 3).
See `docs/INPUT_FORMAT.md` and `templates/vinesplit_input.xlsx`.
`tools/build_vineyard_profiles.py` writes these tables from Ciubotaru et al. 2021 (Front. Plant Sci.
12:693887, Suppl. Table 2) into `data/vineyard/` (e.g. `bianca_2017_48hpi.csv`).

```bash
python scentinel_tool2_discrimination.py --profiles data/vineyard/bianca_2017_48hpi.csv --source experimental
```

How J is computed (exact moments, nothing invented): reading = A·mean, covariance = A·diag(SD²)·Aᵀ
with SD = SE·√n (compounds assumed independent); multivariate Fisher J = dᵀ(Σh+Σd+N)⁻¹d,
N = sensor noise (`--sensor-noise`, 5 %). `>X` Ki cells count as non-binding (`--censored zero`) and
sensitivity uses the s1 scale of main.py (`--scale anchored`: 1 uM -> 1, >= 40 uM -> 0).

`examples/example_vineyard_profiles_SYNTHETIC.csv` is **invented data** (real compound
names, made-up concentrations), there only to show the profile file format.

## Relation to SCENTINEL v3

This module does **not** change the SCENTINEL code, but it uses the same input
data: `Compound_OBP_binding.csv` (Ki, compound x OBP) and the imputed tables.
The "Limitations and future directions" section of the SCENTINEL wiki already
points this way ("extending the framework from scoring single proteins to
scoring panels"); this module is a first prototype of that direction, kept
separate so it can be tested before deciding whether (and how) to merge it
into the main pipeline.

## Limitations

- Tool 1 only protects against the interferents **you** put in the library; an
  unknown compound outside the list is not covered.
- The forward model assumes a linear regime (no Langmuir-type saturation).
- Everything depends on S: if sensitivities come from unvalidated docking or
  imputation, calibrate them against a few real measurements.
- Summary mode assumes compounds vary independently within a group (the
  published tables give no correlations), and reads 'n.d.' as 0, which is only
  right if 'n.d.' means "not detected".
