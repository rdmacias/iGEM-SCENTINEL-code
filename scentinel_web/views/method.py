import streamlit as st

import ui

ui.page_header(None, "Method & data", "How every number is computed, where the data come from, and how missing "
                                      "binding constants are handled.")

c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown('<div class="scn-h3">Data</div>', unsafe_allow_html=True)
        st.markdown("""
- 621 VOCs × 437 OBPs: binding constants Ki (µM) from published competitive-fluorescence assays
  (`Compound_OBP_binding.csv`, 7,852 measured values). Lower Ki = stronger binding. `>X` = no binding seen up to X µM.
- Tanimoto read-across fills gaps from structurally similar compounds (`*_imputed_pred.csv`, central estimate;
  `*_imputed_lower.csv`, conservative). Each prediction has a confidence c1 (0–1) from `tanimoto_diag_table.csv`.
- Vineyard VOC profiles: Ciubotaru et al. 2021 (*Vitis* 'Bianca', 0–96 h post-infection).
""")
    with st.container(border=True):
        st.markdown(f'<div class="scn-h3"><img src="data:image/png;base64,{ui.ICONS["diana"]}" '
                    'style="width:18px;vertical-align:-3px;margin-right:6px">Diana</div>', unsafe_allow_html=True)
        st.markdown("""
Score = weighted sum of four sub-scores: s1 affinity (Ki 1 µM → 1, 40 µM → 0, log scale), s2 selectivity
min(1, Ki interferent / (Ki target × τ)), s4 stability (OBP family) and s5 promiscuity.
Confidence weighs how much evidence backs each sub-score; conservative score = score × confidence.
Weight perturbation and pure SMAA test how robust the ranking is to the weights.
""")
with c2:
    with st.container(border=True):
        st.markdown(f'<div class="scn-h3"><img src="data:image/png;base64,{ui.ICONS["proteus"]}" '
                    'style="width:18px;vertical-align:-3px;margin-right:6px">Proteus</div>', unsafe_allow_html=True)
        st.markdown("""
Signal = 1/Ki. For a panel P, the residual = min over concentrations c ≥ 0 of ‖A·c − b‖ (NNLS): the distance
between the target's signals *b* and the best fake mixture of interferents *A·c*. The panel works when it is above
the sensor noise (0.05 ≈ 5 points of occupancy at target 1 µM). Candidates need c2 ≥ 0.5 and must bind something
(OBPs with only `>X` values are removed). With experimental data the search prefers the best-measured OBP that
helps (gain × c2²); with Tanimoto data, the highest residual. Every interferent must bind at least one panel OBP.
""")
    with st.container(border=True):
        st.markdown(f'<div class="scn-h3"><img src="data:image/png;base64,{ui.ICONS["dionysus"]}" '
                    'style="width:18px;vertical-align:-3px;margin-right:6px">Dionysus</div>', unsafe_allow_html=True)
        st.markdown("""
Signal ('anchored'): log10(40/Ki) / log10(40), clipped to 0–1. Reading = Σ signal × amount. With mean ± SD per
compound: J = dᵀ (Σ_healthy + Σ_diseased + N)⁻¹ d (N = 5 % sensor noise). One OBP:
J = (difference of means)² / (spread healthy + spread diseased). plants misclassified = Φ(−√(J/2))
(5 % → J ≥ 5.4).
""")

ui.section("Missing binding constants: verdicts and bets")
with st.container(border=True):
    st.markdown("""
During the search a missing Ki counts as 'does not bind'. That can make a panel look better than it is, so after
the search Proteus and Dionysus move every unmeasured Ki of the panel from 0.1 µM to 10 mM, one at a time (and,
in Dionysus, each compound on all panel OBPs at once), and report:

| Verdict | Meaning |
|---|---|
| **SOLID** | no single unverified value can plausibly break the panel |
| **CONDITIONAL** | it rests on unverified values, but the Tanimoto predictions do not contradict them |
| **WEAK** | a bet goes against the prediction, or the panel fails if every prediction is true at once: measure first |

Each bet is a sentence such as "hexanal binds DcitOBP3 weakly or not at all (Ki > 7.9 µM), AGAINST prediction
(Ki 3.8 µM, confidence 0.47)": what must be true for the panel to work, and what the database says about it.
""")
st.caption("Results are computational predictions meant to prioritise lab work.")
