"""
Tool 1 -- minimum panel so nothing imitates your target.

Formal goal: given a target VOC and a library of realistic interferents,
find the smallest OBP subset such that no non-negative combination of
interferents produces (within noise) the same response vector as the
target.

Core question per candidate panel: "does some non-negative mixture of
interferents hit the target's signature on this panel?" That's exactly a
non-negative least squares (NNLS) problem -- solve for the best possible
interferent mixture, and look at what's left over:

    residual = || A c* - b ||

where A = interferent sensitivities on the panel, b = the target's
signature on the panel, and c* is the NNLS-optimal (non-negative) mixture.
residual = 0 means some mixture perfectly fakes the target on this panel.
Because NNLS finds the *closest* possible imitation (not just checks one
guess), a residual bigger than your sensor noise is a real guarantee: no
interferent recipe -- of the ones in your library -- gets any closer than
this. That's the "cone" picture from the notes: each interferent is a
direction, every non-negative combination fills a cone, and the question is
only whether the target's direction pokes outside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .greedy_core import GreedyResult, greedy_select
from .model import SensitivityMatrix


def best_imitation_residual(
    S: SensitivityMatrix,
    panel: List[str],
    target: str,
    interferents: List[str],
    target_conc: float = 1.0,
) -> float:
    """The NNLS residual described above, for one candidate panel.

    Empty panel -> 0 (no sensors, nothing distinguishes anything).
    Panel with no interferents to imitate with -> the raw norm of the
    target's signature (nothing *can* imitate it, trivially safe).
    """
    if not panel:
        return 0.0
    b = S.signature(target, target_conc, obps=panel)
    if not interferents:
        return float(np.linalg.norm(b))
    A = np.nan_to_num(S.S.loc[panel, interferents].values)
    _, residual = nnls(A, b)
    return float(residual)


def best_imitation_recipe(
    S: SensitivityMatrix,
    panel: List[str],
    target: str,
    interferents: List[str],
    target_conc: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Same computation as best_imitation_residual, but also returns the
    NNLS-optimal interferent concentrations c* (useful for reporting *what*
    the closest fake mixture would actually be, not just how far off it is)."""
    b = S.signature(target, target_conc, obps=panel)
    A = np.nan_to_num(S.S.loc[panel, interferents].values)
    c_star, residual = nnls(A, b)
    return c_star, float(residual)


def monte_carlo_collision_probability(
    S: SensitivityMatrix,
    panel: List[str],
    target: str,
    interferents: List[str],
    noise: float,
    target_conc: float = 1.0,
    conc_range: Tuple[float, float] = (0.0, 10.0),
    n_draws: int = 20_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """P(collision): fraction of *random* interferent mixtures (drawn
    uniformly from conc_range, not just the NNLS-optimal one) whose response
    on `panel` falls within `noise` of the target's signature -- i.e. would
    read as "target present" by accident. A complementary, more empirical
    check to the residual test above (useful when you want a probability
    rather than a worst-case distance, e.g. to report a P(false positive) <= X
    the way the original spec frames it)."""
    if rng is None:
        rng = np.random.default_rng()
    if not panel or not interferents:
        return 1.0
    b = S.signature(target, target_conc, obps=panel)
    A = np.nan_to_num(S.S.loc[panel, interferents].values)
    draws = rng.uniform(conc_range[0], conc_range[1], size=(n_draws, len(interferents)))
    responses = draws @ A.T
    dist = np.linalg.norm(responses - b, axis=1)
    return float(np.mean(dist <= noise))


@dataclass
class SpecificityResult:
    result: GreedyResult
    residual_trace: List[float]

    @property
    def panel(self) -> List[str]:
        return self.result.panel

    @property
    def stop_reason(self) -> str:
        return self.result.stop_reason


def greedy_panel_for_specificity(
    S: SensitivityMatrix,
    target: str,
    interferents: List[str],
    noise: float,
    candidates: Optional[List[str]] = None,
    target_conc: float = 1.0,
    min_gain: float = 1e-3,
    max_sensors: Optional[int] = None,
    tie_break=None,
    lookahead: int = 1,
    lookahead_pool: int = 40,
    initial: Optional[List[str]] = None,
    pick_key=None,
) -> SpecificityResult:
    """Greedy panel selection for Tool 1.

    Starts from an empty panel and repeatedly adds whichever remaining OBP
    most increases the best-imitation residual, i.e. whichever one shrinks
    the "cone" of interferent mixtures the most. This automatically prefers
    *diverse* OBPs over merely *strong* ones: an OBP whose sensitivity row
    looks like ones you already picked adds a near-redundant equation and
    barely moves the residual, so the diminishing_returns stop catches it.

    Stops when the residual clears `noise` (target is safely un-fakeable by
    this interferent library), when the best available gain drops below
    `min_gain` (further sensors would be redundant), or when candidates run
    out without clearing noise (an honest "this library isn't enough, get a
    more diverse protein or fall back on physical separation" verdict).

    initial: OBPs forced into the panel first (e.g. the best single protein
    from SCENTINEL v3); the search then only adds complementary OBPs.
    """
    candidates = candidates if candidates is not None else S.obps

    def score_fn(panel: List[str]) -> float:
        return best_imitation_residual(S, panel, target, interferents, target_conc)

    result = greedy_select(
        candidates,
        score_fn,
        target_score=noise,
        min_gain=min_gain,
        max_sensors=max_sensors,
        higher_is_better=True,
        tie_break=tie_break,
        lookahead=lookahead,
        lookahead_pool=lookahead_pool,
        initial=initial,
        pick_key=pick_key,
    )
    trace = [step.score for step in result.trace]
    return SpecificityResult(result, trace)


# ---------------------------------------------------------------------------
# Weak points: which single Ki value would bring the residual back under noise
# ---------------------------------------------------------------------------
#
# The residual of a finished panel rests on every cell of the panel. Cells that
# were never measured count as "does not bind" (signal 0), measured ones as
# exact. Both can be wrong. These helpers change ONE cell at a time and ask:
# does the panel still pass?  The answer turns "8 pairs to measure" into
# "these 3 pairs can break the panel, and only if their Ki is below X uM".

DEFAULT_KI_GRID = np.logspace(-1, 4, 101)        # 0.1 uM .. 10 mM, 20 points per decade


def _panel_system(S: SensitivityMatrix, panel: List[str], target: str,
                  interferents: List[str], target_conc: float = 1.0):
    A = np.nan_to_num(S.S.loc[panel, interferents].values.astype(float))
    b = np.nan_to_num(S.S.loc[panel, target].values.astype(float)) * target_conc
    return A, b


def _residual_cell(A0, b0, r, j, signal, target_conc):
    """Residual with cell (row r, interferent column j; j=None -> target) set to `signal`."""
    if j is None:
        b = b0.copy()
        b[r] = signal * target_conc
        A = A0
    else:
        A = A0.copy()
        A[r, j] = signal
        b = b0
    if A.shape[1] == 0:
        return float(np.linalg.norm(b))
    return float(nnls(A, b)[1])


def residual_with_cell(S: SensitivityMatrix, panel: List[str], target: str,
                       interferents: List[str], obp: str, compound: str,
                       ki: float, target_conc: float = 1.0) -> float:
    """NNLS residual of `panel` after setting ONE cell (obp, compound) to
    signal 1/ki (ki = inf or <= 0 -> signal 0, i.e. 'does not bind')."""
    A, b = _panel_system(S, panel, target, interferents, target_conc)
    s = 0.0 if (not np.isfinite(ki) or ki <= 0) else 1.0 / ki
    j = None if compound == target else interferents.index(compound)
    return _residual_cell(A, b, panel.index(obp), j, s, target_conc)


def breaking_ki_range(S: SensitivityMatrix, panel: List[str], target: str,
                      interferents: List[str], obp: str, compound: str,
                      noise: float, target_conc: float = 1.0,
                      ki_grid: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
    """Scan Ki for one cell over `ki_grid` and return (lowest, highest) Ki at
    which the panel FAILS (residual < noise), or (None, None) if no Ki on the
    grid breaks it. For a missing interferent cell the failing Ki usually run
    from the bottom of the grid up to a threshold: 'breaks if Ki <= highest'.
    For a cell that already has a value it can be the other way round: the
    interferent's signal there is what keeps it from imitating, so a WEAKER
    true binding ('Ki >= lowest') breaks the panel."""
    grid = DEFAULT_KI_GRID if ki_grid is None else np.asarray(ki_grid, float)
    A0, b0 = _panel_system(S, panel, target, interferents, target_conc)
    r = panel.index(obp)
    j = None if compound == target else interferents.index(compound)
    fails = [ki for ki in grid if _residual_cell(A0, b0, r, j, 1.0 / ki, target_conc) < noise]
    if not fails:
        return None, None
    return float(min(fails)), float(max(fails))


def measured_tolerance(S: SensitivityMatrix, panel: List[str], target: str,
                       interferents: List[str], obp: str, compound: str,
                       noise: float, target_conc: float = 1.0,
                       max_factor: float = 2.0, steps: int = 100) -> Tuple[Optional[float], Optional[float]]:
    """For a cell that HAS a value: smallest relative change of its Ki that
    breaks the panel, looking down (Ki x f, f < 1) and up (Ki x f, f > 1) up to
    max_factor. Returns (down, up) as fractions, e.g. (-0.11, None) = breaks if
    the Ki is 11 % lower, never if it is up to 2x higher."""
    s0 = S.S.loc[obp, compound]
    if pd.isna(s0) or s0 <= 0:
        return None, None
    ki0 = 1.0 / float(s0)
    A0, b0 = _panel_system(S, panel, target, interferents, target_conc)
    r = panel.index(obp)
    j = None if compound == target else interferents.index(compound)
    out = []
    for direction in (-1, 1):
        found = None
        for f in np.exp(np.linspace(0, direction * np.log(max_factor), steps + 1)[1:]):
            if _residual_cell(A0, b0, r, j, 1.0 / (ki0 * f), target_conc) < noise:
                found = float(f - 1.0)
                break
        out.append(found)
    return out[0], out[1]


def complete_coverage(
    S: SensitivityMatrix,
    panel: List[str],
    target: str,
    interferents: List[str],
    candidates: List[str],
    min_cover: int = 1,
    target_conc: float = 1.0,
    known: Optional[pd.DataFrame] = None,
) -> Tuple[List[Tuple[str, List[str], float]], List[str]]:
    """Make every interferent VISIBLE: add OBPs until each interferent has a
    known value (not NaN) on at least `min_cover` OBPs of the panel.

    Why: a missing cell counts as 'does not bind'. If an interferent has no
    known value anywhere on the panel, the search cannot see it at all, and if
    it does bind the target's best OBP it can be scaled up to imitate the
    target for free. One OBP where it is known closes that door.

    Each round adds the candidate that covers the most still-uncovered
    interferents; ties -> the one giving the highest residual. Adding an OBP
    never lowers the residual (one more equation for the fake mixture to meet).

    known (optional, OBP x compound bool): which cells count as 'the OBP sees
    this interferent'. Default: any value (S not NaN). Tool 1 passes 'a real
    binding' (measured and not censored '>X', or a used prediction), since an
    OBP that does not bind the interferent would not respond to it either.

    Returns ([(obp, interferents it covered, residual after adding)], still_uncovered).
    """
    panel = list(panel)
    added = []
    known = S.S.notna() if known is None else known.reindex(index=S.S.index, columns=S.S.columns,
                                                             fill_value=False)
    while True:
        uncovered = [i for i in interferents if int(known.loc[panel, i].sum()) < min_cover]
        if not uncovered:
            return added, []
        pool = [o for o in candidates if o not in panel and known.loc[o, uncovered].any()]
        if not pool:
            return added, uncovered
        best = max(pool, key=lambda o: (int(known.loc[o, uncovered].sum()),
                                        best_imitation_residual(S, panel + [o], target, interferents,
                                                                target_conc)))
        panel.append(best)
        covered = [i for i in uncovered if known.loc[best, i]]
        added.append((best, covered,
                      best_imitation_residual(S, panel, target, interferents, target_conc)))
