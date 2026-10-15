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
    )
    trace = [step.score for step in result.trace]
    return SpecificityResult(result, trace)
