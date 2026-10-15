"""
Shared greedy panel-building loop.

Both tools -- "make sure nothing imitates my target" and "separate two
biological states" -- reduce to the same pipeline: library, forward model,
greedy optimizer, with only the objective function swapped (see
IDEA_SOFTWARE.pdf, "Lo que las une"). This module is that shared loop; the
two tool modules only supply score_fn.

The stop conditions mirror the three "paradas" the document walks through,
in priority order:

  1. target_score crossed -- the real win. For Tool 1 that's the
     best-imitation residual clearing the sensor noise floor; for Tool 2
     it's separation clearing whatever level you've decided counts as
     "usably discriminating".
  2. diminishing_returns -- the next-best OBP barely moves the score. Its
     sensitivity row is too similar to what's already in the panel (an
     ~redundant equation); adding it costs money/effort for ~no extra
     safety, so it's reported but not added.
  3. library_exhausted -- every candidate got added and target_score was
     never crossed. This is not a bug, it's an honest verdict: the target
     can't be separated from its interferents (or the two classes can't be
     told apart) with the proteins you currently have.
  4. max_sensors_reached -- an optional hard cap you set yourself, for
     cost/practicality reasons unrelated to the science.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class GreedyStep:
    added: str
    panel_so_far: List[str]
    score: float
    gain: float


@dataclass
class GreedyResult:
    panel: List[str]
    trace: List[GreedyStep]
    stop_reason: str  # "target_crossed" | "diminishing_returns" | "library_exhausted" | "max_sensors_reached"


def greedy_select(
    candidates: List[str],
    score_fn: Callable[[List[str]], float],
    *,
    target_score: Optional[float] = None,
    min_gain: float = 1e-3,
    max_sensors: Optional[int] = None,
    higher_is_better: bool = True,
) -> GreedyResult:
    """Generic greedy set-selection: repeatedly add whichever remaining candidate
    most improves score_fn(panel), until one of the stop conditions above fires.

    score_fn(panel) -> float to maximize (or minimize, if higher_is_better=False).
    It is called once per remaining candidate at each step, so keep it cheap
    (both tool modules only re-solve a small NNLS / re-project a small matrix,
    which is fine).
    """
    panel: List[str] = []
    remaining = list(candidates)
    trace: List[GreedyStep] = []
    current_score = float("-inf") if higher_is_better else float("inf")

    while remaining:
        if max_sensors is not None and len(panel) >= max_sensors:
            return GreedyResult(panel, trace, "max_sensors_reached")

        best_obp, best_score, best_gain = None, None, None
        for obp in remaining:
            s = score_fn(panel + [obp])
            gain = (s - current_score) if higher_is_better else (current_score - s)
            better = (
                best_score is None
                or (higher_is_better and s > best_score)
                or (not higher_is_better and s < best_score)
            )
            if better:
                best_obp, best_score, best_gain = obp, s, gain

        # current_score starts at +-inf, so the first pick always has "infinite"
        # gain -- only check diminishing returns once the panel is non-empty.
        if panel and best_gain is not None and best_gain < min_gain:
            return GreedyResult(panel, trace, "diminishing_returns")

        panel.append(best_obp)
        remaining.remove(best_obp)
        current_score = best_score
        trace.append(GreedyStep(best_obp, list(panel), best_score, best_gain))

        if target_score is not None:
            crossed = (current_score > target_score) if higher_is_better else (current_score < target_score)
            if crossed:
                return GreedyResult(panel, trace, "target_crossed")

    return GreedyResult(panel, trace, "library_exhausted")
