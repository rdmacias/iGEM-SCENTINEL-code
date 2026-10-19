"""
Shared greedy panel-building loop.

Both tools -- "make sure nothing imitates my target" and "separate two
biological states" -- reduce to the same pipeline: library, forward model,
greedy optimizer, with only the objective function swapped (see
IDEA_SOFTWARE.pdf, section "What they have in common"). This module is that shared loop; the
two tool modules only supply score_fn.

The stop conditions mirror the three stopping points the document walks through,
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
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple


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
    tie_break: Optional[Callable[[str], float]] = None,
    tie_tol: float = 1e-12,
    lookahead: int = 1,
    lookahead_pool: int = 40,
    initial: Optional[Sequence[str]] = None,
    pick_key: Optional[Callable[[str, float, float], object]] = None,
) -> GreedyResult:
    """Generic greedy set-selection: repeatedly add whichever remaining candidate
    most improves score_fn(panel), until one of the stop conditions above fires.

    score_fn(panel) -> float to maximize (or minimize, if higher_is_better=False).
    It is called once per remaining candidate at each step, so keep it cheap
    (both tool modules only re-solve a small NNLS / re-project a small matrix,
    which is fine).

    tie_break(obp) -> float, optional: when several candidates give the same
    score (within tie_tol) the one with the highest tie_break wins, instead of
    whichever comes first in `candidates`. On real data this matters for the
    very first pick of Tool 1: a single sensor can always be imitated
    (residual 0 for every OBP), so without a tie-break the first OBP would be
    chosen alphabetically.

    lookahead (default 1 = plain greedy): if the best single addition leaves
    the score EXACTLY unchanged (a flat plateau, not merely a small gain), try
    adding 2..lookahead OBPs at once, drawn from the best `lookahead_pool`
    remaining candidates (ranked by tie_break). In Tool 1 with many
    interferents the residual is 0 for every 2-sensor panel (the interferent
    cone fills the whole quadrant), so plain greedy would stop after one OBP
    even though a 3- or 4-sensor panel can escape the cone.

    initial (optional): OBPs that are forced into the panel before the search
    starts (e.g. the best single protein chosen by SCENTINEL v3). The loop then
    only looks for the OBPs that complement them. They appear first in the
    trace with gain = NaN, and count towards max_sensors.

    pick_key(obp, gain, score) -> sortable, optional: changes WHICH useful OBP is
    added. Among the candidates whose gain >= min_gain, the one with the highest
    pick_key wins (instead of the highest score). Used by Tool 1's measured-first
    mode: any OBP that really helps qualifies, and among those the best-measured
    one is preferred. If no candidate reaches min_gain the normal rules apply
    (lookahead, diminishing_returns).
    """
    panel: List[str] = []
    remaining = list(candidates)
    trace: List[GreedyStep] = []
    current_score = float("-inf") if higher_is_better else float("inf")

    for obp in (initial or []):
        if obp in panel:
            continue
        panel.append(obp)
        if obp in remaining:
            remaining.remove(obp)
        current_score = score_fn(panel)
        trace.append(GreedyStep(obp, list(panel), current_score, float("nan")))
    if panel and target_score is not None:
        crossed = (current_score > target_score) if higher_is_better else (current_score < target_score)
        if crossed:
            return GreedyResult(panel, trace, "target_crossed")

    while remaining:
        if max_sensors is not None and len(panel) >= max_sensors:
            return GreedyResult(panel, trace, "max_sensors_reached")

        best_obp, best_score, best_gain = None, None, None
        keyed = None
        for obp in remaining:
            s = score_fn(panel + [obp])
            gain = (s - current_score) if higher_is_better else (current_score - s)
            if pick_key is not None and panel and gain >= min_gain:
                k = pick_key(obp, gain, s)
                if keyed is None or k > keyed[0]:
                    keyed = (k, obp, s, gain)
            if best_score is not None and abs(s - best_score) <= tie_tol:
                better = tie_break is not None and tie_break(obp) > tie_break(best_obp)
            else:
                better = (
                    best_score is None
                    or (higher_is_better and s > best_score)
                    or (not higher_is_better and s < best_score)
                )
            if better:
                best_obp, best_score, best_gain = obp, s, gain

        if keyed is not None:
            _, best_obp, best_score, best_gain = keyed

        # current_score starts at +-inf, so the first pick always has "infinite"
        # gain -- only check diminishing returns once the panel is non-empty.
        if panel and best_gain is not None and best_gain < min_gain:
            combo = None
            if lookahead > 1 and abs(best_gain) <= tie_tol:
                room = lookahead if max_sensors is None else min(lookahead, max_sensors - len(panel))
                combo = _plateau_lookahead(panel, remaining, score_fn, current_score, room,
                                           lookahead_pool, min_gain, higher_is_better, tie_break)
            if combo is None:
                return GreedyResult(panel, trace, "diminishing_returns")
            prev = current_score
            for obp in combo:
                panel.append(obp)
                remaining.remove(obp)
                s = score_fn(panel)
                gain = (s - prev) if higher_is_better else (prev - s)
                trace.append(GreedyStep(obp, list(panel), s, gain))
                prev = s
            current_score = prev
            if target_score is not None:
                crossed = (current_score > target_score) if higher_is_better else (current_score < target_score)
                if crossed:
                    return GreedyResult(panel, trace, "target_crossed")
            continue

        panel.append(best_obp)
        remaining.remove(best_obp)
        current_score = best_score
        trace.append(GreedyStep(best_obp, list(panel), best_score, best_gain))

        if target_score is not None:
            crossed = (current_score > target_score) if higher_is_better else (current_score < target_score)
            if crossed:
                return GreedyResult(panel, trace, "target_crossed")

    return GreedyResult(panel, trace, "library_exhausted")


def _plateau_lookahead(panel, remaining, score_fn, current_score, max_size, pool_size,
                       min_gain, higher_is_better, tie_break) -> Optional[Tuple[str, ...]]:
    """Best set of 2..max_size candidates to add together when single additions
    are stuck on a flat plateau. Returns None if no set gains >= min_gain."""
    pool: Sequence[str] = list(remaining)
    if tie_break is not None:
        pool = sorted(pool, key=tie_break, reverse=True)
    pool = pool[:pool_size]
    for size in range(2, max_size + 1):
        best, best_s = None, None
        for combo in combinations(pool, size):
            s = score_fn(panel + list(combo))
            if best_s is None or (s > best_s if higher_is_better else s < best_s):
                best, best_s = combo, s
        if best is not None:
            gain = (best_s - current_score) if higher_is_better else (current_score - best_s)
            if gain >= min_gain:
                # order the set so the trace shows a sensible build-up
                order = sorted(best, key=lambda o: score_fn(panel + [o]),
                               reverse=higher_is_better)
                return tuple(order)
    return None
