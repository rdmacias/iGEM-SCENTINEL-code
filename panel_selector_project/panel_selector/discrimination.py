"""
Tool 2 -- which proteins separate healthy from diseased (or any two
classes) best.

Same core as Tool 1 (sensitivity matrix + forward model + greedy loop),
different objective. Instead of one target VOC, you have two classes of
real GC-MS VOC profiles (with replicates and their natural variability).
The forward model projects each real sample onto the response each
candidate OBP panel *would* give it, turning real chemistry data into
simulated sensor readings -- and separation is measured in that response
space, not on the raw VOC concentrations.

The criterion that falls out is different from, and easy to miss coming
from Tool 1: the best OBP is not the one that binds hardest, it's the one
whose response actually *differs* between the two classes. A protein that
binds tightly to something equally present in healthy and diseased samples
adds a lot of signal and zero discrimination; a protein sensitive to
compounds that shift with disease state (MeSA up, C6 aldehydes down, a
fungal alcohol appearing) is what the greedy search should surface on its
own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .greedy_core import GreedyResult, greedy_select
from .model import SensitivityMatrix

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is an optional extra here
    _HAVE_SKLEARN = False


def project_profiles(S: SensitivityMatrix, profiles: pd.DataFrame, panel: List[str]) -> np.ndarray:
    """profiles: rows = replicate GC-MS samples, columns = compound
    concentrations (same forward model as Tool 1: response = S @ c).
    Returns an (n_samples, n_panel) matrix of simulated sensor readings --
    what this panel of OBPs "would have seen" for each real sample."""
    A = np.nan_to_num(S.S.loc[panel, profiles.columns].values)  # (n_panel, n_compounds)
    return profiles.values @ A.T  # (n_samples, n_panel)


def fisher_ratio(class_a_responses: np.ndarray, class_b_responses: np.ndarray) -> float:
    """Multivariate separation score, generalizing the classic Fisher ratio
    J = (mu_a - mu_b)^2 / (var_a + var_b) to a panel of several sensors at
    once: squared distance between the class means in response space,
    scaled by the total within-class spread. No extra dependency; use this
    as the default metric, or when replicate counts are too small for
    reliable cross-validation."""
    mu_a, mu_b = class_a_responses.mean(axis=0), class_b_responses.mean(axis=0)
    var_a = class_a_responses.var(axis=0, ddof=1) if len(class_a_responses) > 1 else np.zeros_like(mu_a)
    var_b = class_b_responses.var(axis=0, ddof=1) if len(class_b_responses) > 1 else np.zeros_like(mu_b)
    denom = float(np.sum(var_a + var_b))
    if denom <= 0:
        return float("inf") if not np.allclose(mu_a, mu_b) else 0.0
    return float(np.sum((mu_a - mu_b) ** 2) / denom)


def cv_classifier_accuracy(class_a_responses: np.ndarray, class_b_responses: np.ndarray, folds: int = 5) -> float:
    """Cross-validated logistic-regression accuracy at telling the two
    classes apart from this panel's simulated responses alone. Needs
    scikit-learn and enough replicates per class for the requested number
    of folds; use fisher_ratio if either is unavailable."""
    if not _HAVE_SKLEARN:
        raise NotImplementedError("scikit-learn is not installed; use metric='fisher' instead.")
    X = np.vstack([class_a_responses, class_b_responses])
    y = np.array([0] * len(class_a_responses) + [1] * len(class_b_responses))
    n_splits = min(folds, int(np.min(np.bincount(y))))
    if n_splits < 2:
        raise ValueError("Not enough replicates per class for cross-validation; add replicates or use 'fisher'.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv)
    return float(scores.mean())


_METRICS = {"fisher": fisher_ratio, "cv_accuracy": cv_classifier_accuracy}


@dataclass
class DiscriminationResult:
    result: GreedyResult
    score_trace: List[float]

    @property
    def panel(self) -> List[str]:
        return self.result.panel

    @property
    def stop_reason(self) -> str:
        return self.result.stop_reason


def greedy_panel_for_discrimination(
    S: SensitivityMatrix,
    class_a_profiles: pd.DataFrame,
    class_b_profiles: pd.DataFrame,
    metric: str = "fisher",
    target_score: Optional[float] = None,
    candidates: Optional[List[str]] = None,
    min_gain: float = 1e-3,
    max_sensors: Optional[int] = None,
) -> DiscriminationResult:
    """Greedy panel selection for Tool 2.

    Repeatedly adds whichever remaining OBP most increases separation
    (`metric`) between the two classes' simulated responses. Stops when
    separation clears `target_score` (e.g. a Fisher ratio or cross-validated
    accuracy you've decided counts as "good enough"), when gains flatten
    (diminishing_returns), or when candidates run out (library_exhausted --
    with the proteins on hand, the two states can't be told apart; you need
    a protein sensitive to a VOC that actually shifts with disease state).

    metric: "fisher" (default, no extra dependency) or "cv_accuracy"
    (cross-validated logistic regression; needs scikit-learn + enough
    replicates per class).
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {list(_METRICS)}")
    scorer = _METRICS[metric]
    candidates = candidates if candidates is not None else S.obps

    def score_fn(panel: List[str]) -> float:
        ra = project_profiles(S, class_a_profiles, panel)
        rb = project_profiles(S, class_b_profiles, panel)
        return scorer(ra, rb)

    result = greedy_select(
        candidates,
        score_fn,
        target_score=target_score,
        min_gain=min_gain,
        max_sensors=max_sensors,
        higher_is_better=True,
    )
    trace = [step.score for step in result.trace]
    return DiscriminationResult(result, trace)
