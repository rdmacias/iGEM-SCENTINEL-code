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
    tie_break=None,
    lookahead: int = 1,
    lookahead_pool: int = 40,
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
        tie_break=tie_break,
        lookahead=lookahead,
        lookahead_pool=lookahead_pool,
    )
    trace = [step.score for step in result.trace]
    return DiscriminationResult(result, trace)


# ---------------------------------------------------------------------------
# Summary-statistics mode: published mean +/- SE, no replicate values.
# ---------------------------------------------------------------------------
# Many papers (e.g. Ciubotaru et al. 2021, Suppl. Table 2) only report, per
# group and compound, the mean, the standard error and n. Inventing replicate
# values from that is NOT safe: how the invented replicates of different
# compounds are paired changes the within-group spread of the simulated OBP
# readings, so J can change by 20x between pairings.
#
# Instead the readings' moments are computed exactly from the forward model,
# assuming the compounds vary independently inside a group (the only
# assumption the summary table allows):
#     mean reading      mu  = A @ mean
#     reading covariance Sig = A @ diag(SD^2) @ A.T          (SD = SE * sqrt(n))
# A is the sensitivity matrix of the panel (OBPs x compounds). The covariance
# is a full matrix: two OBPs that both read the same noisy compound get
# correlated noise, so a panel can cancel it (problem 2: an OBP that is useless
# alone can be useful as a noise reference).
#
# Multivariate Fisher ratio (Mahalanobis separation of the two class means):
#     J = d.T @ (Sig_a + Sig_b + N)^-1 @ d,   d = mu_b - mu_a
# N = sensor measurement noise, diag((noise * mean reading)^2). It is needed
# because with more OBPs than varying compounds Sig is singular and J would
# blow up on a direction no real sensor can resolve. For one OBP and noise=0
# this is exactly d^2 / (var_a + var_b).

def summary_moments(S: SensitivityMatrix, panel: List[str], mean: pd.Series, sd: pd.Series):
    A = np.nan_to_num(S.S.loc[panel, mean.index].values)
    mu = A @ mean.values
    cov = A @ np.diag(sd.values ** 2) @ A.T
    return mu, cov


def fisher_from_summary(mu_a, cov_a, mu_b, cov_b, sensor_noise: float = 0.05) -> float:
    d = np.atleast_1d(mu_b - mu_a)
    C = np.atleast_2d(cov_a + cov_b)
    level = (np.abs(np.atleast_1d(mu_a)) + np.abs(np.atleast_1d(mu_b))) / 2.0
    C = C + np.diag((sensor_noise * level) ** 2)
    if not np.any(d):
        return 0.0
    if np.allclose(C, 0):
        return float("inf")
    C = C + 1e-12 * np.trace(C) / len(d) * np.eye(len(d))
    return float(d @ np.linalg.solve(C, d))


def greedy_panel_from_summary(
    S: SensitivityMatrix,
    class_a: pd.DataFrame,
    class_b: pd.DataFrame,
    sensor_noise: float = 0.05,
    target_score: Optional[float] = None,
    candidates: Optional[List[str]] = None,
    min_gain: float = 1e-3,
    max_sensors: Optional[int] = None,
    tie_break=None,
    lookahead: int = 1,
) -> DiscriminationResult:
    """Greedy panel selection from summary statistics.
    class_a / class_b: DataFrames indexed by compound with columns 'mean' and 'sd'
    (same compounds, same order)."""
    candidates = candidates if candidates is not None else S.obps
    comps = list(class_a.index)

    def score_fn(panel: List[str]) -> float:
        mu_a, cov_a = summary_moments(S, panel, class_a.loc[comps, "mean"], class_a.loc[comps, "sd"])
        mu_b, cov_b = summary_moments(S, panel, class_b.loc[comps, "mean"], class_b.loc[comps, "sd"])
        return fisher_from_summary(mu_a, cov_a, mu_b, cov_b, sensor_noise)

    result = greedy_select(candidates, score_fn, target_score=target_score, min_gain=min_gain,
                           max_sensors=max_sensors, higher_is_better=True,
                           tie_break=tie_break, lookahead=lookahead)
    return DiscriminationResult(result, [st.score for st in result.trace])


# ---------------------------------------------------------------------------
# Weak points for Tool 2: how much J depends on Ki values nobody measured
# ---------------------------------------------------------------------------
# A missing Ki counts as 'does not bind' (sensitivity 0). In Tool 2 that can
# push J either way:
#   - missing on a compound that does not change but varies a lot between
#     plants -> the OBP's reading looks steadier than it is -> J too HIGH
#   - missing on a compound that changes with disease -> signal is lost -> J too LOW
# j_cell_scan changes ONE cell of the panel's sensitivity matrix over a range
# of Ki and records J, so the caller can say "J falls below X if Ki < Y".

def summary_j_function(class_a: pd.DataFrame, class_b: pd.DataFrame, comps: List[str],
                       sensor_noise: float = 0.05):
    """J(A) for a panel sensitivity matrix A (n_panel x n_comps), summary mode."""
    ma, sa = class_a.loc[comps, "mean"].values, class_a.loc[comps, "sd"].values
    mb, sb = class_b.loc[comps, "mean"].values, class_b.loc[comps, "sd"].values

    def J(A: np.ndarray) -> float:
        return fisher_from_summary(A @ ma, A @ np.diag(sa ** 2) @ A.T,
                                   A @ mb, A @ np.diag(sb ** 2) @ A.T, sensor_noise)
    return J


def replicate_j_function(class_a_profiles: pd.DataFrame, class_b_profiles: pd.DataFrame, comps: List[str]):
    """J(A) for a panel sensitivity matrix A (n_panel x n_comps), per-sample profiles."""
    Xa, Xb = class_a_profiles[comps].values, class_b_profiles[comps].values

    def J(A: np.ndarray) -> float:
        return fisher_ratio(Xa @ A.T, Xb @ A.T)
    return J


def j_cell_scan(J, A0: np.ndarray, r: int, j: int, signals: np.ndarray) -> np.ndarray:
    """J of the panel with cell (row r, compound column j) set to each value in `signals`."""
    out = np.empty(len(signals))
    for k, s in enumerate(signals):
        A = A0.copy()
        A[r, j] = s
        out[k] = J(A)
    return out


# ---------------------------------------------------------------------------
# J <-> fraction of plants misclassified
# ---------------------------------------------------------------------------
# With Gaussian readings, equal spread in both classes and the decision cut
# half-way between the two class means, the Mahalanobis distance between the
# classes is Delta^2 = 2 J (J uses Sig_a + Sig_b), and the share of plants put
# in the wrong class is
#     error = Phi(-Delta / 2) = Phi(-sqrt(J / 2))
# (Phi = standard normal CDF). J = 1 -> 24 %, 4 -> 8 %, 5.4 -> 5 %, 8 -> 2.3 %.
# It is an idealised figure: real readings are not perfectly Gaussian and
# n = 3 plants per group gives a rough spread.

def error_from_j(J: float) -> float:
    from scipy.stats import norm
    if not np.isfinite(J):
        return 0.0
    return float(norm.cdf(-np.sqrt(max(J, 0.0) / 2.0)))


def j_for_error(err: float) -> float:
    from scipy.stats import norm
    return float(2.0 * norm.ppf(1.0 - err) ** 2)
