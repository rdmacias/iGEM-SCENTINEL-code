"""
panel_selector -- standalone module implementing the two panel-selection
tools sketched in IDEA_SOFTWARE.pdf, built on the shared sensitivity-matrix
core described there. Independent from SCENTINEL v3 (no shared code), but
reads the same kind of Ki table (OBP x compound, uM) if you want to point it
at real data later.

Tool 1 (specificity.py): smallest OBP panel such that no non-negative
    mixture of interferents can imitate a target VOC's response, via NNLS
    residuals.
Tool 2 (discrimination.py): smallest OBP panel that best separates two
    classes of real VOC profiles (e.g. diseased vs healthy).

Both are the same greedy_core.greedy_select loop with the objective
swapped -- see greedy_core.py.
"""
from .model import SensitivityMatrix
from .greedy_core import GreedyResult, GreedyStep, greedy_select
from .specificity import (
    SpecificityResult,
    best_imitation_residual,
    best_imitation_recipe,
    greedy_panel_for_specificity,
    monte_carlo_collision_probability,
)
from .discrimination import (
    DiscriminationResult,
    fisher_ratio,
    cv_classifier_accuracy,
    greedy_panel_for_discrimination,
    project_profiles,
)

__all__ = [
    "SensitivityMatrix",
    "GreedyResult",
    "GreedyStep",
    "greedy_select",
    "SpecificityResult",
    "best_imitation_residual",
    "best_imitation_recipe",
    "greedy_panel_for_specificity",
    "monte_carlo_collision_probability",
    "DiscriminationResult",
    "fisher_ratio",
    "cv_classifier_accuracy",
    "greedy_panel_for_discrimination",
    "project_profiles",
]
