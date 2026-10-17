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
    residual_with_cell,
    breaking_ki_range,
    measured_tolerance,
    complete_coverage,
)
from .discrimination import (
    DiscriminationResult,
    fisher_ratio,
    cv_classifier_accuracy,
    greedy_panel_for_discrimination,
    project_profiles,
    summary_moments,
    fisher_from_summary,
    greedy_panel_from_summary,
    summary_j_function,
    replicate_j_function,
    j_cell_scan,
    error_from_j,
    j_for_error,
)

from .scentinel_data import (
    ScentinelData,
    CompoundAmbiguous,
    CompoundNotFound,
    convert_ki_to_float,
    read_interferent_file,
    diag_confidence,
)

__all__ = [
    "ScentinelData",
    "CompoundAmbiguous",
    "CompoundNotFound",
    "convert_ki_to_float",
    "read_interferent_file",
    "diag_confidence",
    "SensitivityMatrix",
    "GreedyResult",
    "GreedyStep",
    "greedy_select",
    "SpecificityResult",
    "best_imitation_residual",
    "best_imitation_recipe",
    "greedy_panel_for_specificity",
    "monte_carlo_collision_probability",
    "residual_with_cell",
    "breaking_ki_range",
    "measured_tolerance",
    "complete_coverage",
    "DiscriminationResult",
    "fisher_ratio",
    "cv_classifier_accuracy",
    "greedy_panel_for_discrimination",
    "project_profiles",
    "summary_moments",
    "fisher_from_summary",
    "greedy_panel_from_summary",
    "summary_j_function",
    "replicate_j_function",
    "j_cell_scan",
    "error_from_j",
    "j_for_error",
]
