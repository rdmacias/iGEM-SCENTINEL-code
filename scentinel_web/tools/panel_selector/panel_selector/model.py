"""
Shared core: sensitivity matrix + forward model.

Everything starts from two pieces of data (see IDEA_SOFTWARE.pdf):

  - A sensitivity matrix S, shape OBPs x compounds, where s_ji says how much
    protein j responds to compound i. In practice s = 1/Ki (from measured
    competitive-fluorescence tables, e.g. Compound_OBP_binding.csv in
    SCENTINEL), or a docking-predicted value for pairs you haven't measured.

  - A forward model that turns concentrations into responses. In the simple
    (linear, non-saturating) regime:

        response_j = sum_i s_ji * c_i

    i.e. a weighted sum -- exactly what SCENTINEL's own scoring already
    assumes when it treats Ki as an affinity axis. With this matrix you can
    simulate on a computer what any panel of proteins "would see" for any
    mixture, without making anything yet.

This module holds only that shared core (S, forward_response, signature). It
does not import scipy/sklearn on purpose -- specificity.py and
discrimination.py are the two tools built on top of it, and greedy_core.py is
the shared optimizer loop between them.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


class SensitivityMatrix:
    """Wraps an OBPs x compounds sensitivity table (s = 1/Ki, or a docking score).

    Missing (untested) OBP/compound pairs may be left as NaN; forward_response
    and signature() treat NaN sensitivities as 0 (no known response) rather
    than silently dropping the compound, so gaps in your data don't distort
    downstream selection -- they just mean "we currently assume no
    sensitivity here", which is the same caution SCENTINEL applies before its
    Tanimoto imputation step fills them in.
    """

    def __init__(self, data: pd.DataFrame):
        self.S = data.copy()
        self.S.index.name = self.S.index.name or "obp"
        self.S.columns.name = self.S.columns.name or "compound"

    @classmethod
    def from_csv(cls, path: str, index_col: int = 0) -> "SensitivityMatrix":
        """Load a sensitivity table directly (rows=OBP, cols=compound, values=s=1/Ki)."""
        return cls(pd.read_csv(path, index_col=index_col))

    @classmethod
    def from_ki_csv(cls, path: str, index_col: int = 0) -> "SensitivityMatrix":
        """Load a Ki (uM) table in the same layout as SCENTINEL's
        Compound_OBP_binding.csv and convert to sensitivity via s = 1/Ki.
        Untested pairs (blank cells) stay NaN."""
        ki = pd.read_csv(path, index_col=index_col)
        return cls(1.0 / ki)

    @classmethod
    def from_ki_table(cls, ki_df: pd.DataFrame) -> "SensitivityMatrix":
        return cls(1.0 / ki_df)

    @property
    def obps(self) -> List[str]:
        return list(self.S.index)

    @property
    def compounds(self) -> List[str]:
        return list(self.S.columns)

    def subset(self, obps: Iterable[str]) -> "SensitivityMatrix":
        return SensitivityMatrix(self.S.loc[list(obps)])

    def forward_response(self, concentrations: pd.Series, obps: Optional[List[str]] = None) -> np.ndarray:
        """response_j = sum_i s_ji * c_i for the given panel (default: every OBP in S)."""
        S = self.S if obps is None else self.S.loc[list(obps)]
        c = concentrations.reindex(S.columns).fillna(0.0).values
        return np.nan_to_num(S.values) @ c

    def signature(self, compound: str, concentration: float = 1.0, obps: Optional[List[str]] = None) -> np.ndarray:
        """Response vector a *pure* sample of `compound` at `concentration` gives on `obps`.
        This is the target's "firm" you're trying to protect from imitation in Tool 1."""
        c = pd.Series(0.0, index=self.compounds)
        c[compound] = concentration
        return self.forward_response(c, obps=obps)
