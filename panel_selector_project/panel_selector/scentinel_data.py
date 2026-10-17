"""
SCENTINEL data adapter -- lets Tool 1 (specificity) and Tool 2
(discrimination) run on exactly the same files as the OBP selector
(main.py / generate_report.py):

    Compound_OBP_binding.csv                 experimental Ki (uM)
    Compound_OBP_binding_imputed_pred.csv    Tanimoto-imputed Ki, central estimate
    Compound_OBP_binding_imputed_lower.csv   Tanimoto-imputed Ki, conservative (lower) bound
    tanimoto_imputation_diagnostic.csv       per imputed cell: T_max, sigma_log, n_donors, decision
    interferents_*.txt                       one name or CAS per line, '#' = comment

Layout of the binding CSVs (SCENTINEL style):
    rows    = compounds   (col 0 = CAS-number, col 1 = Compound name)
    columns = OBPs        (Ki in uM; blank = not measured; '>20' = above the
                           assay range -> 20 x 1.1, same rule as main.py)

panel_selector wants the transpose (rows = OBP, columns = compound) and
sensitivities s = 1/Ki. This module does that conversion, resolves
compounds by CAS / name / synonym the same way main.validate_voc does, and
uses the Tanimoto diagnostic to attach a confidence to every imputed cell
(same c1 formula as confidence.py), so low-confidence predictions can be
dropped before the panel search.

Measured vs imputed is decided from the EXPERIMENTAL CSV (a value there =
measured, confidence 1.0), never from the diagnostic: the diagnostic can come
from a different imputation run than the imputed CSVs, and a filled cell that
is missing from it must not silently count as "measured".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .model import SensitivityMatrix

# -- File names (same as main.py) ---------------------------------------------
BINDING_FILE_EXPERIMENTAL = "Compound_OBP_binding.csv"
BINDING_FILE_PREDICTED = "Compound_OBP_binding_imputed_lower.csv"            # selectivity / interferents
BINDING_FILE_PREDICTED_AFFINITY = "Compound_OBP_binding_imputed_pred.csv"    # affinity / target
DIAG_FILE_CANDIDATES = (
    "tanimoto_imputation_diagnostic.csv",
    os.path.join("results", "tanimoto_diag_table.csv"),
    "tanimoto_diag_table.csv",
)

# -- Confidence constants (same values as confidence.py) ----------------------
SIGMA_MAX = 1.0           # donors disagreeing by 10x in Ki -> zero confidence
USE_SUPPORT_FACTOR = True # 1 - 1/(1+n_donors)
C1_FALLBACK = 0.5         # imputed cell with no row in the diagnostic


# -- Small helpers ------------------------------------------------------------

def convert_ki_to_float(raw_value) -> float:
    """Same rule as main.convert_ki_to_float: '>X' -> X*1.1, junk -> NaN."""
    if pd.isna(raw_value):
        return np.nan
    text = str(raw_value).strip().replace("\xa0", "").replace(" ", "")
    if text.startswith(">"):
        try:
            return float(re.sub(r"[^\d.]", "", text)) * 1.1
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_interferent_file(path: str) -> List[str]:
    """Same format as main.read_interferent_file: one entry per line, '#' comments."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _synonyms(name: str) -> List[str]:
    """Split on ' / ' (a slash with whitespace on at least one side), so
    '(+/-) nerolidol' stays whole while 'a / b' gives ['a', 'b']."""
    parts = re.split(r"\s+/\s*|\s*/\s+", str(name))
    return [p.strip() for p in parts if p.strip()]


def short_name(name: str, max_len: int = 28) -> str:
    """Readable display name: the first synonym, unless it is a long IUPAC
    string -> then the shortest synonym of >= 4 chars (avoids codes like '10:Ald')."""
    parts = _synonyms(name)
    if not parts:
        return str(name)
    if len(parts[0]) <= max_len:
        return parts[0]
    ok = [p for p in parts if len(p) >= 4 and ":" not in p] or parts
    return min(ok, key=len)


def find_data_file(data_dir: str, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        p = c if os.path.isabs(c) else os.path.join(data_dir, c)
        if os.path.isfile(p):
            return p
    return None


def _read_binding_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    obps = list(raw.columns[2:])
    ki = raw[obps].apply(lambda col: col.map(convert_ki_to_float)).astype(float)
    ki.insert(0, "CAS", raw.iloc[:, 0].astype(str).str.strip())
    ki.insert(1, "Name", raw.iloc[:, 1].astype(str).str.strip())
    return ki


def diag_confidence(diag: pd.DataFrame,
                    sigma_max: float = SIGMA_MAX,
                    use_support: bool = USE_SUPPORT_FACTOR) -> pd.Series:
    """c1 = T_max x max(0, 1 - sigma/sigma_max) x (1 - 1/(1+n_donors)),
    exactly as confidence.load_diag_c1."""
    dispersion = (1.0 - diag["sigma_log"] / sigma_max).clip(lower=0.0)
    support = (1.0 - 1.0 / (1.0 + diag["n_donors"])) if use_support else 1.0
    return (diag["T_max"] * dispersion * support).clip(0.0, 1.0)


class CompoundNotFound(LookupError):
    pass


class CompoundAmbiguous(CompoundNotFound):
    """Raised when an interactive query matches several compounds."""
    def __init__(self, query: str, matches: Sequence[str], display: Optional[Sequence[str]] = None):
        self.query = str(query)
        self.matches = list(matches)
        self.display_matches = list(display) if display is not None else self.matches
        opts = "; ".join(self.display_matches[:8])
        if len(self.display_matches) > 8:
            opts += "; ..."
        super().__init__(f"'{self.query}' is ambiguous ({len(self.matches)} matches): {opts}. "
                         "Use the exact name or CAS.")


# -- Main container -----------------------------------------------------------

@dataclass
class ScentinelData:
    """Everything loaded from the SCENTINEL CSVs, aligned row-by-row.

    ki_exp / ki_pred / ki_lower : DataFrames (compounds x OBPs), float Ki in uM,
                                  indexed by `label` (the full compound name,
                                  unique in the database).
    confidence                  : same shape; 1.0 measured, c1 imputed, NaN empty.
    decision                    : same shape; 'measured' / 'confident' /
                                  'exploratory' / 'no_diag' / NaN.
    """
    meta: pd.DataFrame                       # columns: row_idx, CAS, Name, label, short
    ki_exp: pd.DataFrame
    ki_pred: Optional[pd.DataFrame]
    ki_lower: Optional[pd.DataFrame]
    confidence: pd.DataFrame
    decision: pd.DataFrame
    paths: Dict[str, Optional[str]] = field(default_factory=dict)
    diag_report: Dict[str, int] = field(default_factory=dict)
    censored: Optional[pd.DataFrame] = None   # True where the experimental CSV says '>X' (no measurable binding)

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls,
             data_dir: str = ".",
             experimental: Optional[str] = None,
             predicted_affinity: Optional[str] = None,
             predicted_lower: Optional[str] = None,
             diagnostic: Optional[str] = None,
             require_predicted: bool = False) -> "ScentinelData":
        """Load the SCENTINEL files. Missing optional files are tolerated
        (experimental-only mode); pass explicit paths to override the defaults."""
        exp_p = experimental or find_data_file(data_dir, [BINDING_FILE_EXPERIMENTAL])
        if not exp_p or not os.path.isfile(exp_p):
            raise FileNotFoundError(f"Cannot find {BINDING_FILE_EXPERIMENTAL} in '{data_dir}'.")
        pred_p = predicted_affinity or find_data_file(data_dir, [BINDING_FILE_PREDICTED_AFFINITY])
        low_p = predicted_lower or find_data_file(data_dir, [BINDING_FILE_PREDICTED])
        diag_p = diagnostic or find_data_file(data_dir, DIAG_FILE_CANDIDATES)
        if require_predicted and not (pred_p and low_p):
            raise FileNotFoundError(
                f"Predicted mode needs {BINDING_FILE_PREDICTED_AFFINITY} and "
                f"{BINDING_FILE_PREDICTED} in '{data_dir}'.")

        exp = _read_binding_csv(exp_p)
        obps = list(exp.columns[2:])
        meta = exp[["CAS", "Name"]].copy()
        meta.insert(0, "row_idx", np.arange(len(exp)))
        meta["label"] = meta["Name"]
        dup = meta["label"].duplicated(keep=False)
        meta.loc[dup, "label"] = meta.loc[dup, "Name"] + " [" + meta.loc[dup, "CAS"] + "]"
        meta["short"] = meta["Name"].map(short_name)
        labels = meta["label"].values

        def aligned(path):
            if not path:
                return None
            t = _read_binding_csv(path)
            if list(t.columns[2:]) != obps or len(t) != len(exp) or \
               not (t["Name"].values == exp["Name"].values).all():
                raise ValueError(
                    f"{path} does not line up with {exp_p} (different rows or OBP columns).\n"
                    "The imputed CSVs must come from the same compound table as the experimental one.")
            out = t[obps].copy()
            out.index = labels
            return out

        ki_exp = exp[obps].copy()
        ki_exp.index = labels
        raw_exp = pd.read_csv(exp_p)
        censored = raw_exp[obps].apply(lambda col: col.astype(str).str.strip().str.startswith(">"))
        censored.index = labels
        ki_pred = aligned(pred_p)
        ki_lower = aligned(low_p)

        # -- confidence / decision matrices ----------------------------------
        measured = ki_exp.notna()
        filled = measured.copy()
        for t in (ki_pred, ki_lower):
            if t is not None:
                filled |= t.notna()
        imputed = filled & ~measured

        conf = pd.DataFrame(np.nan, index=labels, columns=obps)
        dec = pd.DataFrame(np.nan, index=labels, columns=obps, dtype=object)
        conf[measured] = 1.0
        dec[measured] = "measured"
        conf[imputed] = C1_FALLBACK
        dec[imputed] = "no_diag"

        report = {"measured_cells": int(measured.values.sum()),
                  "imputed_cells": int(imputed.values.sum()),
                  "diag_rows": 0, "diag_used": 0,
                  "imputed_without_diag": int(imputed.values.sum()),
                  "diag_on_measured_cell": 0, "diag_on_empty_cell": 0}
        if diag_p and os.path.isfile(diag_p):
            d = pd.read_csv(diag_p)
            need = {"row_idx", "OBP", "T_max", "sigma_log", "n_donors"}
            if not need.issubset(d.columns):
                raise ValueError(f"{diag_p} lacks columns {sorted(need - set(d.columns))}")
            d = d[d["OBP"].isin(obps) & (d["row_idx"] >= 0) & (d["row_idx"] < len(exp))].copy()
            d["c1"] = diag_confidence(d).values
            r = d["row_idx"].astype(int).values
            c = np.array([obps.index(o) for o in d["OBP"]])
            is_imp = imputed.values[r, c]
            is_meas = measured.values[r, c]
            cv, dv = conf.values.copy(), dec.values.copy()
            cv[r[is_imp], c[is_imp]] = d["c1"].values[is_imp]
            if "decision" in d.columns:
                dv[r[is_imp], c[is_imp]] = d["decision"].astype(str).values[is_imp]
            else:
                dv[r[is_imp], c[is_imp]] = "diag"
            conf = pd.DataFrame(cv, index=labels, columns=obps)
            dec = pd.DataFrame(dv, index=labels, columns=obps)
            report.update(diag_rows=len(d), diag_used=int(is_imp.sum()),
                          imputed_without_diag=int(imputed.values.sum() - is_imp.sum()),
                          diag_on_measured_cell=int(is_meas.sum()),
                          diag_on_empty_cell=int((~is_imp & ~is_meas).sum()))
        else:
            diag_p = None

        return cls(meta=meta, ki_exp=ki_exp, ki_pred=ki_pred, ki_lower=ki_lower,
                   confidence=conf, decision=dec,
                   paths={"experimental": exp_p, "predicted_affinity": pred_p,
                          "predicted_lower": low_p, "diagnostic": diag_p},
                   diag_report=report, censored=censored)

    # ---------------------------------------------------------------- basics
    @property
    def obps(self) -> List[str]:
        return list(self.ki_exp.columns)

    @property
    def has_predicted(self) -> bool:
        return self.ki_pred is not None and self.ki_lower is not None

    def short(self, label: str) -> str:
        return self.meta.set_index("label").loc[label, "short"]

    def summary(self) -> str:
        r = self.diag_report
        lines = [f"Compounds: {len(self.meta)}   OBPs: {len(self.obps)}",
                 f"Measured cells: {r['measured_cells']:,}   imputed cells: {r['imputed_cells']:,}"]
        if self.paths.get("diagnostic"):
            lines.append(f"Diagnostic: {os.path.basename(self.paths['diagnostic'])} "
                         f"({r['diag_rows']:,} rows, {r['diag_used']:,} matched to imputed cells)")
            if r["imputed_without_diag"]:
                lines.append(f"  ! {r['imputed_without_diag']:,} imputed cells have no diagnostic row "
                             f"-> confidence {C1_FALLBACK} (diagnostic and CSVs from different runs?)")
            if r["diag_on_empty_cell"] or r["diag_on_measured_cell"]:
                lines.append(f"  ! {r['diag_on_empty_cell']:,} diagnostic rows point to empty cells and "
                             f"{r['diag_on_measured_cell']:,} to measured cells -> ignored")
        elif self.has_predicted:
            lines.append(f"  ! No Tanimoto diagnostic found: every imputed cell gets confidence {C1_FALLBACK}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- lookup
    def resolve(self, query: str) -> str:
        """Name or CAS -> compound label. Priority (like main.validate_voc):
        exact CAS, exact full name, exact synonym, then a UNIQUE partial match.
        Raises CompoundNotFound with suggestions otherwise (never asks
        interactively, so it works the same in scripts and CLIs)."""
        q = str(query).strip()
        m = self.meta
        hit = m[m["CAS"] == q]
        if len(hit):
            return hit["label"].iloc[0]
        ql = q.lower()
        hit = m[m["Name"].str.lower() == ql]
        if len(hit):
            return hit["label"].iloc[0]
        syn = m["Name"].map(lambda n: ql in [p.lower() for p in _synonyms(n)]
                            or ql in [p.strip().lower() for p in str(n).split("/")])
        if syn.any():
            return m.loc[syn, "label"].iloc[0]
        if not re.match(r"^\d+-\d+-\d+$", q):
            part = m[m["Name"].str.contains(q, case=False, regex=False, na=False)]
            if len(part) == 1:
                return part["label"].iloc[0]
            if len(part) > 1:
                matches = part["label"].tolist()
                display = [f"{n} (CAS {c})" for n, c in zip(part["Name"], part["CAS"])]
                raise CompoundAmbiguous(q, matches, display)
        raise CompoundNotFound(f"'{q}' not found in the database (by CAS, name or synonym).")

    def resolve_many(self, queries: Iterable[str], strict: bool = True
                     ) -> Tuple[List[str], List[str]]:
        """Resolve a list (e.g. an interferent file). Returns (labels, errors).
        strict=True raises if anything fails, listing ALL errors at once."""
        labels, errors = [], []
        for q in queries:
            try:
                lab = self.resolve(q)
                if lab not in labels:
                    labels.append(lab)
            except CompoundNotFound as e:
                errors.append(str(e))
        if errors and strict:
            raise CompoundNotFound("\n".join(errors))
        return labels, errors

    # ---------------------------------------------------------------- matrices
    def ki_table(self, kind: str = "experimental", min_confidence: float = 0.0,
                 only_confident: bool = False) -> pd.DataFrame:
        """Ki (compounds x OBPs) for kind in {'experimental', 'pred', 'lower'}.
        Imputed cells with confidence < min_confidence (or not flagged
        'confident' when only_confident) are blanked; measured cells always stay."""
        if kind in ("experimental", "exp"):
            return self.ki_exp.copy()
        table = {"pred": self.ki_pred, "lower": self.ki_lower}.get(kind)
        if table is None:
            raise ValueError(f"Ki table '{kind}' not loaded (need the imputed CSVs).")
        keep = self.decision.eq("measured") | (self.confidence >= min_confidence)
        if only_confident:
            keep &= self.decision.isin(["measured", "confident"])
        return table.where(keep)

    def sensitivity(self, kind: str = "experimental", min_confidence: float = 0.0,
                    only_confident: bool = False, censored: str = "keep",
                    scale: str = "inverse", ki_min: float = 1.0, ki_max: float = 40.0) -> SensitivityMatrix:
        """SensitivityMatrix (rows = OBP, cols = compound label), s = 1/Ki.

        censored: what to do with '>X' cells (assay saw no binding up to X uM).
          'keep' -> s = 1/(1.1 X), same rule as main.py (conservative for Tool 1:
                    an interferent is assumed to bind a little).
          'zero' -> s = 0, i.e. measured as non-binding (right for Tool 2: a
                    '>50 uM' cell must not count as the OBP 'seeing' the compound).
        scale: 'inverse'  -> s = 1/Ki (linear occupancy at low concentration).
               'anchored' -> s = log10(ki_max/Ki) / log10(ki_max/ki_min), clipped to
                             [0, 1]: the s1 scale of main.py, 1 uM = 1 and >= 40 uM = 0."""
        ki = self.ki_table(kind, min_confidence, only_confident)
        ki = ki.where(ki > 0)
        if censored == "zero" and self.censored is not None:
            ki = ki.mask(self.censored, np.inf)    # 1/inf = 0: measured, no binding
        if scale == "anchored":
            # Same affinity scale as s1 in main.py (literature anchors, Cui et al.):
            # Ki <= ki_min -> 1, Ki >= ki_max -> 0, log-linear in between.
            with np.errstate(divide='ignore'):          # censored cells: log10(0) = -inf -> clipped to 0
                sens = (np.log10(ki_max / ki) / np.log10(ki_max / ki_min)).clip(0.0, 1.0)
            return SensitivityMatrix(sens.where(ki.notna()).T)
        return SensitivityMatrix((1.0 / ki).T)

    def specificity_matrix(self, target: str, interferents: Sequence[str],
                           source: str = "experimental", min_confidence: float = 0.0,
                           only_confident: bool = False) -> SensitivityMatrix:
        """Matrix for Tool 1, mirroring main.py's predicted mode:
        target column from *_imputed_pred (central estimate, = affinity file)
        and interferent columns from *_imputed_lower (conservative: they bind
        at least this strongly, so they imitate at least this well)."""
        if source == "experimental":
            return self.sensitivity("experimental")
        S_pred = self.sensitivity("pred", min_confidence, only_confident).S
        S_low = self.sensitivity("lower", min_confidence, only_confident).S
        S = S_pred.copy()
        for i in interferents:
            if i != target:
                S[i] = S_low[i]
        return SensitivityMatrix(S)

    def cell_evidence(self, obps: Sequence[str], compounds: Sequence[str],
                      source: str = "experimental", min_confidence: float = 0.0,
                      only_confident: bool = False) -> pd.DataFrame:
        """Per OBP: how many of `compounds` it has data for, how many measured,
        how many imputed, and mean confidence of the cells actually used."""
        if source == "experimental":
            have = self.ki_exp.loc[list(compounds), list(obps)].notna()
        else:
            have = self.ki_table("pred", min_confidence, only_confident).loc[
                list(compounds), list(obps)].notna()
        meas = self.decision.loc[list(compounds), list(obps)].eq("measured") & have
        conf = self.confidence.loc[list(compounds), list(obps)].where(have)
        out = pd.DataFrame({
            "with_data": have.sum(axis=0),
            "measured": meas.sum(axis=0),
            "imputed": (have & ~meas).sum(axis=0),
            "missing": (~have).sum(axis=0),
            "mean_conf": conf.mean(axis=0),
        })
        out.index.name = "OBP"
        return out

    # ---------------------------------------------------------------- Tool 2
    def load_profiles(self, path: str, class_col: str = "class",
                      sample_col: Optional[str] = None
                      ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Read a GC-MS profile table (rows = samples, one column with the
        class label, other columns = compound concentrations named by CAS or
        name). Headers are resolved to database compounds; unknown headers are
        returned as a list of errors instead of crashing.

        Returns ({class_name: DataFrame(samples x compound labels)}, errors)."""
        df = pd.read_csv(path)
        if class_col not in df.columns:
            raise ValueError(f"Profile file needs a '{class_col}' column. Found: {list(df.columns)}")
        skip = {class_col} | ({sample_col} if sample_col else set())
        # text columns (sample IDs, notes...) are not concentrations
        skip |= {c for c in df.columns if c not in skip
                 and pd.to_numeric(df[c], errors="coerce").isna().all() and df[c].notna().any()}
        rename, errors = {}, []
        for col in df.columns:
            if col in skip:
                continue
            try:
                lab = self.resolve(col)
                if lab in rename.values():
                    errors.append(f"'{col}' maps to the same compound as another column ({lab}).")
                else:
                    rename[col] = lab
            except CompoundNotFound as e:
                errors.append(str(e))
        conc = df[list(rename)].rename(columns=rename).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        classes = {str(k): conc[df[class_col].astype(str) == str(k)].reset_index(drop=True)
                   for k in df[class_col].astype(str).unique()}
        return classes, errors

    # ---------------------------------------------------------------- Tool 2, input table
    INPUT_COLUMNS = ["CAS", "name", "mean healthy", "SE healthy", "mean diseased", "SE diseased"]
    ND_TOKENS = {"n.d.", "nd", "n.d", "", "-", "\u2013", "\u2014", "\u00ad"}

    @staticmethod
    def _cell_number(v):
        """Number from a cell ('3,31E+04' and '1518,4' accepted). None = n.d."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        t = str(v).strip()
        if t.lower() in ScentinelData.ND_TOKENS:
            return None
        return float(t.replace("\u00a0", "").replace(" ", "").replace(",", "."))

    def load_input(self, path: str, n: int = 3):
        """Read the VineSplit input table (.xlsx first sheet, or .csv), one row per compound:
            CAS | name | mean healthy | SE healthy | mean diseased | SE diseased
        CAS is mandatory (rows without it, or with a CAS not in the Ki table, are skipped
        and reported); name is optional. 'n.d.' (or an empty cell) is read as 0.
        SE = standard error with n plants per group: SD = SE * sqrt(n).
        Returns (healthy, diseased, report): two DataFrames indexed by compound label
        with columns mean, sd, n."""
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, dtype=object)
        else:
            df = pd.read_csv(path, dtype=object, keep_default_na=False)
        norm = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
        cols = {}
        for want in self.INPUT_COLUMNS:
            key = want.lower()
            if key in norm:
                cols[want] = norm[key]
            elif want != "name":
                raise ValueError(f"{path}: column '{want}' is missing. Columns needed: {self.INPUT_COLUMNS} (name optional)")
        report = {"skipped": [], "nd": []}
        rows = {"healthy": [], "diseased": []}
        for i, r in df.iterrows():
            cas = r[cols["CAS"]]
            name = str(r[cols["name"]]).strip() if "name" in cols and not pd.isna(r[cols["name"]]) else ""
            cas = "" if pd.isna(cas) else str(cas).strip()
            if not cas:
                if name:
                    report["skipped"].append((name, "no CAS"))
                continue
            try:
                label = self.resolve(cas)
            except CompoundNotFound:
                report["skipped"].append((name or cas, f"CAS {cas} not in the Ki table"))
                continue
            for grp in ("healthy", "diseased"):
                m = self._cell_number(r[cols[f"mean {grp}"]])
                se = self._cell_number(r[cols[f"SE {grp}"]])
                if m is None:
                    report["nd"].append((name or cas, f"mean {grp}")); m = 0.0
                if se is None:
                    report["nd"].append((name or cas, f"SE {grp}")); se = 0.0
                rows[grp].append({"label": label, "mean": m, "sd": se * np.sqrt(n), "n": n})
        out = [pd.DataFrame(rows[g]).drop_duplicates("label").set_index("label") for g in ("healthy", "diseased")]
        return out[0], out[1], report
