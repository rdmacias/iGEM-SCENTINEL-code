"""Smoke tests on the real SCENTINEL files: the analysis functions and the PDF report."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
REAL = ROOT / "data"
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (REAL / "Compound_OBP_binding.csv").is_file(), reason="real data not present")


@pytest.fixture(scope="module")
def D():
    from panel_selector import ScentinelData
    return ScentinelData.load(str(REAL))


def test_tool1_measured_first_panel(D):
    import generate_panel_report as rep
    import scentinel_tool1_specificity as t1
    from types import SimpleNamespace
    a = SimpleNamespace(min_confidence=0.3, min_c2=0.5, noise=0.05)
    interf = D.resolve_many(["r-limonene", "L-Linalool", "hexanal", "benzaldehyde", "1-hexanol"])[0]
    run = t1.analyse_tool1(D, D.resolve("119-36-8"), interf, "experimental", ["DcitOBP3"],
                           rep.tool1_opts("experimental", a))
    assert run["panel"] == ["DcitOBP3", "AlinOBP6", "LstiGOBP2"]
    assert run["passed"] and run["grade"] == "WEAK"
    assert run["bets"]["against"].sum() == 2


def test_tool2_summary_real_table(D):
    import generate_panel_report as rep
    import scentinel_tool2_discrimination as t2
    from types import SimpleNamespace
    a = SimpleNamespace(min_confidence=0.3, sensor_noise=0.05, n=3, max_error=0.05)
    A, B, _ = D.load_input(str(ROOT / "examples" / "tool2_mesa_interferentes" / "mesa_interferentes_real_48hpi.csv"))
    run = t2.analyse_tool2_summary(D, "experimental", A, B, rep.tool2_opts("experimental", a))
    assert run["panel"][0] == "GmolPBP2"
    assert run["grade"] == "SOLID"
    assert run["J0"] == pytest.approx(13.42, abs=0.05)


def test_pdf_report_is_written(tmp_path, monkeypatch):
    import generate_panel_report as rep
    monkeypatch.setattr(sys, "argv", ["generate_panel_report.py", "--data-dir", str(REAL),
                                      "--target", "methyl salicylate",
                                      "--interferents", str(REAL / "interferents_example.txt"),
                                      "--start-with", "DcitOBP3",
                                      "--profiles", str(ROOT / "examples" / "tool2_mesa_interferentes" /
                                                        "mesa_interferentes_real_48hpi.csv"),
                                      "--out", str(tmp_path)])
    rep.main()
    pdfs = list(tmp_path.glob("SCENTINEL_panel_report_*.pdf"))
    assert len(pdfs) == 1 and pdfs[0].stat().st_size > 50_000
