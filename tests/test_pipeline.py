"""Freeze the end-to-end behavior."""
import pytest

from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score

pytestmark = pytest.mark.slow


def _candidate(text: str) -> CandidateInput:
    text = " ".join(text.split())
    return CandidateInput(text=text, tokens=text.split())


def test_clean_transcript_passes(exam_fixture):
    report = score(_candidate(exam_fixture.reference_rendering), exam_fixture)
    r = report.result
    assert r.overall_pass is True
    assert r.coverage == (8, 10)                      # 8 scored, 2 deferred
    assert (r.weighted_passed, r.weighted_scored) == (8, 8)


def test_degraded_transcript_fails_on_the_right_units(exam_fixture):
    ref = " ".join(exam_fixture.reference_rendering.split())
    bad = (ref.replace("las pruebas", "la evidencia")
              .replace("dos cargos", "tres cargos")
              .replace("agresión con agravantes", "asalto agravado"))
    report = score(_candidate(bad), exam_fixture)
    r = report.result
    v = {verdict.unit_id: verdict.status for verdict in r.verdicts}

    assert r.overall_pass is False
    assert v["u002"] == "fail"      # false cognate (planted)
    assert v["u004"] == "fail"      # legal vocab (planted)
    assert v["u006"] == "fail"      # number (planted)
    assert v["u003"] == "pass"      # untouched — must stay passing
    assert v["u001"] == "pass"      # grammar


def test_report_has_one_line_per_unit(exam_fixture):
    report = score(_candidate(exam_fixture.reference_rendering), exam_fixture)
    unit_lines = [ln for ln in report.lines
                  if any(u.id in ln for u in exam_fixture.units)]
    assert len(unit_lines) == len(exam_fixture.units)