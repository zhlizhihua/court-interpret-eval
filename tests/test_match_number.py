"""Freeze the number matcher, including its known over-reach as an xfail."""
import pytest

from evaleng.interfaces import CandidateInput
from evaleng.localize import localize
from evaleng.match.number import match_number


@pytest.mark.parametrize("region, expected_status", [
    ("acusado de dos cargos de agresión",    "pass"),   # spoken words
    ("acusado de 2 cargos de agresión",      "pass"),   # digits
    ("acusado de tres cargos de agresión",   "fail"),   # wrong number
    ("acusado de varios cargos de agresión", "fail"),   # no number at all
])
def test_number_value_cases(unit_by_id, make_candidate, region, expected_status):
    candidate, location = make_candidate(region)
    verdict = match_number(unit_by_id("u006"), candidate, location)
    assert verdict.status == expected_status


def test_number_fails_when_not_located(unit_by_id, make_candidate):
    candidate, location = make_candidate(None)
    verdict = match_number(unit_by_id("u006"), candidate, location)
    assert verdict.status == "fail"


@pytest.mark.xfail(strict=True,
                   reason="number matcher over-reaches to a stray number elsewhere; "
                          "expected fixed by W4 reference-projected localization")
def test_number_ignores_stray_number_elsewhere(exam_fixture, unit_by_id):
    """DESIRED behavior: a wrong count must fail even if '2' appears elsewhere.
    """
    text = ("fue acusado de cinco cargos de agresión; "
            "dos testigos declararon en el juicio")
    candidate = CandidateInput(text=text, tokens=text.split())
    location = localize(candidate, exam_fixture).locations["u006"]
    verdict = match_number(unit_by_id("u006"), candidate, location)
    assert verdict.status == "fail"