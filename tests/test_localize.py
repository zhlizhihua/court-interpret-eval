"""Freeze reference-projected localization (see week4-instructions.md, Track A)."""
import pytest

import evaleng.localize as L
from evaleng.interfaces import CandidateInput
from evaleng.localize import localize


@pytest.fixture(autouse=True)
def _projection_only(monkeypatch):
    """These freeze the reference projection itself; phrase expansion has its own tests."""
    monkeypatch.setattr(L, "_EXPAND", False)


def _cand(text: str) -> CandidateInput:
    return CandidateInput(text=text, tokens=text.split())


def test_projects_number_unit_onto_the_rendered_count(exam_fixture):
    # The wrong count in the right place must be located ON the wrong count,
    # not on a stray number elsewhere (the W1 over-reach, now fixed).
    ref = " ".join(exam_fixture.reference_rendering.split())
    bad = ref.replace("dos cargos", "cinco cargos") + " y dos testigos declararon"
    loc = localize(_cand(bad), exam_fixture).locations["u006"]
    start, end = loc.token_span
    region = bad.split()[start:end]
    assert "cinco" in region and "testigos" not in region


def test_omission_reports_not_located(exam_fixture):
    text = "un texto que no contiene esa unidad en absoluto"
    assert localize(_cand(text), exam_fixture).locations["u008"].token_span is None


def test_confidence_is_populated_on_a_hit(exam_fixture):
    ref = " ".join(exam_fixture.reference_rendering.split())
    assert localize(_cand(ref), exam_fixture).locations["u008"].confidence > 0.0

def test_span_indexes_candidate_tokens_even_with_bare_punctuation(exam_fixture):
    """W11 Step 0: a token that normalizes to '' must not shift later regions."""
    text = "Según la víctima, García lo golpeó — con una pistola cargada y huyó."
    cand = _cand(text)
    s, e = localize(cand, exam_fixture).locations["u007"].token_span
    assert "cargada" in cand.tokens[s:e] and "pistola" in cand.tokens[s:e]
