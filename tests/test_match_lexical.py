"""Freeze the lexical matcher. These load Stanza, so they're marked slow."""
import pytest

from evaleng.dispatch import resolve_policy
from evaleng.match.lexical import match_lexical

pytestmark = pytest.mark.slow   # applies the 'slow' marker to every test in this file


@pytest.mark.parametrize("unit_id, region, expected_status", [
    ("u004", "agresión con agravantes", "pass"),   # exact
    ("u002", "la prueba",               "pass"),   # singular candidate vs plural rendering
    ("u004", "agresion con agravantes", "pass"),   # ASR dropped an accent (lemma OR fuzzy cover)
    ("u004", "asalto agravado",         "fail"),   # plausible but wrong (agresión has no close match)
    ("u002", "la evidencia",            "fail"),   # the false-cognate trap
])
def test_lexical_cases(unit_by_id, make_candidate, unit_id, region, expected_status):
    unit = unit_by_id(unit_id)
    candidate, location = make_candidate(region)
    verdict = match_lexical(unit, candidate, location, resolve_policy(unit))
    assert verdict.status == expected_status


def test_lexical_singular_plural_is_a_lemma_match(unit_by_id, make_candidate):
    """The case that justifies the whole lemma tier: 'la prueba' ≈ 'las pruebas'."""
    unit = unit_by_id("u002")
    candidate, location = make_candidate("la prueba")
    verdict = match_lexical(unit, candidate, location, resolve_policy(unit))
    assert verdict.status == "pass"
    assert verdict.method == "lemma_set"


def test_lexical_not_located_fails(unit_by_id, make_candidate):
    unit = unit_by_id("u002")
    candidate, location = make_candidate(None)
    verdict = match_lexical(unit, candidate, location, resolve_policy(unit))
    assert verdict.status == "fail"
    assert verdict.method == "lexical_not_located"

# ---------------------------------------------------------------- fastText backup wiring (W9)
# These never load a fastText model: they check the two switches (process + per-type policy).

def test_fasttext_off_by_default(unit_by_id, make_candidate, monkeypatch):
    """No model configured -> a position fuzzy fail stays a fuzzy fail."""
    monkeypatch.delenv("EVALENG_FASTTEXT", raising=False)
    unit = unit_by_id("u009")                       # position: "according to the victim"
    candidate, location = make_candidate("La víctima contó que")
    v = match_lexical(unit, candidate, location, resolve_policy(unit))
    assert (v.status, v.method) == ("fail", "fuzzy_cover")


def test_ineligible_type_never_touches_the_model(unit_by_id, make_candidate, monkeypatch):
    """false_cognate has no backup: a bogus model path must never be opened."""
    monkeypatch.setenv("EVALENG_FASTTEXT", "/does/not/exist.bin")
    unit = unit_by_id("u002")                       # false_cognate: "the evidence"
    candidate, location = make_candidate("la evidencia")
    v = match_lexical(unit, candidate, location, resolve_policy(unit))
    assert (v.status, v.method) == ("fail", "fuzzy_cover")


def test_eligible_type_reaches_the_backup(unit_by_id, make_candidate, monkeypatch):
    """position has the backup: with a bogus path, a fuzzy fail tries to load the model."""
    monkeypatch.setenv("EVALENG_FASTTEXT", "/does/not/exist.bin")
    unit = unit_by_id("u009")
    candidate, location = make_candidate("La víctima contó que")
    with pytest.raises(FileNotFoundError):
        match_lexical(unit, candidate, location, resolve_policy(unit))
