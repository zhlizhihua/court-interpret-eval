"""Strict legal-term matcher. These load Stanza, so they're marked slow.

Two groups:
- Fixture path, on reference_draft.yaml u004 ("aggravated assault"), using the real glossary.
- Glossary path, with `concepts_for` monkeypatched to a small hand-built glossary, so the
  tests don't change every time the real CSV is reviewed.
"""
import pytest

from evaleng.dispatch import resolve_policy
from evaleng.ingest.glossary import GlossaryEntry
from evaleng.match.legal import match_legal
from evaleng.schema.models import Unit, UnitType

pytestmark = pytest.mark.slow

WA = "WA Courts Glossary of Legal Terms EN-ES (2008)"


def _entry(en, es, status="acceptable"):
    return GlossaryEntry(en_term=en, es_term=es, status=status,
                         source="" if status == "proposed" else WA)


def _judge(unit, make_candidate, region):
    candidate, location = make_candidate(region)
    return match_legal(unit, candidate, location, resolve_policy(unit))


# ---------------------------------------------------------------- fixture path (real glossary)

@pytest.mark.parametrize("region, status, method", [
    ("agresión con agravantes",               "pass", "legal_fixture"),
    ("dos cargos de agresion con agravantes", "pass", "legal_fixture"),   # ASR accent drop + slack words
    ("lesiones agravadas",                    "pass", "legal_fixture"),   # another listed rendering
    ("asalto agravado",                       "fail", "legal_strict"),    # also planted in test_pipeline
    ("agravantes con agresión",               "fail", "legal_strict"),    # lemma_set would have passed this
    ("agresión simple",                       "fail", "legal_strict"),    # 'aggravated' concept missing
])
def test_fixture_path(unit_by_id, make_candidate, region, status, method):
    v = _judge(unit_by_id("u004"), make_candidate, region)
    assert (v.status, v.method) == (status, method)


def test_unlisted_variant_fails_until_glossary_approves(unit_by_id, make_candidate):
    """'agresión con circunstancias agravantes' passed in v0.1 (lemma_set). Strict fails it
    unless an approved glossary row lists it. That's intended (see the legal D-entry)."""
    v = _judge(unit_by_id("u004"), make_candidate, "agresión con circunstancias agravantes")
    assert v.status == "fail"


def test_not_located_fails(unit_by_id, make_candidate):
    v = _judge(unit_by_id("u004"), make_candidate, None)
    assert (v.status, v.method) == ("fail", "legal_not_located")


def test_score_is_binary(unit_by_id, make_candidate):
    assert _judge(unit_by_id("u004"), make_candidate, "agresión con agravantes").score == 1.0
    assert _judge(unit_by_id("u004"), make_candidate, "asalto agravado").score == 0.0


# ---------------------------------------------------------------- glossary path (patched glossary)

WAIVER = Unit(id="t_waiver", type=UnitType.LEGAL_VOCABULARY,
              source_span="you waive your right to a jury trial",
              acceptable_renderings=["renuncia a su derecho a un juicio por jurado"])

WAIVER_GLOSSARY = {("waive",): [_entry("Waive", "Renunciar a")],
                   ("jury",): [_entry("Jury", "Jurado")]}


@pytest.fixture
def waiver_glossary(monkeypatch):
    monkeypatch.setattr("evaleng.match.legal.concepts_for", lambda span: WAIVER_GLOSSARY)


def test_glossary_pass_when_every_concept_present(waiver_glossary, make_candidate):
    # NOTE: relies on Stanza lemmatizing 'renuncia' (verb) as 'renunciar'. If this fails, print
    # _term_lemmas("renuncia a su derecho") — Stanza may tag it NOUN. The fix is a glossary row
    # for the inflected form, not code (model-coupling principle).
    v = _judge(WAIVER, make_candidate, "culpable, renuncia a su derecho a un juicio con jurado y a")
    assert (v.status, v.method) == ("pass", "legal_glossary")
    assert WA in v.reason                                  # the pass cites its source


def test_opposite_meaning_fails_on_the_missing_concept(waiver_glossary, make_candidate):
    """v0.1 scored this fuzzy 0.82, just under threshold. Strict names what's missing."""
    v = _judge(WAIVER, make_candidate, "culpable, conserva su derecho a un juicio ante jurado y a")
    assert v.status == "fail" and "waive" in v.reason


def test_missing_jury_fails(waiver_glossary, make_candidate):
    v = _judge(WAIVER, make_candidate, "culpable, pierde su derecho a un juicio y a")
    assert v.status == "fail" and "jury" in v.reason


def test_proposed_rows_never_pass(monkeypatch, make_candidate):
    monkeypatch.setattr("evaleng.match.legal.concepts_for",
                        lambda span: {("jury", "trial"): [_entry("jury trial", "juicio ante jurado", "proposed")]})
    unit = Unit(id="t_jt", type=UnitType.LEGAL_VOCABULARY, source_span="a jury trial",
                acceptable_renderings=["juicio por jurado"])
    assert _judge(unit, make_candidate, "un juicio ante jurado").status == "fail"


ASSAULT = Unit(id="t_assault", type=UnitType.LEGAL_VOCABULARY, source_span="assault",
               acceptable_renderings=["agresión"])


@pytest.fixture
def assault_glossary(monkeypatch):
    monkeypatch.setattr("evaleng.match.legal.concepts_for", lambda span: {
        ("assault",): [_entry("Assault", "Agresión"), _entry("Assault", "asalto", "rejected")]})


def test_rejected_term_explains_the_fail(assault_glossary, make_candidate):
    v = _judge(ASSAULT, make_candidate, "cargos de asalto")
    assert v.status == "fail"
    assert "known incorrect" in v.reason and "asalto" in v.reason


def test_rejected_term_never_causes_a_false_fail(assault_glossary, make_candidate):
    """A rejected string next to an approved term must not override the pass."""
    v = _judge(ASSAULT, make_candidate, "agresión, no asalto")
    assert v.status == "pass"
