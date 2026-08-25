"""End-to-end freeze: grammar matcher fires on a tense break in u001.

Companion to the Week 1 pipeline freezes. Degrades ONLY the tense on u001
(future -> present) and asserts u001 flips to FAIL via the grammar_feature
method, with every other unit unchanged from the clean run.
"""
from __future__ import annotations
from pathlib import Path

import yaml
import pytest

from evaleng.schema.models import Fixture
from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score

# Adjust if your Week 1 freezes locate the fixture differently (conftest, etc.).
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "reference_draft.yaml"

# Stanza (depparse) load makes this slow; align with your e2e-freeze convention.
pytestmark = pytest.mark.slow


def _load_fixture() -> Fixture:
    return Fixture(**yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8")))


def _candidate(text: str) -> CandidateInput:
    # Mirror the CLI's transcript handling: collapse whitespace, split on it.
    # If cli.py normalizes differently, factor that into a shared helper so the
    # freeze exercises the exact path the CLI does.
    norm = " ".join(text.split())
    return CandidateInput(text=norm, tokens=norm.split())


def _verdict(report, unit_id: str):
    return next(v for v in report.result.verdicts if v.unit_id == unit_id)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return _load_fixture()


@pytest.fixture(scope="module")
def reference(fixture: Fixture) -> str:
    return " ".join(fixture.reference_rendering.split())


def test_u001_passes_on_clean(fixture: Fixture, reference: str):
    """Paired guard: on the clean rendering u001 must PASS via grammar_feature.

    Without this, the FAIL freeze below could pass for the wrong reason.
    """
    report = score(_candidate(reference), fixture)
    v = _verdict(report, "u001")
    assert v.status == "pass"
    assert v.method == "grammar_feature"


def test_u001_fails_on_broken_tense(fixture: Fixture, reference: str):
    """Future -> present on u001 only; u001 must FAIL via grammar_feature."""
    # Guard: if the corrected exemplar moved this span, fail loudly rather than
    # silently produce a transcript identical to clean (a vacuous pass).
    assert "será sentenciado" in reference, "u001 span moved — update this freeze"
    degraded = reference.replace("será sentenciado", "es sentenciado")
    assert degraded != reference

    report = score(_candidate(degraded), fixture)

    v = _verdict(report, "u001")
    assert v.status == "fail"
    assert v.method == "grammar_feature"

    # Isolation: nothing else changed, so no other unit should differ from clean.
    clean = score(_candidate(reference), fixture)
    clean_status = {x.unit_id: x.status for x in clean.result.verdicts}
    for x in report.result.verdicts:
        if x.unit_id == "u001":
            continue
        assert x.status == clean_status[x.unit_id], (
            f"{x.unit_id} changed under a u001-only edit — check for span bleed"
        )