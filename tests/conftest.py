"""Shared pytest fixtures (test setup) for the evaluation-engine tests.

NOTE: 'fixture' here means pytest test-setup, NOT the exam Fixture model.
The exam fixture (reference_draft.yaml) is loaded by the `exam_fixture` fixture.
"""
from pathlib import Path

import pytest

from evaleng.schema.models import Fixture
from evaleng.interfaces import CandidateInput, UnitLocation
from evaleng.ingest.loader import load_fixture

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "reference_draft.yaml"


@pytest.fixture(scope="session")
def exam_fixture() -> Fixture:
    """The draft exemplar, parsed once and shared across the whole test run."""
    return load_fixture(FIXTURE_PATH)



@pytest.fixture
def unit_by_id(exam_fixture):
    """A lookup: unit_by_id('u006') -> the Unit with that id (KeyError if absent)."""
    index = {u.id: u for u in exam_fixture.units}
    def _lookup(uid: str):
        return index[uid]
    return _lookup


@pytest.fixture
def make_candidate():
    """Factory: turn a region string into a (CandidateInput, full-span location).

    Pass region=None to simulate 'not located' (an omission), for matcher tests
    that need to exercise the None-span path directly.
    """
    def _make(region):
        if region is None:
            ci = CandidateInput(text="(none)", tokens=["(none)"])
            loc = UnitLocation(unit_id="test", token_span=None, confidence=0.0)
            return ci, loc
        tokens = region.split()
        ci = CandidateInput(text=region, tokens=tokens)
        loc = UnitLocation(unit_id="test", token_span=(0, len(tokens)), confidence=1.0)
        return ci, loc
    return _make