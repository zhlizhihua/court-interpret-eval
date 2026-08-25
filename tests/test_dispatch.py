"""Freeze the routing rules."""
import pytest

from evaleng.schema.models import UnitType, MatchingPolicy, Unit
from evaleng.dispatch import DISPATCH, matcher_for, resolve_policy, MatcherKind


def test_every_unit_type_is_routed():
    """The exhaustiveness guarantee: no UnitType may be left unrouted."""
    for unit_type in UnitType:
        assert unit_type in DISPATCH


@pytest.mark.parametrize("unit_type, expected_matcher", [
    (UnitType.NUMBER,             MatcherKind.NUMBER),
    (UnitType.FALSE_COGNATE,      MatcherKind.LEXICAL),
    (UnitType.LEGAL_VOCABULARY,   MatcherKind.LEXICAL),
    (UnitType.IDIOM,              MatcherKind.LEXICAL),
    (UnitType.GRAMMAR,            MatcherKind.GRAMMAR),
    (UnitType.REGISTER,           MatcherKind.DEFERRED),
    (UnitType.POSITION,           MatcherKind.DEFERRED),
])
def test_matcher_routing(unit_type, expected_matcher):
    assert matcher_for(unit_type) is expected_matcher


def test_policy_defaults_when_unit_has_none(unit_by_id):
    """u006 carries no explicit policy, so it must inherit the type default."""
    policy = resolve_policy(unit_by_id("u006"))
    assert isinstance(policy, MatchingPolicy)


def test_policy_raises_for_deferred_unit():
    """A still-deferred type with no policy of its own must fail loudly."""
    unit = Unit(
        id="x_deferred",
        type=UnitType.POSITION,          # still deferred (W4); swap to REGISTER if you prefer
        source_span="according to the victim",
        acceptable_renderings=["según la víctima"],
        matching_policy=None,
    )
    with pytest.raises(ValueError):
        resolve_policy(unit)