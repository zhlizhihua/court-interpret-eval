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
    (UnitType.LEGAL_VOCABULARY,   MatcherKind.LEGAL),
    (UnitType.IDIOM,              MatcherKind.LEXICAL),
    (UnitType.GRAMMAR,            MatcherKind.GRAMMAR),
    (UnitType.REGISTER,           MatcherKind.REGISTER),
    (UnitType.POSITION,           MatcherKind.LEXICAL),
])
def test_matcher_routing(unit_type, expected_matcher):
    assert matcher_for(unit_type) is expected_matcher


def test_policy_defaults_when_unit_has_none(unit_by_id):
    """u006 carries no explicit policy, so it must inherit the type default."""
    policy = resolve_policy(unit_by_id("u006"))
    assert isinstance(policy, MatchingPolicy)


# def test_policy_raises_for_deferred_unit():
#     """A still-deferred type with no policy of its own must fail loudly."""
#     unit = Unit(
#         id="x_deferred",
#         type=UnitType.POSITION,
#         source_span="according to the victim",
#         acceptable_renderings=["según la víctima"],
#         matching_policy=None,
#     )
#     with pytest.raises(ValueError):
#         resolve_policy(unit)


# The fastText soft-match backup is opt-in per type (week 9). Probe on cc.es.100: sister terms
# and antonyms score above synonyms, so only `position` is allowed. Widening this list is a
# deliberate test change. Legal units never reach match_lexical (routing test below).
NO_EMBEDDINGS = [UnitType.FALSE_COGNATE, UnitType.GENERAL_VOCABULARY, UnitType.LEGAL_VOCABULARY,
                 UnitType.MODIFIER, UnitType.IDIOM, UnitType.SLANG]


@pytest.mark.parametrize("unit_type", NO_EMBEDDINGS)
def test_no_fasttext_for_trap_types(unit_type):
    assert DISPATCH[unit_type].default_policy.embedding_backup is False


def test_legal_routes_to_strict_matcher():
    assert DISPATCH[UnitType.LEGAL_VOCABULARY].matcher is MatcherKind.LEGAL


def test_only_position_gets_fasttext():
    allowed = [t for t, rule in DISPATCH.items()
               if rule.default_policy is not None and rule.default_policy.embedding_backup]
    assert allowed == [UnitType.POSITION]
