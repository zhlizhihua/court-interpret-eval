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


# Trap types never get the fastText soft-match backup (week 9). The flag is checked by
# match_lexical; legal units don't reach it at all, which the routing test below freezes.
NO_EMBEDDINGS = [UnitType.FALSE_COGNATE, UnitType.LEGAL_VOCABULARY, UnitType.MODIFIER,
                 UnitType.IDIOM, UnitType.SLANG]


@pytest.mark.parametrize("unit_type", NO_EMBEDDINGS)
def test_no_fasttext_for_trap_types(unit_type):
    assert DISPATCH[unit_type].default_policy.embedding_backup is False


def test_legal_routes_to_strict_matcher():
    assert DISPATCH[UnitType.LEGAL_VOCABULARY].matcher is MatcherKind.LEGAL
