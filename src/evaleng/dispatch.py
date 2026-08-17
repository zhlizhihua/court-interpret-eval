"""Routing rules: which matcher judges each unit type, and the default policy."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel

from evaleng.schema.models import UnitType, Unit, MatchingPolicy


class MatcherKind(str, Enum):
    NUMBER = "number"
    LEXICAL = "lexical"
    DEFERRED = "deferred"


_LEXICAL_POLICY = MatchingPolicy(method="lemma_set+fuzzy", fuzzy_threshold=0.85)
_NUMBER_POLICY = MatchingPolicy(method="number_exact", fuzzy_threshold=1.0)


class DispatchRule(BaseModel):
    matcher: MatcherKind
    default_policy: MatchingPolicy | None = None
    note: str = ""


DISPATCH: dict[UnitType, DispatchRule] = {
    UnitType.NUMBER:             DispatchRule(matcher=MatcherKind.NUMBER,
                                              default_policy=_NUMBER_POLICY),
    UnitType.FALSE_COGNATE:      DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY),
    UnitType.GENERAL_VOCABULARY: DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY),
    UnitType.LEGAL_VOCABULARY:   DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY),
    UnitType.IDIOM:              DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY),
    UnitType.MODIFIER:           DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY),
    UnitType.SLANG:              DispatchRule(matcher=MatcherKind.LEXICAL,
                                              default_policy=_LEXICAL_POLICY,
                                              note="W1 scores this by set-membership; "
                                                   "flip to DEFERRED to align with the W3 register track"),
    UnitType.GRAMMAR:            DispatchRule(matcher=MatcherKind.DEFERRED,
                                              note="deferred to W2 (UD-feature checks)"),
    UnitType.REGISTER:           DispatchRule(matcher=MatcherKind.DEFERRED,
                                              note="deferred to W3 (register classifier)"),
    UnitType.POSITION:           DispatchRule(matcher=MatcherKind.DEFERRED,
                                              note="deferred to W4 (needs real alignment)"),
}

_missing = [t for t in UnitType if t not in DISPATCH]
if _missing:
    raise RuntimeError(
        f"DISPATCH is missing a rule for these unit types: {_missing}. "
        f"Every UnitType must be routed."
    )


def matcher_for(unit_type: UnitType) -> MatcherKind:
    return DISPATCH[unit_type].matcher


def resolve_policy(unit: Unit) -> MatchingPolicy:
    if unit.matching_policy is not None:
        return unit.matching_policy
    rule = DISPATCH[unit.type]
    if rule.default_policy is None:
        raise ValueError(
            f"unit {unit.id!r} is type {unit.type.value!r}, which is deferred "
            f"({rule.note}); it has no matching policy in W1"
        )
    return rule.default_policy