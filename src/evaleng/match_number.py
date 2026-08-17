"""Number matcher: did the candidate produce the correct numeric value?"""
from __future__ import annotations
import re

from text_to_num import alpha2digit   # package 'text2num', imports as 'text_to_num'

from evaleng.schema.models import Unit
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict

_DIGITS = re.compile(r"\d+")


def _numbers_in(text: str) -> set[str]:
    """Reduce a string to the set of numeric values it contains (as digit-strings)."""
    return set(_DIGITS.findall(alpha2digit(text, "es", threshold=0)))


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    """The candidate text where this unit was located, or None if not found."""
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def match_number(unit: Unit, candidate: CandidateInput, location: UnitLocation) -> UnitVerdict:
    """Judge one number-type unit: is a required numeric value present?"""
    expected_sets = [nums for r in unit.acceptable_renderings
                     if (nums := _numbers_in(r))]

    region = _region_text(candidate, location)
    cand_numbers = _numbers_in(region) if region else set()
    widened = False
    if not cand_numbers:
        cand_numbers = _numbers_in(candidate.text)
        widened = True

    matched = any(exp <= cand_numbers for exp in expected_sets)

    if matched:
        status, score = "pass", 1.0
        reason = f"required number present (found {sorted(cand_numbers)})"
    elif not cand_numbers:
        status, score = "fail", 0.0
        reason = "no number found in the rendering (likely omitted)"
    else:
        status, score = "fail", 0.0
        expected = sorted({n for s in expected_sets for n in s})
        reason = f"expected {expected}, candidate produced {sorted(cand_numbers)}"

    if widened and status == "pass":
        reason += " [found outside the located window]"

    return UnitVerdict(
        unit_id=unit.id,
        unit_type=unit.type,
        status=status,
        candidate_text=region if region else candidate.text,
        method="number_exact",
        reason=reason,
        score=score,
    )