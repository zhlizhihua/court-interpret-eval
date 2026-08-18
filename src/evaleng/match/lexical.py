"""Lexical matcher: did the candidate produce an acceptable wording?"""
from __future__ import annotations
import functools

import stanza
from rapidfuzz import fuzz

from evaleng.schema.models import Unit, MatchingPolicy
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict

_PIPELINE = None


def _pipeline() -> stanza.Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = stanza.Pipeline("es", processors="tokenize,pos,lemma", verbose=False)
    return _PIPELINE


@functools.lru_cache(maxsize=4096)
def _lemmas(text: str) -> tuple[str, ...]:
    doc = _pipeline()(text)
    return tuple(w.lemma.lower() for s in doc.sentences for w in s.words
                 if w.lemma and w.upos != "PUNCT")


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def match_lexical(unit: Unit, candidate: CandidateInput,
                  location: UnitLocation, policy: MatchingPolicy) -> UnitVerdict:
    region = _region_text(candidate, location)

    if region is None:
        return UnitVerdict(
            unit_id=unit.id, unit_type=unit.type, status="fail",
            candidate_text=None, method="lexical_not_located",
            reason="unit not located in the candidate (wrong wording or omission)",
            score=0.0,
        )

    cand_lemmas = set(_lemmas(region))
    for rendering in unit.acceptable_renderings:
        rlemmas = set(_lemmas(rendering))
        if rlemmas and rlemmas <= cand_lemmas:
            return UnitVerdict(
                unit_id=unit.id, unit_type=unit.type, status="pass",
                candidate_text=region, method="lemma_set",
                reason=f"lemma match to {rendering!r}", score=1.0,
            )

    best_ratio, best_rendering = 0.0, None
    for rendering in unit.acceptable_renderings:
        ratio = fuzz.token_set_ratio(region, rendering) / 100.0
        if ratio > best_ratio:
            best_ratio, best_rendering = ratio, rendering

    passed = best_ratio >= policy.fuzzy_threshold
    return UnitVerdict(
        unit_id=unit.id, unit_type=unit.type,
        status="pass" if passed else "fail",
        candidate_text=region, method="fuzzy",
        reason=(f"fuzzy {'match' if passed else 'miss'} to {best_rendering!r} "
                f"(ratio {best_ratio:.2f} {'≥' if passed else '<'} {policy.fuzzy_threshold})"),
        score=round(best_ratio, 3),
    )