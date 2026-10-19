"""Strict legal-term matcher: a legal term is right only as an approved term, in order."""
from __future__ import annotations
import functools
import unicodedata

from evaleng.analysis import pipeline
from evaleng.ingest.glossary import concepts_for
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict
from evaleng.localize import _find_sublist
from evaleng.schema.models import Unit, MatchingPolicy

_DROP_UPOS = frozenset({"PUNCT", "DET"})   # articles/possessives don't make a term: su/sus, el/la


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@functools.lru_cache(maxsize=4096)
def _term_lemmas(text: str) -> tuple[str, ...]:
    """Lemmas, lowercased, accent-stripped (ASR drops accents), articles and punctuation removed."""
    doc = pipeline()(text)
    return tuple(_strip_accents(w.lemma.lower()) for s in doc.sentences for w in s.words
                 if w.lemma and w.upos not in _DROP_UPOS)


def _contains(region: tuple[str, ...], term: str) -> bool:
    return _find_sublist(list(region), list(_term_lemmas(term))) is not None


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def match_legal(unit: Unit, candidate: CandidateInput,
                location: UnitLocation, policy: MatchingPolicy) -> UnitVerdict:
    def verdict(status, method, reason, text):
        return UnitVerdict(unit_id=unit.id, unit_type=unit.type, status=status,
                           candidate_text=text, method=method, reason=reason,
                           score=1.0 if status == "pass" else 0.0)

    region = _region_text(candidate, location)
    if region is None:
        return verdict("fail", "legal_not_located",
                       "legal term not located in the candidate (wrong term or omission)", None)
    reg = _term_lemmas(region)

    # 1. A fixture rendering, as a contiguous term. These are linguist-approved by definition.
    for r in unit.acceptable_renderings:
        if _contains(reg, r):
            return verdict("pass", "legal_fixture", f"term match to {r!r}", region)

    # 2. Approved glossary equivalents for EVERY legal concept in the unit.
    concepts = concepts_for(unit.source_span)
    found, missing = [], []
    for key, entries in concepts.items():
        hit = next((e for e in entries if e.approved and _contains(reg, e.es_term)), None)
        (found if hit else missing).append(hit or " ".join(key))
    if concepts and not missing:
        cited = "; ".join(f"{e.es_term!r} ({e.source})" for e in found)
        return verdict("pass", "legal_glossary", f"glossary term(s): {cited}", region)

    # 3. Fail. Say *which* concept is missing, and name a known-wrong equivalent if one is present.
    wrong = [e for entries in concepts.values() for e in entries
             if e.status == "rejected" and _contains(reg, e.es_term)]
    why = "no approved legal term found"
    if missing:
        why = f"missing approved term for: {', '.join(missing)}"
    if wrong:
        why += "; known incorrect: " + ", ".join(f"{e.es_term!r} ({e.source})" for e in wrong)
    return verdict("fail", "legal_strict", why, region)