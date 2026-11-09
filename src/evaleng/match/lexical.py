"""Lexical matcher: did the candidate produce an acceptable wording?"""
from __future__ import annotations
import functools
import unicodedata

from rapidfuzz import fuzz

from evaleng.schema.models import Unit, MatchingPolicy
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict
from evaleng.analysis import pipeline
from evaleng import embeddings


@functools.lru_cache(maxsize=4096)
def _lemmas(text: str) -> tuple[str, ...]:
    doc = pipeline()(text)
    return tuple(w.lemma.lower() for s in doc.sentences for w in s.words
                 if w.lemma and w.upos != "PUNCT")


def _fold(s: str) -> str:
    """Strip accents (ASR drops them)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@functools.lru_cache(maxsize=8192)
def _cover_words(text: str) -> tuple[str, ...]:
    """Lowercased, accent-folded words, punctuation stripped, determiners dropped.

    Regex tokens (embeddings.match_words), not Stanza: Stanza splits short renderings badly.
    """
    return tuple(_fold(w) for w in embeddings.match_words(text))


def fuzzy_cover(region: str, rendering: str) -> tuple[float, list[tuple[str, str, float]]] | None:
    """Weakest best-match over the rendering's words, plus the (rendering, region, sim) pairs.

    Every rendering word must have a close spelling (Indel ratio) somewhere in the region;
    extra region words (slack padding) don't hurt. A strict (min instead of mean)
    Monge–Elkan similarity. None if either side has no words.
    """
    want, got = _cover_words(rendering), _cover_words(region)
    if not want or not got:
        return None
    pairs = []
    for w in want:
        g, s = max(((c, fuzz.ratio(w, c) / 100.0) for c in got), key=lambda x: x[1])
        pairs.append((w, g, round(s, 3)))
    return min(s for _, _, s in pairs), pairs


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def _soft_match(unit: Unit, region: str):
    """Best (cosine, rendering, pairs) over the unit's acceptable renderings, or None."""
    best = None
    for rendering in unit.acceptable_renderings:
        res = embeddings.soft_coverage(region, rendering)
        if res is not None and (best is None or res[0] > best[0]):
            best = (res[0], rendering, res[1])
    return best


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

    best_ratio, best_rendering, best_pairs = 0.0, None, []
    for rendering in unit.acceptable_renderings:
        res = fuzzy_cover(region, rendering)
        if res is not None and (best_rendering is None or res[0] > best_ratio):
            best_ratio, best_rendering, best_pairs = res[0], rendering, res[1]

    passed = best_ratio >= policy.fuzzy_threshold

    if not passed and policy.embedding_backup and embeddings.enabled():
        soft = _soft_match(unit, region)
        if soft is not None:
            cos, rendering, pairs = soft
            if cos >= embeddings.COS_THRESHOLD:
                shown = ", ".join(f"{c}≈{r} {s:.2f}" for r, c, s in pairs)
                return UnitVerdict(
                    unit_id=unit.id, unit_type=unit.type, status="pass",
                    candidate_text=region, method="fasttext",
                    reason=(f"soft match to {rendering!r} ({shown}; min {cos:.2f} ≥ "
                            f"{embeddings.COS_THRESHOLD}; fuzzy {best_ratio:.2f} < "
                            f"{policy.fuzzy_threshold}) [{embeddings.model_id()}]"),
                    score=round(cos, 3),
                )

    if passed:
        w = min(best_pairs, key=lambda p: p[2])
        shown = f"weakest {w[0]}≈{w[1]} {w[2]:.2f}"
    else:   # name every word that missed, not just the weakest (often a one-letter preposition)
        missed = [f"{r}≈{c} {s:.2f}" for r, c, s in best_pairs if s < policy.fuzzy_threshold]
        shown = "missing " + ", ".join(missed) if missed else "no words"
    return UnitVerdict(
        unit_id=unit.id,
        unit_type=unit.type,
        status="pass" if passed else "fail",
        candidate_text=region,
        method="fuzzy_cover",
        reason=(f"fuzzy cover {'match' if passed else 'miss'} to {best_rendering!r} "
                f"({shown} {'≥' if passed else '<'} {policy.fuzzy_threshold})"),
        score=round(best_ratio, 3),
    )