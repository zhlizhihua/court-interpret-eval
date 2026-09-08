"""Stub localizer: a deliberately crude 'where did each unit land?' stage."""
from __future__ import annotations
import re

from evaleng.schema.models import Fixture, Unit
from evaleng.interfaces import CandidateInput, UnitLocation, AlignmentResult
from rapidfuzz.distance import Levenshtein

_OVERLAP_FLOOR = 0.6   # min fraction of a rendering's words a window must cover
_SLACK = 2             # pad the located window this many tokens each side


def _normalize(token: str) -> str:
    """Lowercase and strip surrounding punctuation for crude comparison."""
    return re.sub(r"[^\w]", "", token.lower())


def _tokenize(text: str) -> list[str]:
    """Split a rendering into normalized word tokens, dropping empties."""
    return [t for t in (_normalize(w) for w in text.split()) if t]


def _best_window(cand: list[str], probe: list[str]) -> tuple[float, int, int]:
    """Best-overlapping window in `cand` for one rendering `probe`."""
    probe_set = set(probe)
    wlen = len(probe)
    if not probe_set or not cand:
        return (0.0, 0, 0)
    if wlen >= len(cand):
        return (len(set(cand) & probe_set) / len(probe_set), 0, len(cand))
    best = (0.0, 0, wlen)
    for start in range(0, len(cand) - wlen + 1):
        window = cand[start:start + wlen]
        score = len(set(window) & probe_set) / len(probe_set)
        if score > best[0]:
            best = (score, start, start + wlen)
    return best


def _localize_overlap(candidate: CandidateInput, fixture: Fixture) -> AlignmentResult:
    """Locate each unit's region in the candidate by naive token overlap."""
    cand = [t for t in (_normalize(w) for w in candidate.tokens) if t]
    locations: dict[str, UnitLocation] = {}

    for unit in fixture.units:
        best_score, best_start, best_end = 0.0, 0, 0
        for rendering in unit.acceptable_renderings:
            score, start, end = _best_window(cand, _tokenize(rendering))
            if score > best_score:
                best_score, best_start, best_end = score, start, end

        if best_score >= _OVERLAP_FLOOR:
            span = (max(0, best_start - _SLACK), min(len(cand), best_end + _SLACK))
        else:
            span = None   # not found → omission

        locations[unit.id] = UnitLocation(
            unit_id=unit.id,
            token_span=span,
            confidence=round(best_score, 3),
        )

    return AlignmentResult(locations=locations)


def localize(candidate: CandidateInput, fixture: Fixture) -> AlignmentResult:
    """Locate each unit by projecting its reference span onto the candidate.

    Anchor unit spans on the reference, align candidate<->reference,
    then project each span and pad it with slack. Units unanchored on the reference
    fall back to token-overlap.
    """
    ref = _tokenize(fixture.reference_rendering)
    cand = [t for t in (_normalize(w) for w in candidate.tokens) if t]
    ref_spans = _reference_spans(fixture)
    proj = _projection_map(ref, cand)

    locations: dict[str, UnitLocation] = {}
    for unit in fixture.units:
        rspan = ref_spans[unit.id]
        if rspan is None:                                   # not on the reference -> fall back
            locations[unit.id] = _locate_by_overlap(cand, unit)
            continue

        r0, r1 = rspan
        mapped = [proj[i] for i in range(r0, r1) if i in proj]
        if not mapped:                                      # every ref token fell in a gap -> omission
            locations[unit.id] = UnitLocation(unit_id=unit.id, token_span=None, confidence=0.0)
            continue

        start = max(0, min(mapped) - _SLACK)
        end = min(len(cand), max(mapped) + 1 + _SLACK)
        confidence = len(mapped) / (r1 - r0)                # fraction of the ref span that aligned
        locations[unit.id] = UnitLocation(
            unit_id=unit.id, token_span=(start, end), confidence=round(confidence, 3),
        )

    return AlignmentResult(locations=locations)


def _find_sublist(haystack: list[str], needle: list[str]) -> tuple[int, int] | None:
    """First contiguous position of `needle` inside `haystack`, as (start, end); else None."""
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return (i, i + len(needle))
    return None


def _reference_spans(fixture: Fixture) -> dict[str, tuple[int, int] | None]:
    """Where each unit's wording sits in the reference, as a token span.
    """
    ref = _tokenize(fixture.reference_rendering)
    spans: dict[str, tuple[int, int] | None] = {}
    for unit in fixture.units:
        found = None
        for rendering in unit.acceptable_renderings:
            found = _find_sublist(ref, _tokenize(rendering))
            if found is not None:
                break
        spans[unit.id] = found
    return spans


def _projection_map(ref: list[str], cand: list[str]) -> dict[int, int]:
    """Map each reference token index to its aligned candidate index.
    """
    proj: dict[int, int] = {}
    for op in Levenshtein.opcodes(ref, cand):
        if op.tag in ("equal", "replace"):
            for k in range(op.src_end - op.src_start):
                proj[op.src_start + k] = op.dest_start + k
    return proj


def _locate_by_overlap(cand: list[str], unit: Unit) -> UnitLocation:
    """The old token-overlap localization, for one unit. Kept as the fallback for
    units not anchored on the reference, and as the sanity-check baseline."""
    best_score, best_start, best_end = 0.0, 0, 0
    for rendering in unit.acceptable_renderings:
        score, start, end = _best_window(cand, _tokenize(rendering))
        if score > best_score:
            best_score, best_start, best_end = score, start, end
    if best_score >= _OVERLAP_FLOOR:
        span = (max(0, best_start - _SLACK), min(len(cand), best_end + _SLACK))
    else:
        span = None
    return UnitLocation(unit_id=unit.id, token_span=span, confidence=round(best_score, 3))

