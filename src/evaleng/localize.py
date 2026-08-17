"""Stub localizer: a deliberately crude 'where did each unit land?' stage."""
from __future__ import annotations
import re

from evaleng.schema.models import Fixture
from evaleng.interfaces import CandidateInput, UnitLocation, AlignmentResult

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


def localize(candidate: CandidateInput, fixture: Fixture) -> AlignmentResult:
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