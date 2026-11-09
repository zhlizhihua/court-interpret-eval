"""Stub localizer: a deliberately crude 'where did each unit land?' stage."""
from __future__ import annotations
import re

from evaleng.schema.models import Fixture, Unit, UnitType
from evaleng.interfaces import CandidateInput, UnitLocation, AlignmentResult
from rapidfuzz.distance import Levenshtein

_OVERLAP_FLOOR = 0.6   # min fraction of a rendering's words a window must cover
_SLACK = 2             # pad the located window this many tokens each side
_EXPAND = False        # W11 (D17): grow the projected core to its phrase and clip it to its
                       # sentence, from the transcript's Stanza parse. Off until the span sweep
                       # (docs_local/span_sweep.py) confirms the label sets on real parses.
_MAX_GROW = 4          # expansion never adds more than this many tokens on either side

# Dependents that belong to a nominal phrase: articles, adjectives, numerals, quantifiers,
# prepositions, multiword expressions.
_NP_DEPS = frozenset({"det", "amod", "nummod", "advmod", "case", "fixed", "flat", "compound"})
# Dependents that belong to a verb's core: auxiliaries, clitic se, subject, subordinator.
_VP_DEPS = frozenset({"aux", "aux:pass", "cop", "expl:pv", "expl:pass", "expl:impers", "expl",
                      "nsubj", "nsubj:pass", "mark"})
_VERBAL = frozenset({"VERB", "AUX"})


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
    # Same length as candidate.tokens: a token that normalizes to "" (a bare '—') stays as "",
    # so spans index candidate.tokens directly. Dropping it shifted every later region by one.
    cand = [_normalize(w) for w in candidate.tokens]
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
    # Same length as candidate.tokens: a token that normalizes to "" (a bare '—') stays as "",
    # so spans index candidate.tokens directly. Dropping it shifted every later region by one.
    cand = [_normalize(w) for w in candidate.tokens]
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

        core = (min(mapped), max(mapped) + 1)
        if _EXPAND:
            from evaleng.syntax import parse                # needs Stanza; only when expanding
            words = parse(tuple(candidate.tokens))
            core = _clip_to_sentence(core, mapped, words)
            core = _grow(core, unit.type, words)
        start = max(0, core[0] - _SLACK)
        end = min(len(cand), core[1] + _SLACK)
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


def _clip_to_sentence(core, mapped, words):
    """Keep only the part of the core inside the sentence holding most of its mapped tokens."""
    sent_of: dict[int, int] = {}
    for w in words:
        sent_of.setdefault(w.tok, w.sent)
    votes = [sent_of.get(i) for i in mapped]
    best = max(set(votes), key=votes.count)
    inside = [i for i in range(*core) if sent_of.get(i) == best]
    return (min(inside), max(inside) + 1) if inside else core


def _grow(core, unit_type, words):
    """Extend core (a token span) to the phrase(s) it heads; at most _MAX_GROW tokens per side.

    The core's roots are its words whose head lies outside it. Each root absorbs its dependents
    along a closed label set (_VP_DEPS for verbs, _NP_DEPS otherwise), recursively. Three one-step climbs handle cores that landed on a dependent of
    the word carrying the unit: grammar -> governing verb, number -> counted noun,
    position -> a connective head immediately before the core.
    """
    s, e = core
    inside = [w for w in words if s <= w.tok < e]
    if not inside:
        return core
    ids = {w.idx for w in inside}
    roots = [w for w in inside if w.head not in ids]
    has_verb = any(w.upos in _VERBAL for w in inside)

    climbed = []
    for r in roots:
        h = words[r.head] if r.head is not None else None
        if h is None:
            continue
        if unit_type is UnitType.GRAMMAR and not has_verb:
            while h is not None and h.upos not in _VERBAL:
                h = words[h.head] if h.head is not None else None
            if h is not None:
                climbed.append(h)
        elif unit_type is UnitType.NUMBER and r.deprel == "nummod":
            climbed.append(h)
        elif unit_type is UnitType.POSITION and h.tok == s - 1:
            climbed.append(h)

    children: dict[int, list] = {}
    for w in words:
        if w.head is not None:
            children.setdefault(w.head, []).append(w)

    take, seen = set(ids), set()

    def absorb(w, allowed):
        # Walk down from w. A child is taken if its label is allowed; core words are walked
        # through whatever their label, so material hanging under them is reached too.
        for c in children.get(w.idx, []):
            if c.idx in seen or not (c.deprel in allowed or c.idx in ids):
                continue
            seen.add(c.idx)
            take.add(c.idx)
            absorb(c, _VP_DEPS if c.upos in _VERBAL else _NP_DEPS)

    for r in roots + climbed:
        take.add(r.idx)
        seen.add(r.idx)
        absorb(r, _VP_DEPS if r.upos in _VERBAL else _NP_DEPS)

    toks = [words[i].tok for i in take]
    return (min(s, max(min(toks), s - _MAX_GROW)), max(e, min(max(toks) + 1, e + _MAX_GROW)))
