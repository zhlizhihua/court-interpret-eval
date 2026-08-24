"""Grammar matcher: did the candidate preserve the required grammatical feature?"""
from __future__ import annotations
import functools

import stanza

from evaleng.schema.models import Unit, MatchingPolicy, FeatureSpec
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict

_PIPELINE = None


def _pipeline() -> stanza.Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = stanza.Pipeline("es", processors="tokenize,pos,lemma,depparse",
                                    verbose=False)
    return _PIPELINE


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def _parse_feats(feats: str | None) -> dict[str, str]:
    if not feats:
        return {}
    return dict(kv.split("=", 1) for kv in feats.split("|"))


@functools.lru_cache(maxsize=4096)
def _check(region: str, spec_key: tuple) -> tuple[bool, str]:
    """Scan the parsed region for a token satisfying the feature spec.

    spec_key is a hashable flattening of FeatureSpec so lru_cache works
    (Pydantic models aren't hashable). Rebuilt into locals below.
    """
    upos, feats_items, deprel = spec_key
    feats_required = dict(feats_items)
    doc = _pipeline()(region)
    for sent in doc.sentences:
        for w in sent.words:
            if upos and w.upos != upos:
                continue
            wfeats = _parse_feats(w.feats)
            if not all(wfeats.get(k) == v for k, v in feats_required.items()):
                continue
            if deprel and w.deprel != deprel:
                continue
            detail = ", ".join(f"{k}={v}" for k, v in feats_required.items())
            rel = f", {deprel}" if deprel else ""
            return True, f"{w.text!r} carries {detail}{rel}"
    want = ", ".join(f"{k}={v}" for k, v in feats_required.items()) or upos or "feature"
    rel = f" with relation {deprel!r}" if deprel else ""
    return False, f"required {want}{rel} not found in the rendered span"


def _spec_key(spec: FeatureSpec) -> tuple:
    return (spec.upos, tuple(sorted(spec.feats.items())), spec.dep_relation_to_head)


def match_grammar(unit: Unit, candidate: CandidateInput,
                  location: UnitLocation, policy: MatchingPolicy) -> UnitVerdict:
    spec = policy.feature_spec
    if spec is None:
        # resolve_policy should prevent this, but fail loud rather than mis-score.
        return UnitVerdict(
            unit_id=unit.id, unit_type=unit.type, status="fail",
            candidate_text=None, method="grammar_no_spec",
            reason="grammar unit has no feature_spec to check against",
            score=0.0,
        )

    region = _region_text(candidate, location)
    if region is None:
        return UnitVerdict(
            unit_id=unit.id, unit_type=unit.type, status="fail",
            candidate_text=None, method="grammar_not_located",
            reason="unit not located in the candidate (wrong wording or omission)",
            score=0.0,
        )

    passed, reason = _check(region, _spec_key(spec))
    return UnitVerdict(
        unit_id=unit.id,
        unit_type=unit.type,
        status="pass" if passed else "fail",
        candidate_text=region,
        method="grammar_feature",
        reason=reason,
        score=1.0 if passed else 0.0,
    )