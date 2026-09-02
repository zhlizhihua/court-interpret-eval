"""Register matching: how formal is a rendering, and does it meet the unit's requirement?

Deliberately coarse. It reads the tú/usted distinction from Stanza's morphology
(the Polite / Person features) plus a small obscenity list, and returns one of
informal / neutral / formal.

"""
from __future__ import annotations

from pydantic import BaseModel, Field

from evaleng.analysis import pipeline
from evaleng.schema.models import RegisterClass, Unit, MatchingPolicy
from evaleng.interfaces import CandidateInput, UnitLocation, UnitVerdict


# A small, closed set of obscenity markers, matched as lowercase substrings so
# multi-word phrases work. Load-bearing for the vulgar case (D3, limitation 4);
# region-specific and meant to be extended, not exhaustive.
OBSCENITIES: frozenset[str] = frozenset({
    "ni madres", "verga", "pinche", "chingar", "chingada", "puta", "puto",
    "cabrón", "cabron", "pendejo", "mierda", "joder", "coño", "carajo",
})


class RegisterJudgment(BaseModel):
    """The classifier's read on one rendering."""
    observed: RegisterClass
    vulgar: bool = False
    evidence: list[str] = Field(default_factory=list)   # human-readable reasons


def classify(text: str) -> RegisterJudgment:
    """Classify the register of `text` as informal / neutral / formal."""
    evidence: list[str] = []

    # 1. Vulgarity — a lowercase substring scan against the marker list.
    low = text.lower()
    vulgar = False
    for marker in OBSCENITIES:
        if marker in low:
            vulgar = True
            evidence.append(f"obscenity: '{marker}'")

    # 2. Formality cues from Stanza morphology.
    has_formal = False
    has_informal = False
    doc = pipeline()(text)
    for sentence in doc.sentences:
        for word in sentence.words:
            feats = set((word.feats or "").split("|"))
            if "Polite=Form" in feats:            # usted / ustedes  -> formal
                has_formal = True
                evidence.append(f"formal address: '{word.text}' (Polite=Form)")
            elif "Person=2" in feats:             # tú / vos (NOT usted) -> informal
                has_informal = True
                evidence.append(f"tuteo: '{word.text}' (Person=2)")

    # 3. Collapse to one band. Informal (incl. vulgar) dominates; a formal cue
    #    makes it formal; otherwise neutral (the pro-drop / no-cue case).
    if vulgar or has_informal:
        observed = RegisterClass.INFORMAL
    elif has_formal:
        observed = RegisterClass.FORMAL
    else:
        observed = RegisterClass.NEUTRAL

    return RegisterJudgment(observed=observed, vulgar=vulgar, evidence=evidence)


def _region_text(candidate: CandidateInput, location: UnitLocation) -> str | None:
    """The candidate's text for this unit's located span, or None if it wasn't found."""
    if location.token_span is None:
        return None
    start, end = location.token_span
    return " ".join(candidate.tokens[start:end])


def _satisfies(observed: RegisterClass, required: RegisterClass) -> bool:
    """Does the observed register meet the requirement?

    For `formal` use a *floor* rule: formal OR neutral both pass, only informal
    fails. Spanish pro-drop makes a correct formal rendering (usted dropped) read as
    neutral, so demanding positive formality would wrongly fail it. Other
    required levels fall back to exact match.
    """
    if required is RegisterClass.FORMAL:
        return observed is not RegisterClass.INFORMAL
    return observed is required


def match_register(unit: Unit, candidate: CandidateInput,
                   location: UnitLocation, policy: MatchingPolicy) -> UnitVerdict:
    spec = policy.register_spec
    if spec is None:
        # resolve_policy should prevent this; fail loud rather than mis-score.
        return UnitVerdict(
            unit_id=unit.id, unit_type=unit.type, status="fail",
            candidate_text=None, method="register_no_spec",
            reason="register unit has no register_spec to check against", score=0.0,
        )

    region = _region_text(candidate, location)
    if region is None:
        return UnitVerdict(
            unit_id=unit.id, unit_type=unit.type, status="fail",
            candidate_text=None, method="register_not_located",
            reason="unit not located in the candidate (wrong wording or omission)", score=0.0,
        )

    judgment = classify(region)
    passed = _satisfies(judgment.observed, spec.required)
    why = "; ".join(judgment.evidence) or "no register cues found"
    return UnitVerdict(
        unit_id=unit.id, unit_type=unit.type,
        status="pass" if passed else "fail",
        candidate_text=region, method="register_stanza",
        reason=f"register {judgment.observed.value}, required {spec.required.value} — {why}",
        score=1.0 if passed else 0.0,
    )