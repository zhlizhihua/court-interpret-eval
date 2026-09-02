"""Register matching: how formal is a rendering, and does it meet the unit's requirement?

Deliberately coarse. It reads the tú/usted distinction from Stanza's morphology
(the Polite / Person features) plus a small obscenity list, and returns one of
informal / neutral / formal.

"""
from __future__ import annotations

from pydantic import BaseModel, Field

from evaleng.analysis import pipeline
from evaleng.schema.models import RegisterClass


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