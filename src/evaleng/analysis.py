"""The one Stanza pipeline for Spanish, shared by every stage that needs it.
"""
from __future__ import annotations

import stanza

_PIPELINE = None
_PROCESSORS = "tokenize,pos,lemma,depparse"


def pipeline() -> stanza.Pipeline:
    """The Spanish pipeline, built on first use and reused thereafter.

    Includes depparse because the grammar matcher (and, from W3, the register
    matcher) needs dependency relations, and Stanza fixes a pipeline's
    processors at construction time.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = stanza.Pipeline("es", processors=_PROCESSORS, verbose=False)
    return _PIPELINE