"""Dependency parse of a whole candidate transcript, indexed by whitespace token (W11, D17).

The localizer counts whitespace tokens (candidate.tokens). Stanza counts its own words: it splits
'dólares.' into 'dólares' + '.', and the contraction 'del' into 'de' + 'el'. Each Stanza word is
mapped back to the whitespace token it came from by character offset.
"""
from __future__ import annotations
import bisect
import functools
from dataclasses import dataclass

from evaleng.analysis import pipeline


@dataclass(frozen=True)
class Word:
    idx: int            # position in the transcript-wide word list
    tok: int            # index into candidate.tokens
    text: str
    upos: str
    deprel: str
    head: int | None    # idx of the head word; None for a sentence root
    sent: int           # sentence number (Stanza's segmentation)


@functools.lru_cache(maxsize=256)
def parse(tokens: tuple[str, ...]) -> tuple[Word, ...]:
    """Parse ' '.join(tokens) once; every Stanza word knows its whitespace token."""
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1

    def tok_of(char: int) -> int:
        return bisect.bisect_right(starts, char) - 1

    out: list[Word] = []
    for si, sent in enumerate(pipeline()(" ".join(tokens)).sentences):
        base = len(out)
        for token in sent.tokens:                  # multi-word tokens (del, al) share start_char
            ti = tok_of(token.start_char)
            for w in token.words:
                out.append(Word(idx=base + w.id - 1, tok=ti, text=w.text, upos=w.upos,
                                deprel=w.deprel,
                                head=None if w.head == 0 else base + w.head - 1, sent=si))
    return tuple(out)
