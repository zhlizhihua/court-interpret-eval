"""fastText soft-match backup for the lexical matcher (design §6.3; D1 puts it in bounds).

Off unless a model path is configured. When off, fastText is never imported, so the core
scorer runs on machines without the package or the model file.

Scope (W9 probe, cc.es.100): enabled per unit type via MatchingPolicy.embedding_backup, and
currently only for `position`. On single words, sister terms and antonyms scored ABOVE true
synonyms (abrazo/beso 0.841, máxima/mínima 0.833, botella/lata 0.811 vs multa/sanción 0.765),
so no threshold separates them for general vocabulary.
"""
from __future__ import annotations
import functools
import hashlib
import os
from pathlib import Path

import numpy as np

from evaleng.analysis import pipeline

ENV_VAR = "EVALENG_FASTTEXT"          # path to a fastText .bin; unset/empty = backup off
COS_THRESHOLD = 0.61                  # conservative until set from position minimal pairs (W9 6b);
                                      # a forgotten placeholder can only make the backup too strict
_DROP_UPOS = frozenset({"DET", "PUNCT"})   # keep ADP/SCONJ: for position units the connective
                                           # (según, conforme a, de acuerdo con) IS what is tested

_MODEL = None


def model_path() -> str | None:
    p = os.environ.get(ENV_VAR, "").strip()
    return p or None


def enabled() -> bool:
    return model_path() is not None


def model():
    """The fastText model, loaded on first use. Raises if enabled but the file is missing."""
    global _MODEL
    if _MODEL is None:
        path = model_path()
        if path is None:
            raise RuntimeError(f"fastText backup is off ({ENV_VAR} not set)")
        if not Path(path).is_file():
            raise FileNotFoundError(f"{ENV_VAR}={path!r} does not exist")
        import fasttext                       # imported here on purpose: off-path stays dependency-free
        _MODEL = fasttext.load_model(path)
    return _MODEL


@functools.lru_cache(maxsize=1)
def model_id() -> str:
    """Short, stable identity of the model file, recorded with every prediction run."""
    path = model_path()
    if path is None:
        return "off"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{Path(path).stem}:{h.hexdigest()[:8]}"


@functools.lru_cache(maxsize=4096)
def match_words(text: str) -> tuple[str, ...]:
    """Lowercased surface words, determiners and punctuation dropped.

    Surface forms, not lemmas: fastText was trained on surface text and its subword
    n-grams already handle inflection.
    """
    doc = pipeline()(text)
    return tuple(w.text.lower() for s in doc.sentences for w in s.words
                 if w.upos not in _DROP_UPOS)


@functools.lru_cache(maxsize=16384)
def _unit_vec(word: str) -> np.ndarray:
    v = model().get_word_vector(word)
    n = np.linalg.norm(v)
    return v / n if n else v


def soft_coverage(region: str, rendering: str) -> tuple[float, list[tuple[str, str, float]]] | None:
    """Weakest best-match cosine over the rendering's words, plus the (rendering, region, cos) pairs.

    Every rendering word must find a close counterpart somewhere in the region; extra region
    words (slack padding) don't hurt. None if either side has no words to compare.
    """
    r_words, c_words = match_words(rendering), match_words(region)
    if not r_words or not c_words:
        return None
    pairs = []
    for rw in r_words:
        best_w, best_c = max(((cw, float(_unit_vec(rw) @ _unit_vec(cw))) for cw in c_words),
                             key=lambda x: x[1])
        pairs.append((rw, best_w, round(best_c, 3)))
    return min(p[2] for p in pairs), pairs