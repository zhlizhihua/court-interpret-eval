"""ASR (upstream, non-scoring): a Whisper transcript as a CandidateInput.
"""
from __future__ import annotations

from evaleng.interfaces import CandidateInput

_MODEL = None


def _model(size: str = "small"):
    """Load the Whisper model once, reuse it (like analysis.pipeline())."""
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel          # imported lazily so the dep is optional
        _MODEL = WhisperModel(size, device="cpu", compute_type="int8")
    return _MODEL


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def transcribe(audio_path: str, size: str = "small",
               n_alternatives: int = 0) -> CandidateInput:
    """Transcribe Spanish audio into a CandidateInput.

    1-best + word timings always;
    n_alternatives extra sampled passes go into nbest.
    """
    model = _model(size)
    segments, info = model.transcribe(audio_path, language="es", word_timestamps=True)

    tokens: list[str] = []
    timings: list[tuple[float, float]] = []
    for segment in segments:
        for word in (segment.words or []):
            tokens.append(word.word.strip())
            timings.append((word.start, word.end))
    best = " ".join(tokens)

    nbest: list[str] | None = None
    if n_alternatives > 0:
        seen = {_norm(best)}
        alts: list[str] = []
        for _ in range(n_alternatives):
            segs, _ = model.transcribe(audio_path, language="es", temperature=0.6)
            alt = " ".join(w.word.strip() for s in segs for w in (s.words or []))
            if alt and _norm(alt) not in seen:      # dedup against 1-best and each other
                seen.add(_norm(alt))
                alts.append(alt)
        nbest = alts or None

    return CandidateInput(text=best, tokens=tokens, timings=timings,
                          nbest=nbest, audio_duration=info.duration)