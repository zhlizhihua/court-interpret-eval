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


def transcribe(audio_path: str, size: str = "small") -> CandidateInput:
    """Transcribe a Spanish audio file into a CandidateInput (1-best, with word timings)."""
    segments, _info = _model(size).transcribe(
        audio_path, language="es", word_timestamps=True,
    )
    tokens: list[str] = []
    timings: list[tuple[float, float]] = []
    for segment in segments:                             # `segments` is a generator — iterating runs ASR
        for word in (segment.words or []):
            tokens.append(word.word.strip())
            timings.append((word.start, word.end))

    return CandidateInput(text=" ".join(tokens), tokens=tokens, timings=timings)