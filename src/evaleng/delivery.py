"""Delivery metrics (feedback only, never scored).

Computed from word timings + audio duration. All thresholds are provisional.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from evaleng.interfaces import CandidateInput

PAUSE_SECONDS = 2.0                              # a gap longer than this is a "long pause"
FILLERS = frozenset({"eh", "em", "este", "esto", "mmm", "mm",
                     "uh", "um", "ah", "pues", "o sea"})   # provisional, region-dependent


class DeliveryMetrics(BaseModel):
    words: int
    audio_seconds: float | None
    speech_seconds: float                        # last word end - first word start
    words_per_minute: float | None               # gross rate over audio_seconds
    long_pauses: list[tuple[int, float]] = Field(default_factory=list)  # (token_index, gap_secs)
    hesitations: list[str] = Field(default_factory=list)
    false_starts: int = 0                         # crude: immediate token repetitions


def delivery_metrics(candidate: CandidateInput) -> DeliveryMetrics | None:
    """None when there are no timings (a typed-transcript run has no audio)."""
    t = candidate.timings
    if not t:
        return None

    speech = t[-1][1] - t[0][0]
    audio = candidate.audio_duration
    wpm = (len(candidate.tokens) / (audio / 60)) if audio else None

    pauses = [(i, t[i][0] - t[i - 1][1])
              for i in range(1, len(t)) if t[i][0] - t[i - 1][1] > PAUSE_SECONDS]

    low = [w.strip(".,¿?¡!").lower() for w in candidate.tokens]
    hesitations = [w for w in low if w in FILLERS]
    false_starts = sum(1 for i in range(1, len(low)) if low[i] and low[i] == low[i - 1])

    return DeliveryMetrics(
        words=len(candidate.tokens), audio_seconds=audio, speech_seconds=round(speech, 2),
        words_per_minute=round(wpm, 1) if wpm else None,
        long_pauses=[(i, round(g, 2)) for i, g in pauses],
        hesitations=hesitations, false_starts=false_starts,
    )


def delivery_lines(m: DeliveryMetrics) -> list[str]:
    """Render the metrics as a labeled, non-scored block for the report."""
    rate = f"{m.words_per_minute} wpm" if m.words_per_minute else "n/a (no audio duration)"
    audio = f" / {m.audio_seconds}s audio" if m.audio_seconds else ""
    pauses = ("  at tokens " + ", ".join(f"{i}({g}s)" for i, g in m.long_pauses)
              if m.long_pauses else "")
    fillers = ("  " + ", ".join(m.hesitations)) if m.hesitations else ""
    return [
        "",
        "Delivery (not scored):",
        f"  {m.words} words · {m.speech_seconds}s speech{audio} · rate {rate}",
        f"  long pauses (> {int(PAUSE_SECONDS)}s): {len(m.long_pauses)}{pauses}",
        f"  hesitations: {len(m.hesitations)}{fillers}",
        f"  false starts (approx): {m.false_starts}",
    ]