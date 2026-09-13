"""ASR wiring smoke test (slow, skipped without faster-whisper or a sample clip)."""
from pathlib import Path

import pytest

pytest.importorskip("faster_whisper")          # skip the whole module if the dep isn't installed
pytestmark = pytest.mark.slow

CLIP = Path("audio_samples/ref_es.mp3")


@pytest.mark.skipif(not CLIP.exists(),
                    reason="no sample clip")
def test_transcribes_and_captures_timings():
    from evaleng.asr import transcribe
    ci = transcribe(str(CLIP))
    assert "derecho" in ci.text.lower()                 # an anchor word it should hear clearly
    assert ci.tokens and len(ci.timings) == len(ci.tokens)   # one timing per token