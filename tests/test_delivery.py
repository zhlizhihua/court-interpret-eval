from evaleng.interfaces import CandidateInput
from evaleng.delivery import delivery_metrics

def _ci(tokens, timings, audio=None):
    return CandidateInput(text=" ".join(tokens), tokens=tokens, timings=timings, audio_duration=audio)

def test_wpm_uses_audio_duration():
    m = delivery_metrics(_ci(["a", "b", "c", "d"],
                             [(0,.5),(.5,1),(1,1.5),(1.5,2)], audio=120.0))  # 4 words / 2 min
    assert m.words_per_minute == 2.0

def test_detects_a_long_pause():
    m = delivery_metrics(_ci(["uno", "dos"], [(0, 1), (4.0, 5.0)], audio=5.0))  # 3s gap > 2s
    assert len(m.long_pauses) == 1 and m.long_pauses[0][0] == 1

def test_flags_fillers_and_repeats():
    m = delivery_metrics(_ci(["el", "el", "eh", "acusado"],
                             [(0,.3),(.3,.6),(.6,.9),(.9,1.4)], audio=1.4))
    assert "eh" in m.hesitations and m.false_starts == 1

def test_none_without_timings():
    assert delivery_metrics(CandidateInput(text="x", tokens=["x"])) is None