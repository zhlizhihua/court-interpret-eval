from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score

def test_nbest_rescues_a_number_the_1best_botched(exam_fixture):
    ref = " ".join(exam_fixture.reference_rendering.split())
    best = ref.replace("dos cargos", "los cargos")     # 1-best drops the count
    cand = CandidateInput(text=best, tokens=best.split(), nbest=[ref])
    verdicts = {v.unit_id: v for v in score(cand, exam_fixture).result.verdicts}
    assert verdicts["u006"].status == "pass"
    assert "nbest" in verdicts["u006"].method

def test_nbest_never_breaks_a_1best_pass(exam_fixture):
    ref = " ".join(exam_fixture.reference_rendering.split())
    cand = CandidateInput(text=ref, tokens=ref.split(), nbest=["texto irrelevante"])
    assert score(cand, exam_fixture).result.coverage == (10, 10)

def test_nbest_is_noop_when_absent(exam_fixture):
    ref = " ".join(exam_fixture.reference_rendering.split())
    cand = CandidateInput(text=ref, tokens=ref.split())   # nbest=None
    assert score(cand, exam_fixture).result.overall_pass is True