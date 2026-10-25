"""fastText soft-match backup (W9).

Fast tests use a tiny fake model and a whitespace tokenizer, so they need neither the
2 GB model nor Stanza. Model-gated tests (marked slow, skipped unless EVALENG_FASTTEXT
points at a model) run the real thing on position minimal pairs taken from v0.1.

    pytest tests/test_embeddings.py -m "not slow"            # fast only
    EVALENG_FASTTEXT=~/models/fasttext/cc.es.100.bin \\
        pytest tests/test_embeddings.py -m slow -s          # real model; -s prints the scores
"""
import os

import numpy as np
import pytest

from evaleng import embeddings


# ---------------------------------------------------------------- fast: fake model

class FakeModel:
    """Words -> fixed 2-d vectors, so cosines are easy to reason about."""
    VECS = {"multa": (1.0, 0.0), "sanción": (0.9, 0.1), "pecuniaria": (0.0, 1.0),
            "estatal": (0.2, 0.98), "abrazo": (0.0, 1.0)}

    def get_word_vector(self, w):
        return np.array(self.VECS.get(w, (-1.0, 0.0)), dtype=np.float32)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(embeddings, "_MODEL", FakeModel())
    monkeypatch.setattr(embeddings, "match_words", lambda text: tuple(text.lower().split()))
    embeddings._unit_vec.cache_clear()          # vectors are cached; don't leak between tests
    yield
    embeddings._unit_vec.cache_clear()


def test_off_when_env_unset(monkeypatch):
    monkeypatch.delenv(embeddings.ENV_VAR, raising=False)
    assert not embeddings.enabled()
    with pytest.raises(RuntimeError):
        embeddings.model()


def test_missing_model_file_is_loud(monkeypatch):
    monkeypatch.setenv(embeddings.ENV_VAR, "/does/not/exist.bin")
    monkeypatch.setattr(embeddings, "_MODEL", None)
    assert embeddings.enabled()
    with pytest.raises(FileNotFoundError):
        embeddings.model()


def test_every_rendering_word_needs_a_counterpart(fake):
    cos, pairs = embeddings.soft_coverage("estatal y una sanción pecuniaria", "multa")
    assert pairs[0][:2] == ("multa", "sanción")       # best counterpart found
    assert cos == pytest.approx(0.9 / np.hypot(0.9, 0.1), abs=1e-3)


def test_extra_region_words_dont_hurt(fake):
    short, _ = embeddings.soft_coverage("sanción", "multa")
    padded, _ = embeddings.soft_coverage("estatal sanción pecuniaria", "multa")
    assert padded == short


def test_weakest_word_decides(fake):
    """min, not mean: one strong word can't carry a missing one."""
    cos, pairs = embeddings.soft_coverage("sanción", "multa abrazo")
    assert cos == min(p[2] for p in pairs) and cos < 0.5


def test_nothing_to_compare_returns_none(fake):
    assert embeddings.soft_coverage("", "multa") is None
    assert embeddings.soft_coverage("sanción", "") is None


# ---------------------------------------------------------------- slow: real model + Stanza

needs_model = pytest.mark.skipif(not os.environ.get(embeddings.ENV_VAR),
                                 reason="no fastText model configured (EVALENG_FASTTEXT)")

# Regions are real v0.1 regions for position units; renderings are the fixture's.
POSITION_SHOULD_PASS = [
    ("drogas aleatorias. Adicionalmente, se le ordena acudir a", "además, se le ordena"),  # 004 c03
    ("drogas aleatorias. Igualmente, se le ordena acudir una", "además, se le ordena"),    # 004 c05
    ("lesiones agravadas. Conforme al relato de la víctima, García lo", "según la víctima"),  # 001 c03
]
POSITION_SHOULD_FAIL = [
    ("con agravantes. La víctima contó que", "según la víctima"),        # 001 c06: connective dropped
    ("te estreses. A partir de mañana, voy a", "a partir de ahora"),     # 006 c07: 'mañana' ≠ 'ahora'
    ("te estreses. Como siempre, voy a", "a partir de ahora"),           # 006 c08
]


@pytest.mark.slow
@needs_model
@pytest.mark.parametrize("region, rendering", POSITION_SHOULD_FAIL)
def test_position_traps_stay_below_threshold(region, rendering):
    """Safety check: known wrong position renderings must never soft-pass."""
    res = embeddings.soft_coverage(region, rendering)
    assert res is None or res[0] < embeddings.COS_THRESHOLD, res


@pytest.mark.slow
@needs_model
def test_print_position_pair_scores():
    """Not an assertion: prints every pair's score so you can choose COS_THRESHOLD (run with -s)."""
    print(f"\nmodel {embeddings.model_id()}  ·  COS_THRESHOLD {embeddings.COS_THRESHOLD}")
    for label, pairs in (("should pass", POSITION_SHOULD_PASS), ("should fail", POSITION_SHOULD_FAIL)):
        for region, rendering in pairs:
            res = embeddings.soft_coverage(region, rendering)
            cos, words = (None, []) if res is None else res
            shown = ", ".join(f"{r}≈{c} {s:.2f}" for r, c, s in words)
            print(f"  {label:11} {cos if cos is None else f'{cos:.3f}':>6}  {rendering!r:26} <- {region!r}  [{shown}]")
