"""Freeze the W10 fuzzy tier on minimal pairs (D16). No Stanza: fuzzy_cover is pure string work."""
import pytest

from evaleng.match.lexical import fuzzy_cover

T = 0.85   # dispatch._LEXICAL_POLICY.fuzzy_threshold (per word)


@pytest.mark.parametrize("rendering, region, expected", [
    ("pistola cargada", "llevaba una pistola en su", "fail"),             # modifier dropped
    ("pistola cargada", "llevaba una pistola descargada en su", "fail"),  # antonym (0.82 < 0.85)
    ("asistir a sesiones semanales de consejería",
     "ordena asistir a sesiones de consejería y pagar", "fail"),          # long rendering, one word dropped
    ("pena máxima", "que la pena mínima es de", "fail"),
    ("de acuerdo con la víctima", "De acuerdo con el testigo, el", "fail"),
    ("pistola cargada", "llevaba una pistola cargda en su", "pass"),      # ASR typo
    ("conforme al acuerdo", "mil dólares. Conforme al acuerdo, el fiscal", "pass"),  # case + punct
    ("perdió los estribos", "y perdio los estribos y me", "pass"),        # accent dropped
])
def test_minimal_pairs(rendering, region, expected):
    score, _ = fuzzy_cover(region, rendering)
    assert ("pass" if score >= T else "fail") == expected


def test_extra_region_words_dont_hurt():
    assert fuzzy_cover("a b c pistola cargada x y z", "pistola cargada")[0] == 1.0


def test_determiners_are_ignored():
    assert fuzzy_cover("llevaba pistola cargada", "una pistola cargada")[0] == 1.0


def test_reason_material_names_the_weakest_pair():
    _, pairs = fuzzy_cover("llevaba una pistola en su", "pistola cargada")
    assert min(pairs, key=lambda p: p[2])[0] == "cargada"


def test_no_words_is_none():
    assert fuzzy_cover("...", "pistola cargada") is None
