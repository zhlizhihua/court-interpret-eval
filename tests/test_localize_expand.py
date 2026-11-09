"""W11 (D17): phrase growth and sentence clipping, on hand-built parses (no Stanza).

Each parse is written in Universal Dependencies form: (text, upos, deprel, head) with head as a
1-based word number in its sentence, 0 for the root. One word per whitespace token here, so the
token index equals the word index. Labels follow what the installed Stanza model actually emits
(W11 probe): 'me' in 'me dio' is obl:arg; 'de acuerdo con' is a fixed expression headed by 'De'.
"""
import pytest

import evaleng.localize as L
from evaleng.schema.models import UnitType as T
from evaleng.syntax import Word


def parse(*sentences):
    words, tok = [], 0
    for si, sent in enumerate(sentences):
        base = len(words)
        for i, (text, upos, deprel, head) in enumerate(sent):
            words.append(Word(idx=base + i, tok=tok, text=text, upos=upos, deprel=deprel,
                              head=None if head == 0 else base + head - 1, sent=si))
            tok += 1
    return tuple(words)


def span_text(words, span):
    return " ".join(w.text for w in words if span[0] <= w.tok < span[1])


GUN = parse([("golpeó", "VERB", "root", 0), ("con", "ADP", "case", 4), ("una", "DET", "det", 4),
             ("pistola", "NOUN", "obl", 1), ("cargada", "ADJ", "amod", 4),
             ("en", "ADP", "case", 7), ("mano", "NOUN", "obl", 1)])

YEARS = parse([("condenado", "VERB", "root", 0), ("a", "ADP", "case", 7),
               ("no", "ADV", "advmod", 4), ("más", "ADV", "advmod", 6),
               ("de", "ADP", "fixed", 4), ("cinco", "NUM", "nummod", 7),
               ("años", "NOUN", "obl", 1)])

GONE = parse([("cuando", "SCONJ", "mark", 2), ("llegó", "VERB", "advcl", 8),
              ("el", "DET", "det", 4), ("imbécil", "NOUN", "nsubj", 8),
              ("ya", "ADV", "advmod", 8), ("se", "PRON", "expl:pv", 8),
              ("había", "AUX", "aux", 8), ("ido", "VERB", "root", 0)])

SENTENCED = parse([("se", "PRON", "expl:pass", 2), ("dictará", "VERB", "root", 0),
                   ("sentencia", "NOUN", "nsubj", 2), ("contra", "ADP", "case", 6),
                   ("el", "DET", "det", 6), ("acusado", "NOUN", "obl", 2)])

SEGUN = parse([("según", "SCONJ", "mark", 2), ("declaró", "VERB", "advcl", 6),
               ("la", "DET", "det", 4), ("víctima", "NOUN", "nsubj", 2),
               ("él", "PRON", "nsubj", 6), ("huyó", "VERB", "root", 0)])

ACUERDO = parse([("De", "ADP", "mark", 7), ("acuerdo", "NOUN", "fixed", 1),
                 ("con", "ADP", "fixed", 1), ("lo", "PRON", "det", 5), ("que", "PRON", "obj", 6),
                 ("declaró", "VERB", "root", 0), ("dijo", "VERB", "root", 0)])

CONFORME = parse([("conforme", "ADV", "case", 4), ("a", "ADP", "fixed", 1),
                  ("el", "DET", "det", 4), ("relato", "NOUN", "obl", 6),
                  ("él", "PRON", "nsubj", 6), ("huyó", "VERB", "root", 0)])

SHAME = parse([("Me", "PRON", "obl:arg", 2), ("dio", "VERB", "root", 0),
               ("tanta", "DET", "det", 4), ("vergüenza", "NOUN", "nsubj", 2)])

CRAZY = parse([("dijo", "VERB", "root", 0), ("que", "SCONJ", "mark", 7),
               ("el", "DET", "det", 4), ("sospechoso", "NOUN", "nsubj", 7),
               ("se", "PRON", "expl:pv", 7), ("había", "AUX", "aux", 7),
               ("puesto", "VERB", "ccomp", 1), ("loco", "ADJ", "xcomp", 7)])

HOMIES = parse([("Mis", "DET", "det", 2), ("carnales", "NOUN", "nsubj", 5),
                ("de", "ADP", "case", 4), ("afuera", "ADV", "advmod", 2),
                ("esperan", "VERB", "root", 0)])


@pytest.mark.parametrize("words, core, unit_type, expected", [
    (GUN, (3, 5), T.MODIFIER, "una pistola cargada"),            # article joins; 'con' stays out
    (YEARS, (5, 7), T.NUMBER, "no más de cinco años"),            # quantifier chain, not 'a'
    (YEARS, (5, 6), T.NUMBER, "no más de cinco años"),            # core on the numeral: climbs to años
    (GONE, (4, 8), T.GRAMMAR, "el imbécil ya se había ido"),      # verb pulls in its subject NP
    (SENTENCED, (4, 6), T.GRAMMAR, "se dictará sentencia contra el acusado"),  # climbs to the verb
    (SEGUN, (1, 4), T.POSITION, "según declaró la víctima"),     # verb pulls in its subordinator
    (ACUERDO, (1, 6), T.POSITION, "De acuerdo con lo que declaró"),  # 'acuerdo' climbs to fixed head 'De'
    (CONFORME, (2, 4), T.POSITION, "conforme a el relato"),      # position keeps its preposition
    (SHAME, (1, 4), T.FALSE_COGNATE, "Me dio tanta vergüenza"),   # clitic obl:arg joins the verb
    (CRAZY, (4, 8), T.IDIOM, "se había puesto loco"),             # idiom: no subject, no 'que'
    (HOMIES, (0, 2), T.SLANG, "Mis carnales"),                    # right-side advmod stays out
])
def test_grow_reaches_the_phrase(words, core, unit_type, expected):
    assert span_text(words, L._grow(core, unit_type, words)) == expected


def test_grow_does_not_climb_for_vocabulary():
    # 'pistola' hangs off the verb, but a modifier unit never climbs to it.
    assert span_text(GUN, L._grow((3, 5), T.MODIFIER, GUN)) == "una pistola cargada"


def test_grammar_unit_takes_the_subject():
    assert span_text(CRAZY, L._grow((4, 8), T.GRAMMAR, CRAZY)) == "que el sospechoso se había puesto loco"


def test_grammar_core_with_a_verb_does_not_climb():
    span = L._grow((4, 8), T.GRAMMAR, GONE)
    assert "llegó" not in span_text(GONE, span)


def test_grow_is_capped(monkeypatch):
    monkeypatch.setattr(L, "_MAX_GROW", 1)
    assert span_text(YEARS, L._grow((5, 7), T.NUMBER, YEARS)) == "de cinco años"


def test_grow_never_shrinks_the_core():
    assert L._grow((0, 7), T.MODIFIER, GUN) == (0, 7)


def test_clip_stops_at_the_sentence_end():
    words = parse([("agresión", "NOUN", "root", 0), ("agravada.", "ADJ", "amod", 1)],
                  [("De", "ADP", "case", 2), ("acuerdo", "NOUN", "root", 0)])
    assert L._clip_to_sentence((0, 4), [0, 1, 2], words) == (0, 2)


def test_expand_off_needs_no_parse(exam_fixture, monkeypatch):
    """The default path never imports the parser, so it reproduces W10 spans exactly."""
    monkeypatch.setattr(L, "_EXPAND", False)
    import sys
    sys.modules.pop("evaleng.syntax", None)
    from evaleng.interfaces import CandidateInput
    text = " ".join(exam_fixture.reference_rendering.split())
    L.localize(CandidateInput(text=text, tokens=text.split()), exam_fixture)
    assert "evaleng.syntax" not in sys.modules


@pytest.mark.slow
def test_real_parse_maps_words_to_whitespace_tokens():
    from evaleng.syntax import parse as real_parse
    toks = ("Vendió", "del", "coche", "dos", "mil", "dólares.")
    words = real_parse(toks)
    assert {w.tok for w in words if w.text.lower() in ("de", "el")} == {1}   # 'del' -> de + el
    assert {w.tok for w in words if w.text == "."} == {5}                    # split punctuation
    assert all(0 <= w.tok < len(toks) for w in words)
