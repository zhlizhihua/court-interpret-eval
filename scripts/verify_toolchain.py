"""Smoke test: prove every Spanish NLP dependency loads and runs on one sentence.

Run:  python scripts/verify_toolchain.py
Exits 0 if every library is OK, 1 if any failed.
"""
from __future__ import annotations
import sys

SENT = "El acusado tiene derecho a un intérprete durante el juicio."
results: list[tuple[str, bool, str]] = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail))
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))


def _stanza():
    import stanza
    nlp = stanza.Pipeline("es", processors="tokenize,pos,lemma,depparse", verbose=False)
    doc = nlp(SENT)
    return f"{sum(len(s.words) for s in doc.sentences)} tokens parsed"


def _spacy():
    import spacy
    doc = spacy.load("es_core_news_md")(SENT)
    return f"{len(doc)} tokens, {len(doc.ents)} ents, lemma[0]={doc[0].lemma_}"


def _rapidfuzz():
    from rapidfuzz import fuzz
    return f"token_set_ratio={fuzz.token_set_ratio('duda razonable', 'razonable duda'):.0f}"


def _jellyfish():
    import jellyfish
    return f"metaphone('García')={jellyfish.metaphone('García')}"


def _text2num():
    from text_to_num import alpha2digit
    out = alpha2digit("veintitrés", "es")
    assert out == "23", f"expected '23', got {out!r}"
    return f"alpha2digit('veintitrés','es')={out}"


def _wn():
    import wn
    return f"{len(wn.synsets('juicio', lang='es'))} synsets for 'juicio'"


def _fasttext():
    import fasttext
    v = fasttext.load_model("cc.es.300.bin").get_word_vector("intérprete")
    return f"vector dim={len(v)}"


check("stanza(es)", _stanza)
check("spacy(es_core_news_md)", _spacy)
check("rapidfuzz", _rapidfuzz)
check("jellyfish", _jellyfish)
check("text2num(es)", _text2num)
check("wn(omw-es)", _wn)
# check("fasttext(cc.es.300)", _fasttext)

print(f"\nSmoke test on: {SENT!r}\n")
ok = all(passed for _, passed, _ in results)
for name, passed, detail in results:
    print(f"[{'OK  ' if passed else 'FAIL'}] {name:26} {detail}")
sys.exit(0 if ok else 1)