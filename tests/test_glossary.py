"""Legal glossary loader: provenance rules, collect-then-fail, override precedence, concept mapping.

Fast: no Stanza. CSVs are written into tmp_path so these tests never depend on the real glossary,
except the last one, which guards the committed files.
"""
import pytest

from evaleng.ingest.glossary import (GlossaryError, concepts_for, en_key, load_glossary)

HEADER = "en_term,es_term,status,source,source_ref,note\n"
WA = "WA Courts Glossary of Legal Terms EN-ES (2008)"


def _csv(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


def test_en_key_drops_articles_and_possessives():
    assert en_key("you waive your right to a jury trial") == ("you", "waive", "right", "to", "jury", "trial")
    assert en_key("His Miranda rights") == ("miranda", "rights")


def test_clean_file_loads_grouped_by_concept(tmp_path):
    gen = _csv(tmp_path, "legal_wa_2008.csv", [
        f"Contempt of court,Desacato,acceptable,{WA},p.6,",
        f"Contempt of court,rebeldía,acceptable,{WA},p.6,",
        f"Waive,Renunciar a,acceptable,{WA},p.19,",
    ])
    groups = load_glossary([gen])
    assert {e.es_term for e in groups[("contempt", "of", "court")]} == {"Desacato", "rebeldía"}
    assert [e.es_term for e in groups[("waive",)]] == ["Renunciar a"]


def test_all_problems_reported_in_one_error(tmp_path):
    """Collect-then-fail: three bad rows -> one GlossaryError naming all three lines."""
    gen = _csv(tmp_path, "legal_wa_2008.csv", [
        f"Waive,Renunciar a,maybe,{WA},,",           # line 2: unknown status
        "Jury,Jurado,acceptable,,,",                  # line 3: approved row with no source
        f"Waive,Renunciar a,acceptable,{WA},,",      # line 4: fine
        f"Waive,Renunciar a,acceptable,{WA},,",      # line 5: duplicate of line 4
    ])
    with pytest.raises(GlossaryError) as exc:
        load_glossary([gen])
    msg = str(exc.value)
    assert "3 problem(s)" in msg
    assert "line 2" in msg and "status" in msg
    assert "line 3" in msg and "needs a source" in msg
    assert "line 5" in msg and "duplicate" in msg


def test_proposed_rows_may_omit_source(tmp_path):
    gen = _csv(tmp_path, "legal_wa_2008.csv", ["jury trial,juicio con jurado,proposed,,,linguist to decide"])
    entry = load_glossary([gen])[("jury", "trial")][0]
    assert entry.status == "proposed" and not entry.approved


def test_missing_generated_file_is_an_error_but_overrides_are_optional(tmp_path):
    with pytest.raises(GlossaryError, match="legal_wa_2008.csv: file not found"):
        load_glossary([tmp_path / "legal_wa_2008.csv"])
    gen = _csv(tmp_path, "legal_wa_2008.csv", [f"Waive,Renunciar a,acceptable,{WA},,"])
    assert load_glossary([gen, tmp_path / "legal_overrides.csv"])     # no error


def test_override_replaces_generated_row(tmp_path):
    """Same (en_term, es_term) pair: the later file (overrides) wins."""
    gen = _csv(tmp_path, "legal_wa_2008.csv", [
        f"Assault,Agresión,proposed,{WA},p.3,Spanish parenthetical",
        f"Assault,asalto,rejected,{WA},p.3,dictionary marks this as incorrect",
    ])
    ovr = _csv(tmp_path, "legal_overrides.csv", [f"Assault,Agresión,acceptable,{WA},p.3,approved"])
    entries = {e.es_term: e for e in load_glossary([gen, ovr])[("assault",)]}
    assert entries["Agresión"].status == "acceptable"
    assert entries["asalto"].status == "rejected"          # untouched generated row survives
    assert len(entries) == 2                                # replaced, not duplicated


def test_concepts_for_finds_every_concept_in_the_span(tmp_path):
    gen = _csv(tmp_path, "legal_wa_2008.csv", [
        f"Waive,Renunciar a,acceptable,{WA},,",
        f"Jury,Jurado,acceptable,{WA},,",
        f"Bail,fianza,acceptable,{WA},,",
    ])
    found = concepts_for("you waive your right to a jury trial", load_glossary([gen]))
    assert set(found) == {("waive",), ("jury",)}


def test_concepts_for_longest_match_wins(tmp_path):
    """'contempt of court' must not also demand a term for 'court'."""
    gen = _csv(tmp_path, "legal_wa_2008.csv", [
        f"Contempt of court,Desacato,acceptable,{WA},,",
        f"Court,Tribunal,acceptable,{WA},,",
    ])
    groups = load_glossary([gen])
    assert set(concepts_for("contempt of court", groups)) == {("contempt", "of", "court")}
    assert set(concepts_for("the court", groups)) == {("court",)}   # alone, it still matches


def test_committed_glossary_files_load():
    """Guards every future edit to data/lexicons/*.csv: they must load without errors."""
    groups = load_glossary()
    assert groups
