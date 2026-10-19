"""Legal-terminology glossary (EN concept -> Spanish equivalents), with provenance.

Reads the generated WA glossary plus a hand-curated overrides file. For the same
(en_term, es_term) pair the later file wins, so the generated file can be rebuilt
at any time without losing human decisions.
"""
from __future__ import annotations
import csv
import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

_LEX = Path(__file__).resolve().parents[3] / "data" / "lexicons"   # repo/data/lexicons
DEFAULT_PATHS = (_LEX / "legal_wa_2008.csv", _LEX / "legal_overrides.csv")
OPTIONAL = {"legal_overrides.csv"}                                   # may not exist yet

Status = Literal["preferred", "acceptable", "rejected", "proposed"]
_EN_STOP = frozenset({"a", "an", "the", "his", "her", "their", "your", "my", "our", "its"})


class GlossaryError(Exception):
    """Raised once, listing every problem found across all glossary files."""


class GlossaryEntry(BaseModel):
    en_term: str
    es_term: str
    status: Status
    source: str = ""
    source_ref: str = ""
    note: str = ""

    @property
    def approved(self) -> bool:
        return self.status in ("preferred", "acceptable")


def en_key(text: str) -> tuple[str, ...]:
    """English as lowercase tokens, articles/possessives dropped: 'waive your right' -> ('waive', 'right')."""
    toks = (t.strip(".,;:'\"()").lower() for t in text.split())
    return tuple(t for t in toks if t and t not in _EN_STOP)


def _read_one(path: Path) -> tuple[list[GlossaryEntry], list[str]]:
    """Validated entries from one CSV, plus every problem found in it."""
    entries: list[GlossaryEntry] = []
    errors: list[str] = []
    if not path.exists():
        if path.name not in OPTIONAL:
            errors.append(f"{path.name}: file not found")
        return entries, errors

    seen: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):          # line 1 is the header
            row = {k: (v or "").strip() for k, v in row.items() if k}
            where = f"{path.name} line {i}"
            try:
                e = GlossaryEntry(**row)
            except ValidationError as exc:
                errors.append(f"{where}: {exc.errors()[0]['loc'][0]}: {exc.errors()[0]['msg']}")
                continue
            pair = (e.en_term.lower(), e.es_term.lower())
            if not e.en_term or not e.es_term:
                errors.append(f"{where}: en_term and es_term are required")
            elif e.status != "proposed" and not e.source:
                errors.append(f"{where}: status {e.status!r} needs a source")
            elif not en_key(e.en_term):
                errors.append(f"{where}: en_term {e.en_term!r} is empty after dropping articles")
            elif pair in seen:
                errors.append(f"{where}: duplicate {e.en_term!r} -> {e.es_term!r}")
            else:
                seen.add(pair)
                entries.append(e)
    return entries, errors


def load_glossary(paths=DEFAULT_PATHS) -> dict[tuple[str, ...], list[GlossaryEntry]]:
    """Entries grouped by English concept key; later files override earlier ones.

    Raises GlossaryError listing all problems in all files.
    """
    merged: dict[tuple[str, str], GlossaryEntry] = {}
    errors: list[str] = []
    for path in paths:
        entries, file_errors = _read_one(Path(path))
        errors += file_errors
        for e in entries:
            merged[(e.en_term.lower(), e.es_term.lower())] = e        # later file wins
    if errors:
        raise GlossaryError(f"{len(errors)} problem(s) in glossary files:\n  " + "\n  ".join(errors))

    groups: dict[tuple[str, ...], list[GlossaryEntry]] = {}
    for e in merged.values():
        groups.setdefault(en_key(e.en_term), []).append(e)
    return groups


@functools.lru_cache(maxsize=1)
def glossary() -> dict[tuple[str, ...], list[GlossaryEntry]]:
    """The default glossary, loaded once per process."""
    return load_glossary()


def concepts_for(source_span: str, groups=None) -> dict[tuple[str, ...], list[GlossaryEntry]]:
    """Glossary concepts found in the unit's English source_span, longest match winning.

    A concept whose key sits inside a longer matched key is dropped, so 'contempt of court'
    does not also demand a term for 'court'.
    """
    from evaleng.localize import _find_sublist
    groups = glossary() if groups is None else groups
    span = list(en_key(source_span))
    hits = {}
    for k in groups:
        pos = _find_sublist(span, list(k))
        if pos is not None:
            hits[k] = pos
    keep = {k for k, (s, e) in hits.items()
            if not any(k2 != k and s2 <= s and e <= e2 for k2, (s2, e2) in hits.items())}
    return {k: groups[k] for k in keep}