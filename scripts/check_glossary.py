#!/usr/bin/env python3
"""Show how each legal_vocabulary unit maps onto glossary concepts, and whether each
concept has an approved Spanish term. Exits 1 if any legal unit maps to no concept.

Usage (from the repo root):
    python scripts/check_glossary.py [FIXTURES_DIR]
"""
import sys
from pathlib import Path

from evaleng.ingest.glossary import concepts_for
from evaleng.ingest.loader import load_fixture


def main(fixtures_dir="data/fixtures") -> int:
    no_concepts = 0
    for path in sorted(Path(fixtures_dir).glob("*.yaml")):
        fx = load_fixture(path)
        if fx.id.endswith("_draft"):
            continue
        for unit in fx.units:
            if unit.type.value != "legal_vocabulary":
                continue
            concepts = concepts_for(unit.source_span)
            if not concepts:
                verdict = "NO CONCEPTS"
                no_concepts += 1
            elif all(any(e.approved for e in es) for es in concepts.values()):
                verdict = "OK"
            else:
                verdict = "FIXTURE-ONLY"
            print(f"{fx.id} {unit.id}  {unit.source_span!r}  -> {verdict}")
            for key, entries in concepts.items():
                by = lambda s: [e.es_term for e in entries if e.status == s]
                approved = [e.es_term for e in entries if e.approved]
                print(f"    {' '.join(key):24} approved={approved} "
                      f"rejected={by('rejected')} proposed={by('proposed')}")
    if no_concepts:
        print(f"\n{no_concepts} legal unit(s) map to no glossary concept", file=sys.stderr)
    return 1 if no_concepts else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
