#!/usr/bin/env python3
"""Extract the WA Courts English-Spanish legal glossary (2008) into evaleng glossary CSVs.

Usage:
    python scripts/extract_wa_glossary.py probe   PDF [PAGE]
    python scripts/extract_wa_glossary.py extract PDF RAW_CSV OUT_CSV
"""
import csv
import re
import sys
import unicodedata
from collections import Counter

import pdfplumber

SOURCE = "WA Courts Glossary of Legal Terms EN-ES (2008)"
ES_COL_X = 250.0     # Spanish column starts at x0 = 252.0 on every page (probe)
WRAP_X = 262.0       # wrapped Spanish lines are indented to x0 = 270; new-sense lines start at 252
LINE_TOL = 3.0       # words whose tops differ by less than this are on one printed line
SKIP = re.compile(r"^(glossary of legal terms|\(?english\s*[-–]\s*spanish\)?|march 2008|page \d+.*|\d+)$", re.I)
SECTION = re.compile(r"^[A-Z]$")
PAREN = re.compile(r"\(([^)]*)\)")
DOTS = re.compile(r"(?:\.\s*){2,}|…")
NOT_RE = re.compile(r"NOT\s*[‘'\"]([^’'\"]+)[’'\"]")   # the dictionary's own "( NOT ‘asalto’)" warnings
PREP = {"a", "al", "con", "sin", "de", "del", "en", "por", "ante", "para"}


def nfkc(s: str) -> str:
    """Unicode-normalize: folds PDF ligatures (ﬁ -> fi) and composes accents."""
    return unicodedata.normalize("NFKC", s)


def lines_of(page) -> list[list[dict]]:
    """The page's words grouped into printed lines (top-to-bottom), each line left-to-right."""
    words = sorted(page.extract_words(extra_attrs=["fontname"]), key=lambda w: w["top"])
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(w["top"] - lines[-1][0]["top"]) < LINE_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(ln, key=lambda w: w["x0"]) for ln in lines]


def probe(pdf_path: str, page_no: int = 2) -> None:
    with pdfplumber.open(pdf_path) as doc:
        page = doc.pages[page_no - 1]
        print(f"{len(doc.pages)} pages · width {page.width:.0f}pt · tables: {len(page.find_tables())}")
        for ln in lines_of(page)[:40]:
            print("  ".join(f"{w['x0']:5.0f}:{w['text']}[{w['fontname'].split('+')[-1][:14]}]"
                            for w in ln))


def read_entries(pdf_path: str) -> list[dict]:
    """Printed entries: {'page', 'en', 'es'}. Spanish-only lines continue the previous entry."""
    entries: list[dict] = []
    with pdfplumber.open(pdf_path) as doc:
        for pno, page in enumerate(doc.pages, start=1):
            if pno == 1:                          # title / credits page, no entries
                continue
            for ln in lines_of(page):
                whole = nfkc(" ".join(w["text"] for w in ln)).strip()
                if not whole or SKIP.match(whole) or SECTION.match(whole):
                    continue
                en = nfkc(" ".join(w["text"] for w in ln if w["x0"] < ES_COL_X)).strip()
                es = nfkc(" ".join(w["text"] for w in ln if w["x0"] >= ES_COL_X)).strip()
                if en:
                    entries.append({"page": pno, "en": en, "es": es})   # es may be "" (see report)
                elif entries:
                    first_x = min(w["x0"] for w in ln)
                    glue = " " if first_x >= WRAP_X else " / "   # indented = wrap; flush = new alternative
                    entries[-1]["es"] = (entries[-1]["es"] + glue + es).strip()
    return entries


def split_es(es_raw: str) -> list[str]:
    return [a.strip(" ,;") for a in DOTS.sub(" ", es_raw).split("/") if a.strip(" ,;")]


def to_rows(entry: dict) -> list[dict]:
    """One glossary row per Spanish alternative. Anything ambiguous -> status 'proposed' + reason."""
    en_raw, es_raw = entry["en"], entry["es"]
    glosses = [g.strip() for g in PAREN.findall(en_raw) if g.strip()]
    en = " ".join(DOTS.sub(" ", PAREN.sub(" ", en_raw)).split()).strip(" ,;")
    alts = split_es(es_raw)
    first_len = len(alts[0].split()) if alts else 0

    rows = []
    for i, alt in enumerate(alts or [""]):
        reasons = []
        if not alt:
            reasons.append("no Spanish captured")
        if re.search(r"\S/\S", es_raw):
            reasons.append("tight slash: possibly a partial alternative")
        if i > 0 and first_len > 1 and alt and (len(alt.split()) == 1 or alt.split()[0].lower() in PREP):
            reasons.append("elliptical alternative?")
        if DOTS.search(en_raw) or DOTS.search(es_raw):
            reasons.append("placeholder '...'")
        if not en:
            reasons.append("empty English")
        if PAREN.search(alt):
            reasons.append(f"Spanish parenthetical: {'; '.join(PAREN.findall(alt))}")
        if re.search(r",\s|\s[–-]\s", alt):
            reasons.append("comma/dash inside alternative")
        es_term = " ".join(PAREN.sub(" ", alt).split())
        rows.append({
            "en_term": en,
            "es_term": es_term,
            "status": "proposed" if reasons else "acceptable",
            "source": SOURCE,
            "source_ref": f"p.{entry['page']}: {en_raw} — {es_raw}",
            "note": "; ".join(glosses + reasons),
        })
    for bad in NOT_RE.findall(es_raw):        # the dictionary itself says "NOT 'x'" -> rejected row
        rows.append({"en_term": en, "es_term": bad.strip(), "status": "rejected",
                     "source": SOURCE, "source_ref": f"p.{entry['page']}: {en_raw} — {es_raw}",
                     "note": "dictionary marks this as incorrect"})
    return rows


def extract(pdf_path: str, raw_csv: str, out_csv: str) -> None:
    entries = read_entries(pdf_path)
    with open(raw_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "en", "es"])
        w.writeheader(); w.writerows(entries)

    rows, seen = [], set()
    for e in entries:
        for r in to_rows(e):
            key = (r["en_term"].lower(), r["es_term"].lower())
            if key in seen:                     # the PDF repeats some pairs; keep the first
                continue
            seen.add(key); rows.append(r)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["en_term", "es_term", "status", "source", "source_ref", "note"])
        w.writeheader(); w.writerows(rows)

    by_letter = Counter(e["en"][:1].upper() for e in entries)
    status = Counter(r["status"] for r in rows)
    print(f"{len(entries)} printed entries -> {len(rows)} rows  {dict(status)}")
    print("entries per letter:", " ".join(f"{k}{v}" for k, v in sorted(by_letter.items())))
    for e in entries:
        if not e["es"]:
            print(f"  CHECK p.{e['page']}: no Spanish for {e['en']!r} (wrapped English line?)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "probe":
        probe(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif len(sys.argv) == 5 and sys.argv[1] == "extract":
        extract(*sys.argv[2:])
    else:
        sys.exit(__doc__)