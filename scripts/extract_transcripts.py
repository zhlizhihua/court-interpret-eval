#!/usr/bin/env python3
"""Extract candidate transcripts from Part 2 linguist-review CSV sheets into
one JSON file per transcript.

Each Part 2 sheet is laid out as a header block:

    candidate_id,<id>
    Source,"<english source text>"
    Candidate Transcript,"<spanish transcript>"
    <blank line>
    unit_id,type,source_span,candidate_span,human_verdict,notes
    u001,...
    ...

We read the header block by ROW LABEL (not position), so the script is robust
to Excel re-saves, a UTF-8 BOM, or reordered header rows.

Usage:
    python extract_transcripts.py [INPUT_DIR] [OUTPUT_DIR]

With no arguments it uses the two defaults below.
"""
import csv, glob, json, os, sys

INPUT_DIR  = "/Users/zhihua/git/court-interpret-eval/docs_local/linguist_review/part2_candidates_2"
OUTPUT_DIR = "/Users/zhihua/git/court-interpret-eval/candidates"

# Flip to True to also embed the per-unit review grid (with any human verdicts/notes present).
INCLUDE_REVIEW_GRID = False


def _cell(rows, label):
    """Second-column value of the first row whose first cell == label."""
    for r in rows:
        if r and r[0].strip() == label:
            return r[1] if len(r) > 1 else ""
    return ""


def _units(rows):
    """Rows of the review grid as dicts keyed by the grid's own header."""
    hdr = next((i for i, r in enumerate(rows) if r and r[0].strip() == "unit_id"), None)
    if hdr is None:
        return []
    cols = [c.strip() for c in rows[hdr]]
    out = []
    for r in rows[hdr + 1:]:
        if not r or not r[0].strip().startswith("u"):
            continue
        out.append({cols[i]: (r[i] if i < len(r) else "") for i in range(len(cols))})
    return out


def main(indir, outdir):
    files = sorted(glob.glob(os.path.join(indir, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {indir}")
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        cid = _cell(rows, "candidate_id")
        if not cid:
            print(f"  skip (no candidate_id row): {os.path.basename(path)}")
            continue
        rec = {
            "candidate_id": cid,
            "fixture_id": "_".join(cid.split("_")[:4]),   # st_en_es_00X
            "source_text": _cell(rows, "Source"),
            "transcript": _cell(rows, "Candidate Transcript"),
        }
        if INCLUDE_REVIEW_GRID:
            rec["units"] = _units(rows)
        with open(os.path.join(outdir, cid + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        n += 1
    print(f"wrote {n} JSON file(s) -> {outdir}")


if __name__ == "__main__":
    indir  = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    outdir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(indir, outdir)
