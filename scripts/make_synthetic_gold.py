#!/usr/bin/env python3
"""Write fake "returned" gold sheets to rehearse the validation harness (W6, Step 9).

For each blank Part 2 review sheet, write a copy whose human_verdict column holds the
engine's own frozen verdict, then flip a fixed list of units (pass <-> fail). Running
`evaluate` on the result must report exactly those flipped units as disagreements.

Usage:
    python scripts/make_synthetic_gold.py PREDICTIONS_CSV OUT_DIR [TEMPLATES_DIR]

Example:
    python scripts/make_synthetic_gold.py data/predictions/v0.1.csv runs/synthetic_gold
"""
import csv
import sys
from pathlib import Path

DEFAULT_TEMPLATES = "docs_local/linguist_review/part2_candidates_2"

# Units whose verdict gets flipped: (candidate_id, unit_id).
# Two are in the tune split (001-005) and two in the test split (006-007).
FLIPS = [
    ("st_en_es_001_c01", "u002"),
    ("st_en_es_003_c02", "u005"),
    ("st_en_es_006_c07", "u001"),
    ("st_en_es_007_c03", "u004"),
]
_OPPOSITE = {"pass": "fail", "fail": "pass"}


def load_predictions(path):
    """{(candidate_id, unit_id): engine verdict} from a predictions CSV."""
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["candidate_id"], r["unit_id"]): r["engine"] for r in csv.DictReader(f)}


def fill_sheet(rows, engine, flips, sheet_name):
    """Fill human_verdict in one sheet's rows (in place). Return the flips applied."""
    cid = next((r[1].strip() for r in rows if r and r[0].strip() == "candidate_id"), None)
    if not cid:
        sys.exit(f"{sheet_name}: no candidate_id row")
    hdr = next((i for i, r in enumerate(rows) if r and r[0].strip() == "unit_id"), None)
    if hdr is None:
        sys.exit(f"{sheet_name}: no unit_id header row")
    col = [c.strip() for c in rows[hdr]].index("human_verdict")

    applied = []
    for r in rows[hdr + 1:]:
        if not r or not r[0].strip():
            continue
        key = (cid, r[0].strip())
        if key not in engine:
            sys.exit(f"{sheet_name}: {key} has no engine prediction")
        verdict = engine[key]
        if verdict == "unscored":
            verdict = ""                      # nothing to copy; evaluate needs --allow-blank
        elif key in flips:
            applied.append((*key, verdict, _OPPOSITE[verdict]))
            verdict = _OPPOSITE[verdict]
        while len(r) <= col:                  # short rows: pad so the column exists
            r.append("")
        r[col] = verdict
    return applied


def main(pred_path, out_dir, templates_dir=DEFAULT_TEMPLATES):
    engine = load_predictions(pred_path)

    # Guard: a typo in FLIPS must not silently flip nothing.
    missing = [k for k in FLIPS if k not in engine]
    if missing:
        sys.exit(f"FLIPS not found in predictions: {missing}")
    unscored = [k for k in FLIPS if engine[k] == "unscored"]
    if unscored:
        sys.exit(f"FLIPS point at unscored units (nothing to flip): {unscored}")

    templates = sorted(Path(templates_dir).glob("*.csv"))
    if not templates:
        sys.exit(f"no template sheets in {templates_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    applied = []
    for t in templates:
        with open(t, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        applied += fill_sheet(rows, engine, set(FLIPS), t.name)
        with open(out / t.name, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)

    print(f"wrote {len(templates)} synthetic sheet(s) -> {out}")
    print(f"flipped {len(applied)} unit(s) — evaluate should report exactly these:")
    for cid, uid, was, now in applied:
        print(f"  {cid} {uid}: engine {was} -> gold {now}")
    if len(applied) != len(FLIPS):
        sys.exit(f"expected {len(FLIPS)} flips, applied {len(applied)} (a FLIPS sheet is missing?)")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])