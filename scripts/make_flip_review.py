#!/usr/bin/env python3
"""List every position unit the fastText backup could touch, for hand labelling.

Usage: python scripts/make_flip_review.py OFF_CSV ON_CSV OUT_CSV
  e.g. python scripts/make_flip_review.py runs/w9/off.csv runs/w9/on.csv runs/w9/flip_review.csv
"""
import csv
import re
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["candidate_id"], r["unit_id"]): r for r in csv.DictReader(f)}


def main(off_csv, on_csv, out_csv):
    off, on = load(off_csv), load(on_csv)
    rows = []
    for key, o in off.items():
        # only units the backup could touch: position units that failed at fuzzy with it off
        if o["unit_type"] != "position" or o["method"] != "fuzzy" or o["engine"] != "fail":
            continue
        n = on[key]
        flipped = n["engine"] != o["engine"]
        m = re.search(r"to '([^']*)'", n["reason"] if flipped else o["reason"])
        rows.append({
            "candidate_id": key[0], "unit_id": key[1],
            "region": o["region"],
            "closest_rendering": m.group(1) if m else "",
            "fuzzy": o["score"],
            "fasttext": n["score"] if flipped else "",
            "on_verdict": n["engine"],
            "flipped": "yes" if flipped else "",
            "my_label": "",          # fill in: right / wrong / unsure
            "my_note": "",
            "ft_reason": n["reason"] if flipped else "",
        })
    rows.sort(key=lambda r: (r["flipped"] != "yes", r["candidate_id"], r["unit_id"]))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} position units, {sum(r['flipped'] == 'yes' for r in rows)} flipped -> {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
