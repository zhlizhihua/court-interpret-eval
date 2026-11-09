#!/usr/bin/env python3
"""List every unit whose verdict or region changed between two prediction CSVs.

    python scripts/flip_report.py OLD.csv NEW.csv OUT.csv [--types modifier,idiom]

Adds empty my_label (intended / unintended / unsure) and my_note columns for hand review.
Region-only changes are listed too: they are the main signal for localizer changes.
"""
import argparse
import collections
import csv


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["candidate_id"], r["unit_id"]): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("out")
    ap.add_argument("--types", default="", help="comma-separated unit types to keep")
    a = ap.parse_args()

    old, new = load(a.old), load(a.new)
    if set(old) != set(new):
        raise SystemExit(f"unit sets differ: {len(set(old) ^ set(new))} keys")
    keep = set(filter(None, a.types.split(",")))

    rows = []
    for key in sorted(old):
        o, n = old[key], new[key]
        if keep and o["unit_type"] not in keep:
            continue
        if o["engine"] != n["engine"] or o["region"] != n["region"]:
            rows.append({"candidate_id": key[0], "unit_id": key[1], "unit_type": o["unit_type"],
                         "old": o["engine"], "new": n["engine"],
                         "verdict_changed": "yes" if o["engine"] != n["engine"] else "",
                         "old_method": o["method"], "new_method": n["method"],
                         "old_region": o["region"], "new_region": n["region"],
                         "new_reason": n["reason"], "my_label": "", "my_note": ""})

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["candidate_id"])
        w.writeheader()
        w.writerows(rows)

    flips = [r for r in rows if r["verdict_changed"]]
    by = collections.Counter((r["unit_type"], f"{r['old']}->{r['new']}") for r in flips)
    print(f"{len(flips)} verdict changes, {len(rows) - len(flips)} region-only changes -> {a.out}")
    for (t, d), c in sorted(by.items()):
        print(f"  {t:20} {d:12} {c}")


if __name__ == "__main__":
    main()
