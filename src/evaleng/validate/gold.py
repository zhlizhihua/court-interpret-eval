"""Read the linguist's returned Part 2 sheets into validated gold rows."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Literal
from pydantic import BaseModel


class GoldError(Exception):
    """Raised once, listing every problem found across all sheets."""


# Keys are lowercased/stripped cell values.
VERDICT_MAP = {"pass": "pass", "fail": "fail"}


class GoldRow(BaseModel):
    candidate_id: str
    fixture_id: str
    unit_id: str
    unit_type: str
    sheet_span: str
    human: Literal["pass", "fail"]
    notes: str = ""


def _read_rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:     # -sig strips an Excel BOM
        return list(csv.reader(f))


def _cell(rows, label: str) -> str:
    """Second-column value of the first row labelled `label` (header block is read by label)."""
    for r in rows:
        if r and r[0].strip() == label:
            return r[1] if len(r) > 1 else ""
    return ""


def _grid(rows) -> list[dict[str, str]]:
    """The unit_id table as dicts keyed by its own header row."""
    hdr = next((i for i, r in enumerate(rows) if r and r[0].strip() == "unit_id"), None)
    if hdr is None:
        return []
    cols = [c.strip() for c in rows[hdr]]
    return [{cols[i]: (r[i].strip() if i < len(r) else "") for i in range(len(cols))}
            for r in rows[hdr + 1:] if r and r[0].strip()]


def load_gold(sheet_paths, fixtures: dict, transcripts: dict[str, str],
              allow_blank: bool = False) -> tuple[list[GoldRow], int]:
    """Return (gold rows, number of blank verdicts). Raise GoldError listing all problems."""
    rows, errors, blanks, seen = [], [], 0, set()
    for path in sorted(sheet_paths):
        name = Path(path).name
        raw = _read_rows(Path(path))

        cid = _cell(raw, "candidate_id").strip()
        if not cid:
            errors.append(f"{name}: no candidate_id row"); continue
        if cid in seen:
            errors.append(f"{name}: duplicate candidate {cid}"); continue
        seen.add(cid)

        fid = "_".join(cid.split("_")[:4])                         # st_en_es_00X
        fx = fixtures.get(fid)
        if fx is None:
            errors.append(f"{name}: no fixture {fid!r}"); continue

        sheet_tx = " ".join(_cell(raw, "Candidate Transcript").split())
        if sheet_tx != " ".join(transcripts.get(cid, "").split()):
            errors.append(f"{name}: transcript differs from data/candidates/{cid}.json")

        grid = _grid(raw)
        fx_types = {u.id: u.type.value for u in fx.units}
        sheet_ids = [g["unit_id"] for g in grid]
        if missing := set(fx_types) - set(sheet_ids):
            errors.append(f"{name}: units missing from sheet: {sorted(missing)}")
        if extra := set(sheet_ids) - set(fx_types):
            errors.append(f"{name}: units not in fixture: {sorted(extra)}")

        for g in grid:
            uid = g["unit_id"]
            if uid not in fx_types:
                continue
            if g.get("type") != fx_types[uid]:
                errors.append(f"{name} {uid}: sheet type {g.get('type')!r} != fixture {fx_types[uid]!r}")
            raw_v = g.get("human_verdict", "").strip().lower()
            if not raw_v:
                blanks += 1
                if not allow_blank:
                    errors.append(f"{name} {uid}: blank human_verdict")
                continue
            if raw_v not in VERDICT_MAP:
                errors.append(f"{name} {uid}: unknown verdict {g['human_verdict']!r}")
                continue
            rows.append(GoldRow(candidate_id=cid, fixture_id=fid, unit_id=uid,
                                unit_type=fx_types[uid], sheet_span=g.get("candidate_span", ""),
                                human=VERDICT_MAP[raw_v], notes=g.get("notes", "")))

    if errors:
        raise GoldError(f"{len(errors)} problem(s) in gold sheets:\n  " + "\n  ".join(errors))
    return rows, blanks