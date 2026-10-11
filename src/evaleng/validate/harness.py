"""Validation harness. predict: engine -> frozen CSV. evaluate: CSV + gold -> reports."""
from __future__ import annotations
import csv, json, subprocess
from pathlib import Path

from evaleng.ingest.loader import load_fixture
from evaleng.interfaces import CandidateInput, UnitVerdict
from evaleng.localize import _tokenize, _find_sublist
from evaleng.pipeline import score_with_alignment, aggregate
from evaleng.schema.models import UnitType
from evaleng.validate.metrics import Pair, confusion, bootstrap_ci, METRICS

TEST_FIXTURES = {"st_en_es_006", "st_en_es_007"}      # D11: held out from W7 tuning


def split_of(fixture_id: str) -> str:
    return "test" if fixture_id in TEST_FIXTURES else "tune"


def load_fixtures(fixtures_dir) -> dict:
    """Exam fixtures by id; the rough *_draft exemplar is not part of the gold set."""
    fxs = [load_fixture(p) for p in sorted(Path(fixtures_dir).glob("*.yaml"))]
    return {fx.id: fx for fx in fxs if not fx.id.endswith("_draft")}


def load_transcripts(candidates_dir) -> dict[str, str]:
    out = {}
    for p in sorted(Path(candidates_dir).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[rec["candidate_id"]] = rec["transcript"]
    return out


def engine_version() -> str:
    """Short git sha, '-dirty' if tracked files have uncommitted changes."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def _write_csv(rows: list[dict], path, columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)



PRED_COLUMNS = ["engine_version", "candidate_id", "fixture_id", "unit_id", "unit_type",
                "engine", "method", "score", "reason", "region", "loc_confidence",
                "candidate_overall_pass"]


def predict(candidates_dir, fixtures_dir) -> list[dict]:
    fixtures = load_fixtures(fixtures_dir)
    version = engine_version()
    rows = []
    for p in sorted(Path(candidates_dir).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        fx = fixtures[rec["fixture_id"]]
        text = rec["transcript"].strip()                     # exactly what cli.py does
        cand = CandidateInput(text=text, tokens=text.split())
        report, alignment = score_with_alignment(cand, fx)
        for v in report.result.verdicts:
            loc = alignment.locations[v.unit_id]
            s = loc.token_span
            rows.append({
                "engine_version": version,
                "candidate_id": rec["candidate_id"], "fixture_id": fx.id,
                "unit_id": v.unit_id, "unit_type": v.unit_type.value,
                "engine": v.status, "method": v.method,
                "score": "" if v.score is None else v.score, "reason": v.reason,
                "region": " ".join(cand.tokens[s[0]:s[1]]) if s else "",
                "loc_confidence": loc.confidence,
                "candidate_overall_pass": report.result.overall_pass,
            })
    return rows


def span_status(sheet_span: str, region: str) -> str:
    """Check if the engine's (slack-padded) region contained the span the sheet points at."""
    s, r = _tokenize(sheet_span), _tokenize(region)
    if not s:
        return "sheet_blank"
    if not r:
        return "not_located"
    if _find_sublist(r, s):
        return "contains"
    return "partial" if set(s) & set(r) else "elsewhere"


def _outcome(engine: str, human: str) -> str:
    if engine == "unscored":
        return "engine_unscored"
    if engine == human:
        return "agree"
    return "false_fail" if engine == "fail" else "false_pass"


def _summarize(pairs: list[Pair]) -> dict:
    c = confusion(pairs)
    out = {"n": c.n, "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn}
    for name, fn in METRICS.items():
        ci = bootstrap_ci(pairs, fn)
        out[name] = {"value": fn(c), "ci": None if ci is None else [ci[0], ci[1]]}
    return out


def evaluate(pred_rows: list[dict], gold_rows, fixtures: dict) -> dict:
    preds = {(r["candidate_id"], r["unit_id"]): r for r in pred_rows}
    gold = {(g.candidate_id, g.unit_id): g for g in gold_rows}
    if missing := sorted(set(gold) - set(preds)):
        raise ValueError(f"gold units with no engine prediction: {missing[:5]}…")

    joined = []
    for key in sorted(gold):
        p, g = preds[key], gold[key]
        joined.append({
            "candidate_id": g.candidate_id, "fixture_id": g.fixture_id, "split": split_of(g.fixture_id),
            "unit_id": g.unit_id, "unit_type": g.unit_type,
            "engine": p["engine"], "human": g.human, "outcome": _outcome(p["engine"], g.human),
            "span_status": span_status(g.sheet_span, p["region"]),
            "sheet_span": g.sheet_span, "region": p["region"],
            "method": p["method"], "score": p["score"], "reason": p["reason"], "notes": g.notes,
        })

    pairs = [Pair(j["candidate_id"], j["unit_type"], j["engine"], j["human"], j["split"])
             for j in joined if j["engine"] in ("pass", "fail")]
    slices = {"all": pairs,
              "split:tune": [p for p in pairs if p.split == "tune"],
              "split:test": [p for p in pairs if p.split == "test"]}
    for t in sorted({p.unit_type for p in pairs}):
        slices[f"type:{t}"] = [p for p in pairs if p.unit_type == t]

    return {"joined": joined,
            "metrics": {name: _summarize(ps) for name, ps in slices.items()},
            "candidates": _candidate_level(pred_rows, gold_rows, fixtures),
            "coverage": {"gold_units": len(gold), "predicted_units": len(preds),
                         "engine_unscored": sum(j["outcome"] == "engine_unscored" for j in joined)}}


def _candidate_level(pred_rows, gold_rows, fixtures) -> dict:
    """Both sides judged by the same aggregate(), so only the unit verdicts differ."""
    engine_pass = {r["candidate_id"]: r["candidate_overall_pass"] in (True, "True")
                   for r in pred_rows}
    by_cand: dict[str, list] = {}
    for g in gold_rows:
        by_cand.setdefault(g.candidate_id, []).append(g)

    rows = []
    for cid, gs in sorted(by_cand.items()):
        fx = fixtures[gs[0].fixture_id]
        if len(gs) < len(fx.units):                # partial return: skip, don't guess
            continue
        human_verdicts = [UnitVerdict(unit_id=g.unit_id, unit_type=UnitType(g.unit_type),
                                      status=g.human, method="human", reason="") for g in gs]
        human_pass = aggregate(human_verdicts, fx).overall_pass
        rows.append({"candidate_id": cid, "engine_pass": engine_pass[cid], "human_pass": human_pass})
    agree = sum(r["engine_pass"] == r["human_pass"] for r in rows)
    return {"n": len(rows), "agree": agree, "rows": rows}



JOINED_COLUMNS = ["candidate_id", "fixture_id", "split", "unit_id", "unit_type", "engine", "human",
                  "outcome", "span_status", "sheet_span", "region", "method", "score", "reason", "notes"]


def _fmt(m: dict) -> str:
    if m["value"] is None:
        return "n/a"
    ci = f" [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]" if m["ci"] else " [n/a]"
    return f"{m['value']:.2f}{ci}"


def write_reports(result: dict, out_dir, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(result["joined"], out / "joined.csv", JOINED_COLUMNS)
    _write_csv([j for j in result["joined"] if j["outcome"] in ("false_fail", "false_pass")],
               out / "disagreements.csv", JOINED_COLUMNS)
    (out / "metrics.json").write_text(json.dumps(
        {"meta": meta, "metrics": result["metrics"], "candidates": result["candidates"],
         "coverage": result["coverage"]}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Validation run — {meta['date']}", "",
             f"engine `{meta['engine_version']}` · gold units {result['coverage']['gold_units']}"
             f" · blanks {meta['blanks']} · single rater (no human–human κ ceiling)", "",
             "| slice | n | TP | FP | FN | TN | agreement | precision_fail | recall_fail | κ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                     f"{_fmt(m['agreement'])} | {_fmt(m['precision_fail'])} | "
                     f"{_fmt(m['recall_fail'])} | {_fmt(m['kappa'])} |")
    c = result["candidates"]
    lines += ["", f"Candidate pass/fail agreement: {c['agree']}/{c['n']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")