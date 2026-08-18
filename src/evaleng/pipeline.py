"""The scoring pipeline: drive the matchers, tally the verdicts, render the report."""
from __future__ import annotations

from evaleng.schema.models import Fixture, UnitType
from evaleng.interfaces import CandidateInput, UnitVerdict, ScoreResult, FeedbackReport
from evaleng.dispatch import matcher_for, resolve_policy, MatcherKind, DISPATCH
from evaleng.localize import localize
from evaleng.match_number import match_number
from evaleng.match_lexical import match_lexical


def _unscored_verdict(unit) -> UnitVerdict:
    note = DISPATCH[unit.type].note or "not implemented in Week 1"
    return UnitVerdict(unit_id=unit.id, unit_type=unit.type, status="unscored",
                       candidate_text=None, method="deferred", reason=note, score=None)


def run_matchers(candidate: CandidateInput, fixture: Fixture, alignment) -> list[UnitVerdict]:
    verdicts = []
    for unit in fixture.units:
        location = alignment.locations[unit.id]
        kind = matcher_for(unit.type)
        if kind is MatcherKind.NUMBER:
            verdict = match_number(unit, candidate, location)
        elif kind is MatcherKind.LEXICAL:
            verdict = match_lexical(unit, candidate, location, resolve_policy(unit))
        else:
            verdict = _unscored_verdict(unit)
        verdicts.append(verdict)
    return verdicts


def aggregate(verdicts: list[UnitVerdict], fixture: Fixture) -> ScoreResult:
    weight = {u.id: u.weight for u in fixture.units}
    scored = [v for v in verdicts if v.status in ("pass", "fail")]
    weighted_passed = sum(weight[v.unit_id] for v in verdicts if v.status == "pass")
    weighted_scored = sum(weight[v.unit_id] for v in scored)
    achieved = (weighted_passed / weighted_scored) if weighted_scored else 0.0
    overall_pass = weighted_scored > 0 and achieved >= fixture.pass_ratio

    per_category: dict[UnitType, tuple[int, int]] = {}
    for v in scored:
        passed, total = per_category.get(v.unit_type, (0, 0))
        per_category[v.unit_type] = (passed + (1 if v.status == "pass" else 0), total + 1)

    return ScoreResult(verdicts=verdicts, weighted_passed=weighted_passed,
                       weighted_scored=weighted_scored, pass_ratio=fixture.pass_ratio,
                       overall_pass=overall_pass, coverage=(len(scored), len(verdicts)),
                       per_category=per_category)


_TAG = {"pass": "PASS", "fail": "FAIL", "unscored": "UNSCORED"}


def build_feedback(result: ScoreResult, fixture: Fixture) -> FeedbackReport:
    source_span = {u.id: u.source_span for u in fixture.units}
    verdict_by_id = {v.unit_id: v for v in result.verdicts}
    scored, total = result.coverage
    achieved = (result.weighted_passed / result.weighted_scored) if result.weighted_scored else 0.0
    outcome = "PASS" if result.overall_pass else "FAIL"

    lines = [
        f"Fixture {fixture.id} — {outcome}  (provisional: {total - scored} unit(s) not yet scored)",
        f"  scored {scored}/{total} units · passed {result.weighted_passed}/{result.weighted_scored}"
        f" = {achieved:.0%}  (required {result.pass_ratio:.0%})",
    ]
    if result.per_category:
        cats = ", ".join(f"{t.value} {p}/{n}" for t, (p, n)
                         in sorted(result.per_category.items(), key=lambda kv: kv[0].value))
        lines.append(f"  by category: {cats}")
    lines.append("")
    for unit in fixture.units:
        v = verdict_by_id[unit.id]
        produced = f'→ "{v.candidate_text}" ' if v.candidate_text else ""
        lines.append(f'  {_TAG[v.status]:9}[{unit.id} {unit.type.value}] '
                     f'"{source_span[unit.id]}" {produced}· {v.reason}')
    return FeedbackReport(result=result, lines=lines)


def score(candidate: CandidateInput, fixture: Fixture) -> FeedbackReport:
    alignment = localize(candidate, fixture)
    verdicts = run_matchers(candidate, fixture, alignment)
    result = aggregate(verdicts, fixture)
    return build_feedback(result, fixture)