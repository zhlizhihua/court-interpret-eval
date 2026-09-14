"""The scoring pipeline: drive the matchers, tally the verdicts, render the report."""
from __future__ import annotations

from evaleng.schema.models import Fixture, UnitType
from evaleng.interfaces import CandidateInput, UnitVerdict, ScoreResult, FeedbackReport
from evaleng.dispatch import matcher_for, resolve_policy, MatcherKind, DISPATCH
from evaleng.localize import localize
from evaleng.match.number import match_number
from evaleng.match.lexical import match_lexical
from evaleng.match.grammar import match_grammar
from evaleng.match.register import match_register


def _unscored_verdict(unit) -> UnitVerdict:
    note = DISPATCH[unit.type].note or "not implemented in Week 1"
    return UnitVerdict(unit_id=unit.id, unit_type=unit.type, status="unscored",
                       candidate_text=None, method="deferred", reason=note, score=None)


def _judge(unit, candidate, location) -> UnitVerdict:
    """Run the matcher for one unit against one candidate+location."""
    kind = matcher_for(unit.type)
    if kind is MatcherKind.NUMBER:
        return match_number(unit, candidate, location)
    if kind is MatcherKind.LEXICAL:
        return match_lexical(unit, candidate, location, resolve_policy(unit))
    if kind is MatcherKind.GRAMMAR:
        return match_grammar(unit, candidate, location, resolve_policy(unit))
    if kind is MatcherKind.REGISTER:
        return match_register(unit, candidate, location, resolve_policy(unit))
    return _unscored_verdict(unit)

def run_matchers(candidate, fixture, alignment) -> list[UnitVerdict]:
    return [_judge(unit, candidate, alignment.locations[unit.id])
            for unit in fixture.units]


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
    verdicts = _rescue_with_nbest(verdicts, candidate, fixture)
    result = aggregate(verdicts, fixture)
    return build_feedback(result, fixture)


def _rescue_with_nbest(verdicts, candidate, fixture) -> list[UnitVerdict]:
    """Turn a 1-best fail into a pass if an ASR alternative supports it. Never the reverse."""
    if not candidate.nbest:
        return verdicts
    by_id = {v.unit_id: v for v in verdicts}
    unit_by_id = {u.id: u for u in fixture.units}
    failing = {v.unit_id for v in verdicts if v.status == "fail"}

    for k, alt_text in enumerate(candidate.nbest, start=1):
        if not failing:
            break
        alt = CandidateInput(text=alt_text, tokens=alt_text.split())
        alt_align = localize(alt, fixture)
        for uid in list(failing):
            v = _judge(unit_by_id[uid], alt, alt_align.locations[uid])
            if v.status == "pass":
                v = v.model_copy(update={
                    "method": f"{v.method}+nbest",
                    "reason": f"{v.reason} [rescued via ASR alternative #{k}]",
                })
                by_id[uid] = v
                failing.discard(uid)

    return [by_id[u.id] for u in fixture.units]   # preserve fixture order