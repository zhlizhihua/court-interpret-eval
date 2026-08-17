"""Typed data contracts passed between pipeline stages."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from evaleng.schema.models import UnitType


class CandidateInput(BaseModel):
    """What the candidate produced, as it enters the pipeline."""
    text: str
    tokens: list[str]
    timings: list[tuple[float, float]] | None = None
    nbest: list[str] | None = None


class UnitLocation(BaseModel):
    """Where one scoring unit landed in the candidate — or didn't."""
    unit_id: str
    token_span: tuple[int, int] | None = None   # None = omission
    char_span: tuple[int, int] | None = None
    confidence: float = 0.0


class AlignmentResult(BaseModel):
    """Every unit's location for one candidate, looked up by unit id."""
    locations: dict[str, UnitLocation]


VerdictStatus = Literal["pass", "fail", "unscored"]


class UnitVerdict(BaseModel):
    """The engine's judgment on one unit, with the evidence behind it."""
    unit_id: str
    unit_type: UnitType
    status: VerdictStatus
    candidate_text: str | None = None
    method: str
    reason: str
    score: float | None = None


class ScoreResult(BaseModel):
    """The overall result, computed over the units that were scored."""
    verdicts: list[UnitVerdict]
    weighted_passed: int
    weighted_scored: int
    pass_ratio: float
    overall_pass: bool
    coverage: tuple[int, int]                        # (scored, total)
    per_category: dict[UnitType, tuple[int, int]]    # category -> (passed, total)


class FeedbackReport(BaseModel):
    """The final human-readable report."""
    result: ScoreResult
    lines: list[str]