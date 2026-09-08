from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class UnitType(str, Enum):
    GRAMMAR = "grammar"
    FALSE_COGNATE = "false_cognate"
    GENERAL_VOCABULARY = "general_vocabulary"
    LEGAL_VOCABULARY = "legal_vocabulary"
    IDIOM = "idiom"
    NUMBER = "number"
    MODIFIER = "modifier"
    REGISTER = "register"
    POSITION = "position"
    SLANG = "slang"


class FeatureSpec(BaseModel):
    upos: str | None = None                               # part of speech, e.g. "AUX"
    feats: dict[str, str] = Field(default_factory=dict)   # e.g. {"Tense": "Fut"}
    dep_relation_to_head: str | None = None               # e.g. how a word attaches to another


class RegisterClass(str, Enum):
    """The formality levels a rendering can be judged against, least to most formal."""
    INFORMAL = "informal"
    NEUTRAL  = "neutral"
    FORMAL   = "formal"


class RegisterSpec(BaseModel):
    """What register a unit requires. For now, just the target level."""
    required: RegisterClass


class MatchingPolicy(BaseModel):
    method: str
    fuzzy_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_backup: bool = False
    feature_spec: FeatureSpec | None = None
    register_spec: RegisterSpec | None = None

    @field_validator("method")
    @classmethod
    def method_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("matching_policy.method must be non-empty")
        return v

    # A policy must carry exactly one way to judge its unit: 
    # a fuzzy_threshold (text overlap),
    # a feature_spec (grammar),
    # or a register_spec (formality). A policy with none is half-written.
    @model_validator(mode="after")
    def requires_a_matching_signal(self) -> "MatchingPolicy":
        if (self.feature_spec is None
                and self.fuzzy_threshold is None
                and self.register_spec is None):
            raise ValueError(
                "matching_policy needs one of: fuzzy_threshold, feature_spec, or register_spec"
            )
        return self


class Unit(BaseModel):
    id: str
    type: UnitType
    source_span: str
    acceptable_renderings: list[str] = Field(default_factory=list)
    matching_policy: MatchingPolicy | None = Field(default=None)
    weight: int = Field(1, ge=1)

    @field_validator("id", "source_span")
    @classmethod
    def nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v

    @model_validator(mode="after")
    def require_acceptable_renderings(self) -> "Unit":
        if not self.acceptable_renderings:
            raise ValueError(
                f"unit {self.id!r} ({self.type.value}) has no acceptable_renderings"
            )
        return self


class Fixture(BaseModel):
    id: str
    source_lang: str = "en"
    target_lang: str = "es"
    source_text: str
    reference_rendering: str
    units: list[Unit]
    pass_ratio: float = Field(0.7, ge=0.0, le=1.0) # fixture inherits schema default (0.70). To override, set it in individual fixtures.

    @model_validator(mode="after")
    def coherence(self) -> "Fixture":
        if not self.units:
            raise ValueError("fixture has no units")
        ids = [u.id for u in self.units]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate unit ids: {dupes}")
        return self