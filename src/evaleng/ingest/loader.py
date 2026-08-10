from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import ValidationError
from evaleng.schema.models import Fixture


class FixtureError(Exception):
    """Raised when a fixture is missing, unparseable, or fails validation."""


def load_fixture(path: str | Path) -> Fixture:
    path = Path(path)
    if not path.exists():
        raise FixtureError(f"fixture not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise FixtureError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise FixtureError(f"fixture root must be a mapping, got {type(raw).__name__}")
    try:
        return Fixture.model_validate(raw)
    except ValidationError as e:
        raise FixtureError(f"{path} failed validation:\n{e}") from e