"""Command-line entry point: score a transcript against a fixture."""
from __future__ import annotations
import argparse
import sys

import yaml

from evaleng.schema.models import Fixture
from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a sight-translation rendering against a fixture.")
    parser.add_argument("--fixture", required=True, help="Path to the fixture YAML.")
    parser.add_argument("--transcript", required=True,
                        help="Path to the candidate transcript (plain UTF-8 text).")
    args = parser.parse_args(argv)

    try:
        fixture = Fixture(**yaml.safe_load(open(args.fixture, encoding="utf-8")))
    except Exception as e:
        print(f"error: could not load fixture {args.fixture!r}: {e}", file=sys.stderr)
        return 2

    text = open(args.transcript, encoding="utf-8").read().strip()
    candidate = CandidateInput(text=text, tokens=text.split())

    report = score(candidate, fixture)
    print("\n".join(report.lines))
    return 0 if report.result.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())