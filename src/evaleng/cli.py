"""Command-line entry point: score a transcript against a fixture."""
from __future__ import annotations
import argparse
import sys

from evaleng.ingest.loader import load_fixture, FixtureError
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
        fixture = load_fixture(args.fixture)
    except FixtureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    text = open(args.transcript, encoding="utf-8").read().strip()
    candidate = CandidateInput(text=text, tokens=text.split())

    report = score(candidate, fixture)
    print("\n".join(report.lines))
    return 0 if report.result.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())