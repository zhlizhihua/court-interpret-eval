"""Command-line entry point: score a transcript against a fixture."""
from __future__ import annotations
import argparse
import sys
import logging
import os

from evaleng.ingest.loader import load_fixture, FixtureError
from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a sight-translation rendering against a fixture.")
    parser.add_argument("--fixture", required=True, help="Path to the fixture YAML.")
    parser.add_argument("--transcript",
                        help="Path to the candidate transcript (plain UTF-8 text).")
    parser.add_argument("--audio",
                        help="Path to candidate audio; transcribed with Whisper (ASR).")
    parser.add_argument("--asr-model", default="small",
                        help="Whisper model size for --audio (default: small).")
    parser.add_argument("--nbest", type=int, default=0, metavar="N",
                        help="With --audio, sample N extra ASR hypotheses for n-best rescue "
                             "(default: 0 = 1-best only).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show progress logs (model load, transcription).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    if bool(args.transcript) == bool(args.audio):
        parser.error("provide exactly one of --transcript or --audio")

    try:
        fixture = load_fixture(args.fixture)
    except FixtureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.audio:
        if not os.path.exists(args.audio):
            print(f"error: audio file not found: {args.audio}", file=sys.stderr)
            return 2
        try:
            from evaleng.asr import transcribe
            candidate = transcribe(args.audio, size=args.asr_model, n_alternatives=args.nbest)
        except Exception as e:          # top-level CLI boundary: any ASR failure → clean exit, not a traceback
            print(f"error: transcription failed ({type(e).__name__}: {e})", file=sys.stderr)
            return 2
    else:
        text = open(args.transcript, encoding="utf-8").read().strip()
        candidate = CandidateInput(text=text, tokens=text.split())

    report = score(candidate, fixture)
    print("\n".join(report.lines))
    return 0 if report.result.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())