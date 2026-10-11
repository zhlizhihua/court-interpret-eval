"""python -m evaleng.validate predict|evaluate"""
from __future__ import annotations
import argparse, csv, sys
from datetime import date
from pathlib import Path

from evaleng.validate.gold import load_gold, GoldError
from evaleng.validate.harness import (PRED_COLUMNS, _write_csv, evaluate, load_fixtures,
                                      load_transcripts, predict, write_reports)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaleng.validate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="run the engine on every candidate; write a predictions CSV")
    p.add_argument("--candidates", default="data/candidates")
    p.add_argument("--fixtures", default="data/fixtures")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-dirty", action="store_true",
                   help="permit uncommitted code (never for a freeze)")

    e = sub.add_parser("evaluate", help="compare a predictions CSV to returned gold sheets")
    e.add_argument("--predictions", required=True)
    e.add_argument("--gold", required=True, help="folder of returned Part 2 CSV sheets")
    e.add_argument("--candidates", default="data/candidates")
    e.add_argument("--fixtures", default="data/fixtures")
    e.add_argument("--out", required=True)
    e.add_argument("--allow-blank", action="store_true", help="evaluate a partial return")
    args = ap.parse_args(argv)

    if args.cmd == "predict":
        rows = predict(args.candidates, args.fixtures)
        if rows and rows[0]["engine_version"].endswith("-dirty") and not args.allow_dirty:
            print("error: uncommitted changes — commit first, or pass --allow-dirty", file=sys.stderr)
            return 2
        _write_csv(rows, args.out, PRED_COLUMNS)
        print(f"wrote {len(rows)} unit predictions -> {args.out}")
        return 0

    fixtures = load_fixtures(args.fixtures)
    try:
        gold, blanks = load_gold(Path(args.gold).glob("*.csv"), fixtures,
                                 load_transcripts(args.candidates), allow_blank=args.allow_blank)
    except GoldError as err:
        print(err, file=sys.stderr)
        return 2
    pred_rows = list(csv.DictReader(open(args.predictions, encoding="utf-8")))
    result = evaluate(pred_rows, gold, fixtures)
    write_reports(result, args.out, meta={
        "date": date.today().isoformat(), "engine_version": pred_rows[0]["engine_version"],
        "predictions": args.predictions, "gold": args.gold, "blanks": blanks})
    print((Path(args.out) / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())