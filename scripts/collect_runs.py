#!/usr/bin/env python3
"""Rebuild the table of Teacher runs from their run folders. Install first: pip install -e ."""

import argparse
from pathlib import Path

from compress_ecg.runs import collect_runs, round_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/teacher/runs", help="Folder of run folders")
    parser.add_argument("--output", default="results/teacher/runs.csv", help="Aggregate scores only")
    args = parser.parse_args()
    table = collect_runs(args.root)
    if table.empty:
        raise SystemExit(f"No runs found under {args.root}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    round_scores(table).to_csv(args.output, index=False)
    print(round_scores(table).to_string(index=False))


if __name__ == "__main__":
    main()
