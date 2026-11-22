#!/usr/bin/env python3
"""Compare two saved runs on the same ECGs with a paired interval. Install first: pip install -e ."""

import argparse
import json
from pathlib import Path

from compress_ecg.compare import compare_runs
from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX
from compress_ecg.predictions import load_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Run folder of the first model")
    parser.add_argument("--b", required=True, help="Run folder of the second model, on the same ECGs")
    parser.add_argument("--label", default=LABEL_COLUMNS[SHD_INDEX], choices=LABEL_COLUMNS)
    parser.add_argument("--resamples", type=int, default=2000, help="Redraws of the patients")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="JSON file for the summary numbers, which are safe to commit")
    args = parser.parse_args(argv)
    run_a, run_b = load_run(args.a), load_run(args.b)
    if run_a.manifest["split"] != run_b.manifest["split"]:
        raise SystemExit(f"Runs are on different splits: {run_a.manifest['split']} and {run_b.manifest['split']}")
    result = {
        "split": run_a.manifest["split"], "model_a": run_a.manifest["model"], "model_b": run_b.manifest["model"],
        "seed": args.seed, **compare_runs(run_a, run_b, label=args.label, n_resamples=args.resamples, seed=args.seed),
    }
    text = json.dumps({key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()},
                      indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
