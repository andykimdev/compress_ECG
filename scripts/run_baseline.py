#!/usr/bin/env python3
"""Run the released Mini or train the tabular baseline. Install first: pip install -e ."""

import argparse
import json
from pathlib import Path

from compress_ecg.baseline import DEFAULT_MINI_DIR, run_baseline, run_gradient_boosting
from compress_ecg.data import LABEL_COLUMNS, MODEL_SPLITS, SHD_INDEX
from compress_ecg.predictions import run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON settings; CLI flags override them")
    parser.add_argument("--data-dir", help="Official file root; defaults to ECHONEXT_DIR or data/echonext/1.1.1")
    parser.add_argument("--model", choices=("mini", "gradient_boosting"), default="mini")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", help="Explicit directory for this run")
    output.add_argument("--runs-root", help="Parent of the standard <model>/<split> folders")
    parser.add_argument("--split", choices=MODEL_SPLITS, default="val")
    parser.add_argument("--limit", type=int, default=1000, help="Number of ECGs to score; 0 for the full split")
    parser.add_argument("--train-limit", type=int, help="Limit gradient boosting training for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Mini: cpu, cuda or mps; gradient boosting: cpu")
    parser.add_argument("--mini-dir", default=str(DEFAULT_MINI_DIR))
    parser.add_argument("--data-version", default="1.1.1")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            parser.error("Config must be a JSON object with recognized argument names")
        if preliminary.output_dir is not None:
            settings.pop("runs_root", None)
        if preliminary.runs_root is not None:
            settings.pop("output_dir", None)
        parser.set_defaults(**settings)
    args = parser.parse_args()
    if args.split not in MODEL_SPLITS:
        parser.error("split must be train, val or test")
    if args.model not in ("mini", "gradient_boosting"):
        parser.error("model must be mini or gradient_boosting")
    if args.output_dir is not None and args.runs_root is not None:
        parser.error("Choose either output_dir or runs_root")
    if args.train_limit is not None and (args.model != "gradient_boosting" or args.train_limit <= 0):
        parser.error("--train-limit requires gradient_boosting and a positive limit")
    if args.model == "gradient_boosting" and args.device != "cpu":
        parser.error("gradient_boosting runs on CPU; use --device cpu")
    limit = None if args.limit == 0 else args.limit
    runs_root = args.runs_root or ("runs" if limit is None and args.train_limit is None else "runs/smoke")
    destination = args.output_dir or run_dir(args.model, args.split, runs_root)
    options = {"split": args.split, "limit": limit, "seed": args.seed, "data_version": args.data_version}
    if args.model == "mini":
        result = run_baseline(
            args.data_dir, destination, batch_size=args.batch_size, num_workers=args.num_workers,
            device=args.device, mini_dir=args.mini_dir, **options,
        )
    else:
        result = run_gradient_boosting(args.data_dir, destination, train_limit=args.train_limit, **options)
    summary = {"output_dir": result["output_dir"], "n_records": result["n_records"],
               "shd": result["metrics"][LABEL_COLUMNS[SHD_INDEX]]}
    # Only the Mini run reports its prediction time and model size.
    summary.update({key: result[key] for key in ("predict_time", "model_size_bytes") if key in result})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
