#!/usr/bin/env python3
"""Run A's reproducible Mini baseline. Install first: pip install -e ."""

import argparse
import json
from pathlib import Path

from compress_ecg.baseline import DEFAULT_MINI_DIR, run_baseline
from compress_ecg.data import SHD_INDEX, LABEL_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON settings; CLI flags override them")
    parser.add_argument("--data-dir", help="Official file root; defaults to ECHONEXT_DIR")
    parser.add_argument("--output-dir", default="runs/mini_val_1000_seed42")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=1000, help="0 for full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--mini-dir", default=str(DEFAULT_MINI_DIR))
    parser.add_argument("--data-version", default="1.1.1")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            parser.error("Config must be a JSON object with recognized argument names")
        parser.set_defaults(**settings)
    args = vars(parser.parse_args())
    args.pop("config")
    if args["split"] not in ("train", "val", "test"):
        parser.error("split must be train, val or test")
    if args["limit"] == 0:
        args["limit"] = None
    result = run_baseline(**args)
    print(json.dumps({"output_dir": result["output_dir"], "n_records": result["n_records"],
                      "shd": result["metrics"][LABEL_COLUMNS[SHD_INDEX]]}, indent=2))


if __name__ == "__main__":
    main()
