#!/usr/bin/env python3
"""Role A: measure speed and memory for each model. Install first: pip install -e ."""

import argparse
from pathlib import Path

from compress_ecg.baseline import load_mini
from compress_ecg.profiling import profile_models
from compress_ecg.student import load_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", help="Student checkpoint, once it exists")
    parser.add_argument("--output", type=Path, default=Path("results/profiling/profile.csv"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = parser.parse_args()

    models = {"mini": load_mini(device=args.device)}
    if args.student:
        models["student"] = load_student(args.student, device=args.device)
    table = profile_models(models, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
