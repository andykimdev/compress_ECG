#!/usr/bin/env python3
"""Role A: measure speed and memory for each model. Install first: pip install -e ."""

import argparse
from pathlib import Path

import torch

from compress_ecg.baseline import DEFAULT_MINI_DIR, MINI_CHECKPOINT, load_mini
from compress_ecg.profiling import BACKENDS, profile_models
from compress_ecg.student import load_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", help="Student checkpoint, once it exists")
    parser.add_argument("--mini-dir", type=Path, default=DEFAULT_MINI_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/profiling/profile.csv"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["eager"])
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per runtime")
    parser.add_argument("--trace-dir", type=Path, help="Save PyTorch Profiler traces here")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)

    models = {"mini": load_mini(args.mini_dir)}
    checkpoint_paths = {"mini": args.mini_dir / MINI_CHECKPOINT}
    if args.student:
        models["student"] = load_student(args.student, mini_dir=args.mini_dir)
        checkpoint_paths["student"] = args.student
    table = profile_models(
        models, device=args.device, batch_sizes=tuple(args.batch_sizes), backends=tuple(args.backends),
        warmup=args.warmup, repeats=args.repeats, checkpoint_paths=checkpoint_paths,
        trace_dir=args.trace_dir, compile_backend=args.compile_backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
