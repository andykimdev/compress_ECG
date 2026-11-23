#!/usr/bin/env python3
"""
Profile Student training steps on the real data, split into waiting for data and computing.

Purpose:
- Time each training step's wait for its batch and its device work, for several loader
  worker counts, with the same loaders, model, and step that training uses.
- Check that every worker count serves the same batches in the same order.
- Save a PyTorch Profiler table of the operators that take the most time.

Special Notes:
- Writes aggregate timings only. No patient data goes into the output.
- Install first: pip install -e .
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch
from torch.optim.adamw import AdamW
from torch.profiler import ProfilerActivity, profile, record_function

from compress_ecg.data import EchoNextDataset
from compress_ecg.distill import distillation_step, make_loaders
from compress_ecg.predictions import LogitLookup, _code_version, load_run
from compress_ecg.student import build_student
from compress_ecg.timing import step_summary

# Settings of train_student that the timed steps copy.
LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, HARD_LABEL_WEIGHT = 1e-3, 1e-4, 2.0, 0.5
# Batches whose row IDs are compared across worker counts.
ORDER_CHECK_BATCHES = 50


def time_steps(train, val, teacher_run, *, device, batch_size, num_workers, steps, warmup, seed):
    """Run warmup plus steps training steps and return the timed waits, computes, and batch row IDs."""
    torch.manual_seed(seed)
    student = build_student(device=device)
    student.train()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader, _ = make_loaders(train, val, batch_size=batch_size, num_workers=num_workers, seed=seed)
    batches = iter(loader)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    waits, computes, row_ids = [], [], []
    for index in range(warmup + steps):
        start = time.perf_counter()
        batch = next(batches)
        fetched = time.perf_counter()
        distillation_step(
            student, batch, teacher_run, optimizer, device=device,
            temperature=TEMPERATURE, hard_label_weight=HARD_LABEL_WEIGHT,
        )
        sync()
        done = time.perf_counter()
        if len(row_ids) < ORDER_CHECK_BATCHES:
            row_ids.append(batch["row_id"].tolist())
        if index >= warmup:
            waits.append(fetched - start)
            computes.append(done - fetched)
    return waits, computes, row_ids


def profiler_table(train, val, teacher_run, *, device, batch_size, num_workers, steps, seed) -> str:
    """Profile a few steps with PyTorch Profiler and return its table of the costliest operators."""
    torch.manual_seed(seed)
    student = build_student(device=device)
    student.train()
    optimizer = AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader, _ = make_loaders(train, val, batch_size=batch_size, num_workers=num_workers, seed=seed)
    batches = iter(loader)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
    with profile(activities=activities) as profiler:
        for _ in range(steps):
            with record_function("wait_for_batch"):
                batch = next(batches)
            with record_function("training_step"):
                distillation_step(
                    student, batch, teacher_run, optimizer, device=device,
                    temperature=TEMPERATURE, hard_label_weight=HARD_LABEL_WEIGHT,
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    return profiler.key_averages().table(sort_by="cpu_time_total", row_limit=30)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--teacher-run", default="runs/teacher/train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--steps", type=int, default=300, help="Timed steps per worker count")
    parser.add_argument("--warmup", type=int, default=30, help="Untimed steps first, per worker count")
    parser.add_argument("--profile-steps", type=int, default=40, help="Steps under PyTorch Profiler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", required=True, help="Names this measurement, such as before or after")
    parser.add_argument("--output-dir", default="results/student/profiling")
    args = parser.parse_args(argv)

    train = EchoNextDataset(args.data_dir, "train")
    val = EchoNextDataset(args.data_dir, "val")
    teacher_run = LogitLookup(load_run(args.teacher_run))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows, orders = [], {}
    for workers in args.num_workers:
        waits, computes, orders[workers] = time_steps(
            train, val, teacher_run, device=args.device, batch_size=args.batch_size, num_workers=workers,
            steps=args.steps, warmup=args.warmup, seed=args.seed,
        )
        summary = {"num_workers": workers, **step_summary(waits, computes, args.batch_size)}
        summary["epoch_minutes_estimate"] = len(train) / summary["ecgs_per_second"] / 60
        rows.append(summary)
        print(json.dumps(summary))

    first = orders[args.num_workers[0]]
    record = {
        "label": args.label,
        "workload": "Student training steps (forward, distillation loss, backward, AdamW) on Training ECGs",
        "timed": "wait = time to receive the next batch from the loader; compute = copy to device through "
                 "the optimizer step, ending with a device synchronize",
        "train_ecgs": len(train), "batch_size": args.batch_size, "precision": "fp32",
        "warmup_steps": args.warmup, "timed_steps": args.steps, "seed": args.seed,
        "same_batches_for_every_worker_count": all(order == first for order in orders.values()),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "cpu_count": os.cpu_count(), "python": platform.python_version(), "torch": torch.__version__,
        "code": _code_version(), "rows": rows,
    }
    (output / f"steps_{args.label}.json").write_text(json.dumps(record, indent=2) + "\n")
    table = profiler_table(
        train, val, teacher_run, device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers[0], steps=args.profile_steps, seed=args.seed,
    )
    (output / f"profiler_{args.label}_w{args.num_workers[0]}.txt").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
