#!/usr/bin/env python3
"""Role C: train the Student from saved Teacher logits. Install first: pip install -e ."""

import argparse

from compress_ecg.data import EchoNextDataset
from compress_ecg.distill import train_student
from compress_ecg.predictions import load_run, write_run
from compress_ecg.student import build_student, load_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--teacher-run", default="runs/teacher/train",
                        help="In Phase 1, use EchoNext-Mini's saved run on the Training ECGs")
    parser.add_argument("--output-dir", default="checkpoints/student")
    parser.add_argument("--runs-root", default="runs", help="In Phase 1, use runs/dummy")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    args = parser.parse_args()

    train = EchoNextDataset(args.data_dir, "train")
    val = EchoNextDataset(args.data_dir, "val")
    student = build_student(device=args.device)
    checkpoint = train_student(student, load_run(args.teacher_run), train, val, args.output_dir, device=args.device)
    trained = load_student(checkpoint, device=args.device)
    datasets = [val] + ([EchoNextDataset(args.data_dir, "test")] if args.with_test else [])
    for dataset in datasets:
        print(write_run(trained, "student", dataset, runs_root=args.runs_root, device=args.device))


if __name__ == "__main__":
    main()
