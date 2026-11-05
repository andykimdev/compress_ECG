#!/usr/bin/env python3
"""Role B: fine-tune the Teacher and save its predictions. Install first: pip install -e ."""

import argparse

from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import write_run
from compress_ecg.teacher import build_teacher, load_frozen_teacher, train_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--pretrained", default="checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--output-dir", default="checkpoints/teacher")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    args = parser.parse_args()

    train = EchoNextDataset(args.data_dir, "train")
    val = EchoNextDataset(args.data_dir, "val")
    teacher = build_teacher(args.pretrained, device=args.device)
    checkpoint = train_teacher(teacher, train, val, args.output_dir, device=args.device)
    frozen = load_frozen_teacher(checkpoint, device=args.device)
    datasets = [train, val] + ([EchoNextDataset(args.data_dir, "test")] if args.with_test else [])
    for dataset in datasets:
        print(write_run(frozen, "teacher", dataset, runs_root=args.runs_root, device=args.device))


if __name__ == "__main__":
    main()
