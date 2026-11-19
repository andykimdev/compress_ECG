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
    parser.add_argument("--probe-head", help="Output layer from run_teacher_probe.py to start from")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for ECG-FM")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for the output layer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200, help="Steps between progress lines")
    parser.add_argument("--limit", type=int, help="Use only this many ECGs per split, for a quick check")
    args = parser.parse_args()

    train = EchoNextDataset(args.data_dir, "train", limit=args.limit)
    val = EchoNextDataset(args.data_dir, "val", limit=args.limit)
    teacher = build_teacher(args.pretrained, device=args.device)
    checkpoint = train_teacher(
        teacher, train, val, args.output_dir, device=args.device, probe_head=args.probe_head,
        lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, num_workers=args.num_workers, log_every=args.log_every,
    )
    frozen = load_frozen_teacher(checkpoint, pretrained_path=args.pretrained, device=args.device)
    datasets = [train, val] + ([EchoNextDataset(args.data_dir, "test")] if args.with_test else [])
    for dataset in datasets:
        print(write_run(
            frozen, "teacher", dataset, runs_root=args.runs_root, batch_size=args.batch_size,
            num_workers=args.num_workers, device=args.device,
        ))


if __name__ == "__main__":
    main()
