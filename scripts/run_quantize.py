#!/usr/bin/env python3
"""Role D: make FP16 and INT8 models and save their predictions."""

import argparse

from compress_ecg.baseline import load_mini
from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import write_run
from compress_ecg.quantize import (
    to_fp16,
    to_int8_post_training,
    train_int8_quant_aware,
)
from compress_ecg.student import load_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="EchoNext dataset directory")
    parser.add_argument("--student", help="Student checkpoint; omit to use Mini")
    parser.add_argument("--output-dir", default="checkpoints/quantized")
    parser.add_argument("--runs-root", default="runs/dummy")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument(
        "--fp16-only",
        action="store_true",
        help="Run FP16 only; skip unfinished INT8 and QAT",
    )
    parser.add_argument("--with-test", action="store_true")
    args = parser.parse_args()

    source = (
        load_student(args.student, device=args.device)
        if args.student
        else load_mini(device=args.device)
    )

    val = EchoNextDataset(args.data_dir, "val")

    if args.fp16_only:
        models = {
            "student_fp16": to_fp16(source),
        }
    else:
        train = EchoNextDataset(args.data_dir, "train")
        models = {
            "student_fp16": to_fp16(source),
            "student_int8_post": to_int8_post_training(source, train),
            "student_int8_qat": train_int8_quant_aware(
                source, train, val, args.output_dir, device=args.device
            ),
        }

    datasets = [val] + (
        [EchoNextDataset(args.data_dir, "test")]
        if args.with_test
        else []
    )

    for name, model in models.items():
        for dataset in datasets:
            print(
                write_run(
                    model,
                    name,
                    dataset,
                    runs_root=args.runs_root,
                    device=args.device,
                    source_model="student" if args.student else "EchoNext-Mini",
                    quantization=name,
                )
            )


if __name__ == "__main__":
    main()
