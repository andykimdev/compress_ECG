#!/usr/bin/env python3
"""Role D: make the FP16 and INT8 models and save their predictions. Install first: pip install -e ."""

import argparse
import copy

from compress_ecg.baseline import load_mini
from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import write_run
from compress_ecg.quantize import to_fp16, to_int8_post_training, train_int8_quant_aware
from compress_ecg.student import load_student


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--student", help="Student checkpoint. Leave out in Phase 1 to use EchoNext-Mini")
    parser.add_argument("--output-dir", default="checkpoints/quantized")
    parser.add_argument("--runs-root", default="runs/dummy", help="Use runs in Phase 2")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    args = parser.parse_args(argv)

    source = load_student(args.student, device=args.device) if args.student else load_mini(device=args.device)
    train = EchoNextDataset(args.data_dir, "train")
    val = EchoNextDataset(args.data_dir, "val")
    # Each conversion gets its own copy, since some, such as half(), change a model in place.
    conversions = {
        "student_fp16": lambda: to_fp16(copy.deepcopy(source)),
        "student_int8_post": lambda: to_int8_post_training(copy.deepcopy(source), train),
        "student_int8_qat": lambda: train_int8_quant_aware(
            copy.deepcopy(source), train, val, args.output_dir, device=args.device,
        ),
    }
    datasets = [val] + ([EchoNextDataset(args.data_dir, "test")] if args.with_test else [])
    for name, convert in conversions.items():
        # A version still being written is skipped, so the finished ones can be scored meanwhile.
        try:
            model = convert()
        except NotImplementedError as error:
            print(f"Skipped {name}: {error}")
            continue
        for dataset in datasets:
            print(write_run(model, name, dataset, runs_root=args.runs_root, device=args.device))


if __name__ == "__main__":
    main()
