#!/usr/bin/env python3
"""
Measure a saved Student's prediction speed and GPU memory, the same way time_teacher.py does.

Purpose:
- Time the Student's forward pass at several batch sizes, in FP32 and FP16.
- Write rows in the same format as the Teacher's speed file, so the two compare directly.

Special Notes:
- FP16 converts the weights and inputs to 16 bits, as the FP16 Student is stored.
- Install first: pip install -e .
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compress_ecg.baseline import DEFAULT_MINI_DIR
from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import run_manifest, sha256_file
from compress_ecg.student import load_student
from compress_ecg.timing import forward_seconds, timing_summary

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
TIMED = (
    "The model's forward pass on a batch that is already on the device. "
    "Reading from disk and data loading are not timed."
)
MEMORY = (
    "Bytes from torch.cuda.max_memory_allocated over the warm-up and timed calls of one row. "
    "The model's weights and the input batch are included."
)


def main(argv: list[str] | None = None) -> None:
    """
    Time a saved Student at several batch sizes and write the summary numbers.

    Args:
        argv: Command line arguments without the program name. None reads them from sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="best_student.pt from run_student.py")
    parser.add_argument("--mini-dir", type=Path, default=DEFAULT_MINI_DIR)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="ECGs that fill the batches")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--precisions", nargs="+", default=["fp32"], choices=tuple(DTYPES))
    parser.add_argument("--warmup", type=int, default=10, help="Untimed calls before each measurement")
    parser.add_argument("--repeats", type=int, default=50, help="Timed calls per measurement")
    parser.add_argument("--output", default="results/student/student_speed.json",
                        help="JSON file for the summary numbers, which are safe to commit")
    args = parser.parse_args(argv)

    dataset = EchoNextDataset(args.data_dir, args.split)
    if max(args.batch_sizes) > len(dataset):
        raise SystemExit(
            f"The {args.split} split has {len(dataset)} ECGs, fewer than batch size {max(args.batch_sizes)}"
        )
    batch = next(iter(DataLoader(dataset, batch_size=max(args.batch_sizes), shuffle=False)))
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"

    rows = []
    for precision in args.precisions:
        dtype = DTYPES[precision]
        model = load_student(args.checkpoint, mini_dir=args.mini_dir, device=args.device).to(dtype)
        for batch_size in args.batch_sizes:
            waveform, tabular = (batch[key][:batch_size].to(device, dtype) for key in ("waveform", "tabular"))
            if on_gpu:
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                seconds = forward_seconds(
                    model, waveform, tabular, warmup=args.warmup, repeats=args.repeats,
                    synchronize=(lambda: torch.cuda.synchronize(device)) if on_gpu else None,
                )
            summary = {key: round(value, 4) if isinstance(value, float) else value
                       for key, value in timing_summary(seconds, batch_size).items()}
            rows.append({
                "precision": precision, **summary,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if on_gpu else None,
            })

    manifest = run_manifest("student", dataset)
    result = {
        "model": "student", "checkpoint": args.checkpoint, "checkpoint_sha256": sha256_file(args.checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "code": manifest["code"], "torch": manifest["versions"]["torch"], "split": args.split,
        "device": args.device, "gpu": torch.cuda.get_device_name(device) if on_gpu else None,
        "warmup": args.warmup, "repeats": args.repeats, "timed": TIMED, "peak_gpu_memory": MEMORY,
        "rows": rows,
    }
    text = json.dumps(result, indent=2) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
