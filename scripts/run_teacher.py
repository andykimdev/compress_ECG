#!/usr/bin/env python3
"""Role B: fine-tune the Teacher, or load a saved one, and save its predictions. Install first: pip install -e ."""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import sha256_file, write_run
from compress_ecg.runs import write_result, write_run_record
from compress_ecg.teacher import build_teacher, load_frozen_teacher, train_teacher


def best_epoch(log_path: Path) -> dict:
    """Return the log row with the best Validation mean AUROC, and how many epochs ran."""
    rows = list(csv.DictReader(log_path.open()))
    best = max(range(len(rows)), key=lambda i: (float(rows[i]["val_mean_auroc"]), i))
    return {
        "best_epoch": int(rows[best]["epoch"]), "epochs_run": len(rows),
        "val_mean_auroc": float(rows[best]["val_mean_auroc"]), "val_shd_auroc": float(rows[best]["val_shd_auroc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--pretrained", default="checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--run-id", help="Name of the run. Defaults to the start time")
    parser.add_argument("--output-dir", help="Run folder. Defaults to checkpoints/teacher/runs/<run id>")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--score-splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    parser.add_argument("--checkpoint", help="Score this saved Teacher and skip training")
    parser.add_argument("--probe-head", help="Output layer from run_teacher_probe.py to start from")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for ECG-FM")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for the output layer")
    parser.add_argument("--head-dropout", type=float, default=0.0, help="Dropout before the output layer")
    parser.add_argument("--lead-mask-prob", type=float, default=0.0, help="Chance each lead is blanked in training")
    parser.add_argument("--balance-patients", type=float, default=0.0, help="Weight ECGs by patient count to this power")
    parser.add_argument("--use-tabular", action="store_true", help="Add the 7 routine numbers to the outputs")
    parser.add_argument("--tabular-lr", type=float, default=1e-3, help="Learning rate for the layer that reads them")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200, help="Steps between progress lines")
    parser.add_argument("--limit", type=int, help="Use only this many ECGs per split, for a quick check")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("teacher_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints/teacher/runs") / run_id
    train = EchoNextDataset(args.data_dir, "train", limit=args.limit)
    val = EchoNextDataset(args.data_dir, "val", limit=args.limit)
    started, checkpoint, config_sha256 = time.time(), args.checkpoint, None
    if checkpoint is None:
        record = write_run_record(
            output_dir, run_id=run_id, settings=vars(args), command=sys.argv, data_dir=args.data_dir,
            files={"pretrained": args.pretrained, "probe_head": args.probe_head},
        )
        config_sha256 = record["config_sha256"]
        teacher = build_teacher(
            args.pretrained, device=args.device, head_dropout=args.head_dropout, lead_mask_prob=args.lead_mask_prob,
            use_tabular=args.use_tabular,
        )
        checkpoint = train_teacher(
            teacher, train, val, output_dir, device=args.device, seed=args.seed, probe_head=args.probe_head,
            lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, num_workers=args.num_workers, log_every=args.log_every,
            balance_patients=args.balance_patients, tabular_lr=args.tabular_lr,
        )
    frozen = load_frozen_teacher(checkpoint, pretrained_path=args.pretrained, device=args.device)
    checkpoint_sha256 = sha256_file(checkpoint)
    datasets = {"train": train, "val": val}
    scored = [datasets[name] for name in args.score_splits]
    scored += [EchoNextDataset(args.data_dir, "test")] if args.with_test else []
    for dataset in scored:
        print(write_run(
            frozen, "teacher", dataset, runs_root=args.runs_root, batch_size=args.batch_size,
            num_workers=args.num_workers, device=args.device,
            run_id=run_id, checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256,
        ))
    if args.checkpoint is None:
        write_result(output_dir, {
            "run_id": run_id, **best_epoch(output_dir / "log.csv"), "minutes": round((time.time() - started) / 60, 1),
            "checkpoint_sha256": checkpoint_sha256, "scored": [d.split for d in scored],
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


if __name__ == "__main__":
    main()
