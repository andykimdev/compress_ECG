#!/usr/bin/env python3
"""Role B: fine-tune the Teacher, or load a saved one, and save its predictions. Install first: pip install -e ."""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from compress_ecg.data import EchoNextDataset, open_group
from compress_ecg.predictions import sha256_file, write_run
from compress_ecg.runs import write_result, write_run_record
from compress_ecg.teacher import (
    SELECT_COLUMNS, build_teacher, check_probe_head, load_frozen_teacher, train_teacher,
)


def best_epoch(log_path: Path, select_by: str = "mean") -> dict:
    """
    Summarize a training log for result.json.

    Args:
        log_path: The log.csv that train_teacher wrote.
        select_by: The score that picked the checkpoint, a key of SELECT_COLUMNS.

    Returns:
        The epoch and both scores of the row with the best such score, where the
        latest row wins a tie, and the number of epochs that ran.
    """
    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    best = max(range(len(rows)), key=lambda i: (float(rows[i][SELECT_COLUMNS[select_by]]), i))
    return {
        # With several scorings per epoch the log has several rows per epoch, so rows are not counted.
        "best_epoch": int(rows[best]["epoch"]), "epochs_run": int(rows[-1]["epoch"]),
        "val_mean_auroc": float(rows[best]["val_mean_auroc"]), "val_shd_auroc": float(rows[best]["val_shd_auroc"]),
    }


def main(argv: list[str] | None = None) -> None:
    """
    Train or load a Teacher and save its predictions.

    Args:
        argv: Command line arguments without the program name. None reads them from sys.argv.

    Notes:
        Without a holdout, training uses the train split and the checkpoint is picked on
        Validation. With --holdout-fraction above 0, training uses the fit part of train and
        the checkpoint is picked on the dev part, so Validation is read only when
        --score-splits names it. result.json records which one in scored_on.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--pretrained", default="checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--run-id", help="Name of the run. Defaults to the start time")
    parser.add_argument("--output-dir", help="Run folder. Defaults to checkpoints/teacher/runs/<run id>")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--score-splits", nargs="+", choices=("train", "val", "dev"),
                        help="ECGs to save predictions for. dev is the held-out part of train and needs "
                             "--holdout-fraction. Defaults to train and val, or to dev alone with a holdout")
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    parser.add_argument("--checkpoint", help="Score this saved Teacher and skip training")
    parser.add_argument("--probe-head",
                        help="Output layer from run_teacher_probe.py to start from. With a holdout it must come "
                             "from a probe run with the same holdout, in a folder such as probe_ho0.2_hs0")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for ECG-FM")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for the output layer")
    parser.add_argument("--head-dropout", type=float, default=0.0, help="Dropout before the output layer")
    parser.add_argument("--lead-mask-prob", type=float, default=0.0, help="Chance each lead is blanked in training")
    parser.add_argument("--balance-patients", type=float, default=0.0, help="Weight ECGs by patient count to this power")
    parser.add_argument("--use-tabular", action="store_true", help="Add the 7 routine numbers to the outputs")
    parser.add_argument("--tabular-lr", type=float, default=1e-3, help="Learning rate for the layer that reads them")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Score and save a running average of the weights in place of the trained weights. "
                             "Each update keeps a share of the old average and takes the rest from the new "
                             "weights. The share is (1 + n) / (10 + n) after n updates and stops rising at this "
                             "value. So the average covers roughly the most recent ninth of the updates so far, "
                             "and 0.999 is only reached after 8,990 updates, more than a typical run takes. "
                             "0 is off")
    parser.add_argument("--evals-per-epoch", type=int, default=1,
                        help="Validation scorings per epoch, evenly spaced. Each adds a row to log.csv")
    parser.add_argument("--select-by", choices=tuple(SELECT_COLUMNS), default="mean",
                        help="Validation score that picks the checkpoint: the 12-output mean AUROC or SHD AUROC")
    parser.add_argument("--holdout-fraction", type=float, default=0.0,
                        help="Hold out this share of the Training patients to judge pilot runs. Training then "
                             "uses only the other patients, and the checkpoint is picked and scored on the most "
                             "recent ECG of each held-out patient (dev) in place of Validation, which is then "
                             "not read unless --score-splits names it. The log keeps its val_ column names, "
                             "and result.json says scored_on dev. 0 is off")
    parser.add_argument("--holdout-seed", type=int, default=0, help="Seed of the draw of held-out patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2,
                        help="Validation scorings without a better score before stopping. Epochs, at one per epoch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200, help="Steps between progress lines")
    parser.add_argument("--limit", type=int, help="Use only this many ECGs per split, for a quick check")
    args = parser.parse_args(argv)
    if not 0 <= args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be at least 0 and below 1")
    holdout = args.holdout_fraction > 0
    if args.score_splits is None:
        args.score_splits = ["dev"] if holdout else ["train", "val"]
    if "dev" in args.score_splits and not holdout:
        parser.error("--score-splits dev needs --holdout-fraction above 0")
    # The ECGs that train the model, and the ECGs that pick and score the checkpoint.
    trained_on, scored_on = ("fit", "dev") if holdout else ("train", "val")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("teacher_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints/teacher/runs") / run_id
    # Only the ECGs this run needs are opened, so a holdout run leaves Validation unread.
    needed = args.score_splits + ([trained_on, scored_on] if args.checkpoint is None else [])
    datasets = {
        name: open_group(
            args.data_dir, name, limit=args.limit, holdout_fraction=args.holdout_fraction,
            holdout_seed=args.holdout_seed,
        )
        for name in dict.fromkeys(needed)
    }
    started, checkpoint, config_sha256 = time.time(), args.checkpoint, None
    if checkpoint is None:
        check_probe_head(args.probe_head, args.holdout_fraction, args.holdout_seed)
        record = write_run_record(
            output_dir, run_id=run_id, settings=vars(args), command=sys.argv if argv is None else [__file__, *argv],
            data_dir=args.data_dir, files={"pretrained": args.pretrained, "probe_head": args.probe_head},
        )
        config_sha256 = record["config_sha256"]
        # The new output layer draws its starting weights here, before train_teacher sets its seed.
        torch.manual_seed(args.seed)
        teacher = build_teacher(
            args.pretrained, device=args.device, head_dropout=args.head_dropout, lead_mask_prob=args.lead_mask_prob,
            use_tabular=args.use_tabular,
        )
        checkpoint = train_teacher(
            teacher, datasets[trained_on], datasets[scored_on], output_dir, device=args.device, seed=args.seed,
            probe_head=args.probe_head,
            lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, num_workers=args.num_workers, log_every=args.log_every,
            balance_patients=args.balance_patients, tabular_lr=args.tabular_lr, ema_decay=args.ema_decay,
            evals_per_epoch=args.evals_per_epoch, select_by=args.select_by,
        )
    frozen = load_frozen_teacher(checkpoint, pretrained_path=args.pretrained, device=args.device)
    checkpoint_sha256 = sha256_file(checkpoint)
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
            "run_id": run_id, "scored_on": scored_on, **best_epoch(output_dir / "log.csv", args.select_by),
            "minutes": round((time.time() - started) / 60, 1),
            "checkpoint_sha256": checkpoint_sha256, "scored": [d.group for d in scored],
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


if __name__ == "__main__":
    main()
