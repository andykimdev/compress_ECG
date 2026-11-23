#!/usr/bin/env python3
"""Role C: train the Student from saved Teacher logits. Install first: pip install -e ."""

import argparse
import inspect
import json
from pathlib import Path

import torch

from compress_ecg.data import EchoNextDataset
from compress_ecg.distill import train_student
from compress_ecg.predictions import load_run, write_run
from compress_ecg.student import build_student, load_student

# Settings a config file may pass to train_student. Seed and device come from the command line.
CONFIG_SETTINGS = {
    "batch_size", "learning_rate", "weight_decay", "max_epochs", "patience", "min_delta",
    "temperature", "hard_label_weight", "num_workers",
}


def read_settings(path: str | None) -> dict:
    """
    Read training settings from a JSON config.

    Args:
        path: The config file, or None for the defaults of train_student.

    Returns:
        The settings to pass to train_student.

    Raises:
        ValueError: When the config holds a setting train_student does not take.
    """
    if path is None:
        return {}
    settings = json.loads(Path(path).read_text())
    unknown = set(settings) - CONFIG_SETTINGS
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}. Allowed: {sorted(CONFIG_SETTINGS)}")
    return settings


def effective_settings(settings: dict) -> dict:
    """Return every config setting as training will use it: the given values, then the defaults of train_student."""
    defaults = inspect.signature(train_student).parameters
    return {name: settings.get(name, defaults[name].default) for name in sorted(CONFIG_SETTINGS, key=list(defaults).index)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--teacher-run", default="runs/teacher/train",
                        help="In Phase 1, use EchoNext-Mini's saved run on the Training ECGs")
    parser.add_argument("--config", help="JSON with training settings, such as temperature and hard_label_weight")
    parser.add_argument("--output-dir", default="checkpoints/student")
    parser.add_argument("--runs-root", default="runs", help="In Phase 1, use runs/dummy")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--seed", type=int, default=42, help="Sets the starting weights and the training order")
    parser.add_argument("--with-test", action="store_true", help="Also score Test, in Phase 2 only")
    parser.add_argument("--wandb-project", help="Log to this W&B project. Without it nothing is logged")
    parser.add_argument("--wandb-entity", help="W&B team or user that owns the project")
    parser.add_argument("--wandb-group", help="Groups runs on the dashboard, such as all seeds of one setting")
    parser.add_argument("--wandb-name", help="Run name on the dashboard. Defaults to the output folder name")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"],
                        help="offline saves the run locally, to upload later with wandb sync")
    args = parser.parse_args(argv)

    settings = read_settings(args.config)
    train = EchoNextDataset(args.data_dir, "train")
    val = EchoNextDataset(args.data_dir, "val")
    teacher_run = load_run(args.teacher_run)
    tracker = None
    if args.wandb_project:
        from compress_ecg.tracking import WandbRun

        tracker = WandbRun(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            name=args.wandb_name or Path(args.output_dir).name, mode=args.wandb_mode,
            settings={**effective_settings(settings), "seed": args.seed, "device": args.device, "teacher_manifest": teacher_run.manifest},
            directory=Path(args.output_dir) / "wandb", tags=["student"],
        )
    # Seed before building, so the starting weights repeat as well as the training order.
    torch.manual_seed(args.seed)
    student = build_student(device=args.device)
    checkpoint = train_student(
        student, teacher_run, train, val, args.output_dir, device=args.device, seed=args.seed,
        on_epoch=tracker.log_epoch if tracker else None, **settings,
    )
    trained = load_student(checkpoint, device=args.device)
    datasets = [val] + ([EchoNextDataset(args.data_dir, "test")] if args.with_test else [])
    for dataset in datasets:
        print(write_run(trained, "student", dataset, runs_root=args.runs_root, device=args.device))
    if tracker:
        history = json.loads((Path(args.output_dir) / "history.json").read_text())
        best = max(history, key=lambda record: record["val_shd_auc"])
        tracker.finish({
            "best_val_shd_auroc": best["val_shd_auc"], "best_epoch": best["epoch"],
            "best_val_mean_auroc": best["val_mean_auroc"], "epochs_run": len(history),
            "checkpoint_bytes": Path(checkpoint).stat().st_size,
            "parameters": sum(parameter.numel() for parameter in trained.parameters()),
        })


if __name__ == "__main__":
    main()
