#!/usr/bin/env python3
"""
Copy the finished Teacher runs into W&B, so one dashboard shows the Teacher and the Student.

Purpose:
- Log each fine-tuning run's settings, its per-evaluation history from log.csv, and its
  final scores from result.json.
- Log each weight-averaged Teacher from its record file as a run with final scores only.

Special Notes:
- Only aggregate scores leave the machine. Checkpoints and predictions are never uploaded.
- A run whose name already exists in the project is not logged again, but its final
  scores are written again, so the import can be rerun to repair summaries.
- Install first: pip install -e . wandb
"""

import argparse
import csv
import json
from pathlib import Path

from compress_ecg.tracking import teacher_config, teacher_group, teacher_history, teacher_summary, write_summary

# Record files of the weight-averaged Teachers, which have no training log of their own.
SOUP_RECORDS = ("results/teacher/best_teacher.json", "results/teacher/soup_check_s48to53.json")
# Fields of a soup record that become its final scores.
SOUP_SCORES = ("val_shd_auroc", "val_mean_auroc", "val_ecgs")


def finished_runs(runs_dir: Path) -> list[Path]:
    """Return the run folders that hold a log, a config, and a result, in name order."""
    return sorted(
        folder for folder in runs_dir.iterdir()
        if all((folder / name).is_file() for name in ("log.csv", "config.json", "result.json"))
    )


def soup_summary(record: dict) -> dict:
    """Return a weight-averaged Teacher's final scores, named as best_ scores like the fine-tuning runs."""
    return {(f"best_{name}" if name.startswith("val_") and name != "val_ecgs" else name): record[name]
            for name in SOUP_SCORES if name in record}


def import_run(wandb, folder: Path, *, project: str, entity: str) -> str:
    """Log one fine-tuning run's settings and every logged evaluation, and return its run path."""
    config = json.loads((folder / "config.json").read_text())
    with (folder / "log.csv").open() as handle:
        history = teacher_history(list(csv.DictReader(handle)))
    run = wandb.init(
        project=project, entity=entity, name=folder.name, group=f"teacher_{teacher_group(folder.name)}",
        job_type="teacher-finetune", tags=["teacher"], config=teacher_config(config), reinit="finish_previous",
    )
    for index, entry in enumerate(history, start=1):
        run.log(entry, step=index)
    path = f"{run.entity}/{run.project}/{run.id}"
    run.finish()
    return path


def import_soup(wandb, record: dict, *, project: str, entity: str) -> str:
    """Log one weight-averaged Teacher as a run with its source runs, and return its run path."""
    run = wandb.init(
        project=project, entity=entity, name=record["name"], group="teacher_weight_average",
        job_type="teacher-weight-average", tags=["teacher", "weight-average"], reinit="finish_previous",
        config={
            "source_runs": record["source_runs"], "code_commit": record.get("code_commit"),
            "checkpoint_sha256": record.get("checkpoint_sha256", record.get("sha256")),
            "description": record.get("description"),
        },
    )
    path = f"{run.entity}/{run.project}/{run.id}"
    run.finish()
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="checkpoints/teacher/runs")
    parser.add_argument("--project", default="compress-ecg")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--dry-run", action="store_true", help="List what would be imported and log nothing")
    args = parser.parse_args(argv)

    import wandb

    folders = finished_runs(Path(args.runs_dir))
    soups = [json.loads(Path(path).read_text()) for path in SOUP_RECORDS]
    existing = {run.name: "/".join(run.path) for run in wandb.Api().runs(f"{args.entity}/{args.project}")}
    pending = [folder for folder in folders if folder.name not in existing]
    pending_soups = [record for record in soups if record["name"] not in existing]
    print(f"{len(folders)} fine-tuning runs, {len(pending)} not yet in W&B. "
          f"{len(soups)} weight averages, {len(pending_soups)} not yet in W&B")
    if args.dry_run:
        for folder in pending:
            print("would import", folder.name, "into group", f"teacher_{teacher_group(folder.name)}")
        for record in pending_soups:
            print("would import", record["name"])
        return
    for folder in folders:
        path = existing.get(folder.name) or import_run(wandb, folder, project=args.project, entity=args.entity)
        write_summary(path, teacher_summary(json.loads((folder / "result.json").read_text())))
        print("final scores written for", folder.name)
    for record in soups:
        path = existing.get(record["name"]) or import_soup(wandb, record, project=args.project, entity=args.entity)
        write_summary(path, soup_summary(record))
        print("final scores written for", record["name"])


if __name__ == "__main__":
    main()
