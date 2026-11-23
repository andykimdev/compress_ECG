#!/usr/bin/env python3
"""
Log the saved evaluation, timing, and profiling results to W&B as summary runs.

Purpose:
- Read the result files behind every number in the slides and report.
- Log one run per model or measurement in the evaluation group, with final scores only.

Special Notes:
- Only aggregate numbers leave the machine. No checkpoint, prediction, or patient value is sent.
- A run whose name already exists in the project is not logged again, but its final scores
  are written again, so the script can be rerun to update them.
- Install first: pip install -e . wandb
"""

import argparse
import csv
import json
from pathlib import Path

from compress_ecg.evaluation_log import STUDENT_ARMS, build_runs
from compress_ecg.tracking import write_summary


def read_json(path: Path) -> dict:
    """Return a JSON file's contents."""
    return json.loads(path.read_text())


def csv_metrics(path: Path) -> dict:
    """Return a per-label metrics CSV with output, auroc, and brier columns as a label to scores dict."""
    with path.open() as handle:
        return {row["output"]: {"auroc": float(row["auroc"]), "brier": float(row["brier"])}
                for row in csv.DictReader(handle)}


def seed_metrics(student_dir: Path, prefix: str) -> dict:
    """Return each seed's val_metrics.json for one Student arm, keyed by seed folder name."""
    folders = sorted(student_dir.glob(f"{prefix}_s[0-9]*"))
    if not folders:
        raise FileNotFoundError(f"No {prefix}_s<seed> folders in {student_dir}")
    return {folder.name: read_json(folder / "val_metrics.json") for folder in folders}


def load_results(results_dir: Path) -> dict:
    """Read every result file that build_runs needs from the repository's results folder."""
    baselines, teacher, student = results_dir / "baselines", results_dir / "teacher", results_dir / "student"
    profiling = student / "profiling"
    return {
        "mini_val": read_json(baselines / "mini_val_metrics.json")["metrics"],
        "mini_test": read_json(baselines / "mini_test_metrics.json")["metrics"],
        "logistic_val": read_json(baselines / "logistic_regression_val_metrics.json")["metrics"],
        "boosting_val": read_json(baselines / "gradient_boosting_val_metrics.json")["metrics"],
        "boosting_test": read_json(baselines / "gradient_boosting_test_metrics.json")["metrics"],
        "teacher_val": csv_metrics(teacher / "best_teacher_val.csv"),
        "teacher_gap": read_json(teacher / "val_gap_teacher_vs_mini.json"),
        "teacher_speed": read_json(teacher / "teacher_speed_gpu2.json"),
        "student_speed": read_json(student / "student_speed_gpu2.json"),
        "fp16": read_json(student / "fp16_vs_fp32_val.json"),
        "students": {arm: seed_metrics(student, prefix) for arm, prefix in STUDENT_ARMS.items()},
        "profiles_before": [read_json(path) for path in sorted(profiling.glob("steps_before_r*.json"))],
        "profiles_after": [read_json(path) for path in sorted(profiling.glob("steps_after_r*.json"))],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--project", default="compress-ecg")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print each run's scores and log nothing")
    args = parser.parse_args(argv)

    runs = build_runs(load_results(Path(args.results_dir)))
    if args.dry_run:
        for spec in runs:
            print(json.dumps({"name": spec["name"], "summary": spec["summary"]}, indent=1))
        return

    import wandb

    existing = {run.name: "/".join(run.path) for run in wandb.Api().runs(f"{args.entity}/{args.project}")}
    for spec in runs:
        path = existing.get(spec["name"])
        if path is None:
            run = wandb.init(
                project=args.project, entity=args.entity, name=spec["name"], group=spec["group"],
                job_type=spec["job_type"], tags=["evaluation"], config=spec["config"], reinit="finish_previous",
            )
            path = f"{run.entity}/{run.project}/{run.id}"
            run.finish()
        write_summary(path, spec["summary"])
        print("final scores written for", spec["name"])


if __name__ == "__main__":
    main()
