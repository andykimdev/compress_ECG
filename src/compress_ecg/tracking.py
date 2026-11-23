"""
Send a Student run's settings and epoch scores to Weights & Biases.

Purpose:
- Turn the saved training settings into a flat W&B config.
- Log each epoch's record and the run's final scores.
- Turn a finished Teacher run's saved files into W&B config and history, for the import.

Special Notes:
- Only aggregate numbers leave the machine. No waveform, label, or prediction of a patient
  is ever logged, as the data use agreement requires.
- wandb is imported only when a run starts, so machines without it still train.
"""

from __future__ import annotations

import re
from pathlib import Path

# Scores in a run's result.json that are its best, not its last, and get a best_ name in W&B.
BEST_SCORES = ("val_mean_auroc", "val_shd_auroc")
# Manifest fields that identify which Teacher a Student copied.
TEACHER_FIELDS = {
    "teacher_model": "model", "teacher_run_id": "run_id", "teacher_split": "split",
    "teacher_checkpoint_sha256": "checkpoint_sha256", "teacher_records": "n_records",
}


def wandb_config(settings: dict) -> dict:
    """
    Flatten training settings for the W&B config.

    Args:
        settings: The contents of training_config.json, with the Teacher's manifest.

    Returns:
        Every setting except the manifest, plus the fields in TEACHER_FIELDS, which are
        None when the manifest lacks them.
    """
    manifest = settings.get("teacher_manifest", {})
    config = {name: value for name, value in settings.items() if name != "teacher_manifest"}
    config.update({name: manifest.get(field) for name, field in TEACHER_FIELDS.items()})
    return config


def teacher_group(run_id: str) -> str:
    """Name the group of a Teacher run: its run ID without the trailing seed, such as _s42."""
    return re.sub(r"_s\d+$", "", run_id)


def teacher_history(rows: list[dict[str, str]]) -> list[dict]:
    """
    Turn the rows of a Teacher run's log.csv into W&B history entries.

    Args:
        rows: One dictionary of text values per logged evaluation, as csv.DictReader gives them.

    Returns:
        One entry per row with every value as a number. Blank values are left out, and the
        step column becomes train_step, because W&B keeps step for its own counter.
    """
    return [
        {("train_step" if name == "step" else name): float(value) for name, value in row.items() if value != ""}
        for row in rows
    ]


def teacher_config(config: dict) -> dict:
    """Flatten a Teacher run's config.json into its settings, start time, code commit, and GPU."""
    return {
        **config["settings"], "started_at": config.get("started_at"),
        "code_commit": config.get("code", {}).get("commit"), "gpu": config.get("machine", {}).get("gpu"),
    }


def teacher_summary(result: dict) -> dict:
    """
    Turn a Teacher run's result.json into its W&B summary.

    Args:
        result: The saved result, with the scores of the picked checkpoint.

    Returns:
        Every number in the result. The scores in BEST_SCORES are renamed with a best_
        prefix, because W&B would otherwise show the last logged value under the same name.
    """
    return {
        (f"best_{name}" if name in BEST_SCORES else name): value for name, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def write_summary(run_path: str, summary: dict) -> None:
    """
    Store final scores on a finished W&B run through the public API.

    Args:
        run_path: entity/project/run_id of the run.
        summary: Names and values to add to the run summary.

    Notes:
        Writing after the run is closed means its last logged epoch cannot replace these
        values. The project's run listing can show a cached summary for a while, so check
        a run by fetching it on its own.
    """
    import wandb

    run = wandb.Api().run(run_path)
    run.summary.update(summary)


def epoch_log(record: dict) -> dict:
    """Return an epoch record without its epoch number, which W&B takes as the step, or empty values."""
    return {name: value for name, value in record.items() if name != "epoch" and value is not None}


class WandbRun:
    """One W&B run that receives epoch records and final scores."""

    def __init__(
        self, *, project: str, entity: str | None, name: str, group: str | None, mode: str,
        settings: dict, directory: str | Path, tags: list[str] | None = None,
    ):
        import wandb

        Path(directory).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project, entity=entity, name=name, group=group, mode=mode, tags=tags,
            config=wandb_config(settings), dir=str(directory),
        )

    def log_epoch(self, record: dict) -> None:
        """Log one epoch's record at its epoch number."""
        self.run.log(epoch_log(record), step=record["epoch"])

    def finish(self, summary: dict) -> None:
        """Close the run, then store the final scores in its summary."""
        path = f"{self.run.entity}/{self.run.project}/{self.run.id}"
        self.run.finish()
        write_summary(path, summary)
