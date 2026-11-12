"""
Save and load the prediction files that every model hands to the next role.

Purpose:
- Run any model that follows the shared call format, logits = model(waveform, tabular).
- Write the same four files for every model, so the audit works on any of them.
- Line saved logits up with dataset batches by row ID, which distillation needs.

Special Notes:
- These files hold patient-level values. Keep them in runs/ and never commit them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import AUDIT_COLUMNS, LABEL_COLUMNS, EchoNextDataset, find_metadata_file

# Each name owns one folder under runs/, so the audit can find every model.
MODEL_NAMES = (
    "mini",
    "gradient_boosting",
    "teacher",
    "student",
    "student_fp16",
    "student_int8_post",
    "student_int8_qat",
)
ARRAY_NAMES = (
    "logits", "probabilities", "labels", "label_mask",
    "row_id", "split_row", "patient_key", "label_columns",
)


@dataclass(frozen=True)
class PredictionRun:
    """One model's saved outputs for one group of ECGs."""

    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame
    manifest: dict
    metrics: dict


def run_dir(model_name: str, split: str, runs_root: str | Path = "runs") -> Path:
    """
    Return the folder that holds one model's files for one group of ECGs.

    Args:
        model_name: One of MODEL_NAMES.
        split: train, val, or test.
        runs_root: Parent folder. Use a separate one, such as runs/dummy, for dummy runs.

    Returns:
        The path runs_root/model_name/split.

    Raises:
        ValueError: When the model name or split is not recognized.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}. Choose one of {MODEL_NAMES}")
    if split not in ("train", "val", "test"):
        raise ValueError("Choose train, val or test; no_split is excluded from experiments")
    return Path(runs_root) / model_name / split


@torch.inference_mode()
def predict(
    model: nn.Module, dataset: EchoNextDataset, *, batch_size: int = 64,
    num_workers: int = 0, device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Run a model over a dataset, keeping the dataset's order.

    Args:
        model: Takes waveform [batch, 12, 2500] and tabular [batch, 7], returns logits [batch, 12].
        dataset: The ECGs to score.
        batch_size: ECGs per step.
        num_workers: Data loader workers.
        device: Where the model runs, such as cpu, cuda, or mps.

    Returns:
        The arrays in ARRAY_NAMES. Missing labels stay NaN.

    Raises:
        ValueError: When the settings are invalid or the model breaks the call format.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    model = model.to(device).eval()
    parts = {key: [] for key in ("logits", "labels", "label_mask", "row_id", "split_row")}
    patient_keys = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        logits = model(batch["waveform"].to(device), batch["tabular"].to(device))
        if tuple(logits.shape) != (len(batch["row_id"]), len(LABEL_COLUMNS)):
            raise ValueError(f"Model must produce [batch,12] logits, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits")
        parts["logits"].append(logits.float().cpu().numpy())
        labels = batch["labels"].numpy().copy()
        mask = batch["label_mask"].numpy()
        labels[~mask] = np.nan
        parts["labels"].append(labels)
        parts["label_mask"].append(mask)
        for key in ("row_id", "split_row"):
            parts[key].append(batch[key].numpy())
        patient_keys.extend(batch["patient_key"])
    outputs = {key: np.concatenate(values) for key, values in parts.items()}
    outputs["probabilities"] = torch.sigmoid(torch.from_numpy(outputs["logits"])).numpy()
    outputs["patient_key"] = np.asarray(patient_keys, dtype=str)
    outputs["label_columns"] = np.asarray(LABEL_COLUMNS, dtype=str)
    return outputs


def basic_metrics(outputs: dict[str, np.ndarray]) -> dict:
    """
    Score each output for a quick check. audit.py owns the full scorecard.

    Args:
        outputs: Arrays with labels, probabilities, and label_mask.

    Returns:
        One entry per label with observed count, positive count, AUROC, and Brier score.
        AUROC is None when only one class is present.
    """
    metrics = {}
    for column, name in enumerate(LABEL_COLUMNS):
        mask = outputs["label_mask"][:, column]
        y = outputs["labels"][mask, column]
        p = outputs["probabilities"][mask, column]
        metrics[name] = {
            "n_observed": int(len(y)), "n_positive": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
        }
    return metrics


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_manifest(model_name: str, dataset: EchoNextDataset, *, data_version: str = "1.1.1", **settings) -> dict:
    """
    Record what produced a run, so a result can be traced and repeated.

    Args:
        model_name: Name stored in the manifest.
        dataset: The ECGs that were scored.
        data_version: PhysioNet version of the data.
        **settings: Other values to record, such as batch_size and device.

    Returns:
        A dictionary that can be written as JSON.
    """
    return {
        "model": model_name,
        "code": _code_version(),
        "data_version": data_version,
        "data_dir": str(dataset.root),
        "metadata_sha256": sha256_file(find_metadata_file(dataset.root)),
        "split": dataset.split,
        "n_records": len(dataset),
        "label_columns": list(LABEL_COLUMNS),
        "sampling_rate_hz": 250,
        "preprocessing": "official preprocessed arrays; no additional normalization",
        "waveform_file": (
            None if dataset.waveform_path is None else {
                "name": dataset.waveform_path.name,
                "shape": list(dataset._waveforms.shape),
                "bytes": dataset.waveform_path.stat().st_size,
            }
        ),
        "tabular_file": {"name": dataset.tabular_path.name, "shape": list(dataset._tabular.shape),
                         "bytes": dataset.tabular_path.stat().st_size},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        **settings,
    }


def fresh_run_dir(output_dir: str | Path, data_root: Path) -> Path:
    """
    Check that a run folder is safe to write to.

    Args:
        output_dir: Where the run will be saved.
        data_root: The dataset folder, which must stay unchanged.

    Returns:
        The resolved folder path. The folder is not created here.

    Raises:
        FileExistsError: When the folder already holds files, which would mix runs.
        ValueError: When the folder is inside the dataset folder.
    """
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    if destination == data_root or data_root in destination.parents:
        raise ValueError("Store outputs outside the source dataset directory")
    return destination


def save_run(
    destination: Path, outputs: dict[str, np.ndarray], selected: pd.DataFrame, manifest: dict,
    *, write_artifacts: Callable[[Path], None] | None = None,
) -> dict:
    """
    Write the four run files.

    Args:
        destination: Folder from fresh_run_dir.
        outputs: The arrays in ARRAY_NAMES.
        selected: The scored rows from dataset.selected_metadata, in the same order.
        manifest: Output of run_manifest, plus any model-specific values.
        write_artifacts: Optional writer for additional private files in the run.

    Returns:
        The basic metrics that were saved.

    Raises:
        ValueError: When an array is missing or the rows do not line up.
    """
    missing = set(ARRAY_NAMES) - set(outputs)
    if missing:
        raise ValueError(f"Run is missing arrays: {sorted(missing)}")
    if not np.array_equal(outputs["row_id"], selected["row_id"].to_numpy()):
        raise ValueError("Prediction rows do not line up with the metadata rows")
    metrics = basic_metrics(outputs)
    columns = ["row_id", "split_row", "patient_key", "split"]
    columns += [column for column in AUDIT_COLUMNS if column in selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        np.savez_compressed(staging / "predictions.npz", **outputs)
        selected[columns].to_csv(staging / "metadata.csv", index=False)
        for filename, value in (("manifest.json", manifest), ("metrics.json", metrics)):
            (staging / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if write_artifacts is not None:
            write_artifacts(staging)
        # Commit the whole run together. An already completed run cannot be replaced.
        try:
            staging.rename(destination)
        except OSError as error:
            if destination.is_dir() and any(destination.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {destination}") from error
            raise
    return metrics


def write_run(
    model: nn.Module, model_name: str, dataset: EchoNextDataset, *,
    runs_root: str | Path = "runs", batch_size: int = 64, num_workers: int = 0,
    device: str = "cpu", **manifest_values,
) -> Path:
    """
    Score a dataset with a model and save the run in its standard folder.

    Args:
        model: Follows the shared call format.
        model_name: One of MODEL_NAMES.
        dataset: The ECGs to score.
        runs_root: Parent folder for runs.
        batch_size: ECGs per step.
        num_workers: Data loader workers.
        device: Where the model runs.
        **manifest_values: Extra values to record, such as a checkpoint hash.

    Returns:
        The folder that was written.
    """
    destination = fresh_run_dir(run_dir(model_name, dataset.split, runs_root), dataset.root)
    outputs = predict(model, dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    manifest = run_manifest(
        model_name, dataset, batch_size=batch_size, num_workers=num_workers, device=device, **manifest_values,
    )
    save_run(destination, outputs, dataset.selected_metadata, manifest)
    return destination


def load_run(path: str | Path) -> PredictionRun:
    """
    Load and check a saved run.

    Args:
        path: A folder written by save_run or write_run.

    Returns:
        The saved arrays, audit metadata, manifest, and metrics.

    Raises:
        ValueError: When arrays are missing, have the wrong shape, or do not line up.
    """
    folder = Path(path)
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    missing = set(ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError(f"{folder} is missing arrays: {sorted(missing)}")
    count = len(arrays["row_id"])
    for name in ("logits", "probabilities", "labels", "label_mask"):
        if arrays[name].shape != (count, len(LABEL_COLUMNS)):
            raise ValueError(f"{folder}: {name} must be [{count},12], got {arrays[name].shape}")
    if tuple(arrays["label_columns"]) != LABEL_COLUMNS:
        raise ValueError(f"{folder}: outputs are not in the shared label order")
    metadata = pd.read_csv(folder / "metadata.csv", dtype={"patient_key": "string"})
    if not np.array_equal(arrays["row_id"], metadata["row_id"].to_numpy()):
        raise ValueError(f"{folder}: predictions and metadata.csv do not line up")
    return PredictionRun(
        arrays=arrays,
        metadata=metadata,
        manifest=json.loads((folder / "manifest.json").read_text()),
        metrics=json.loads((folder / "metrics.json").read_text()),
    )


def aligned_logits(run: PredictionRun, row_ids: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Look up saved logits for a batch, which may be shuffled.

    Args:
        run: A saved run, such as the Teacher's on the Training ECGs.
        row_ids: The batch's row_id values.

    Returns:
        Logits [batch, 12] in the same order as row_ids.

    Raises:
        ValueError: When the run has no prediction for one of the rows.
    """
    wanted = np.asarray(row_ids)
    order = np.argsort(run.arrays["row_id"])
    sorted_ids = run.arrays["row_id"][order]
    positions = np.searchsorted(sorted_ids, wanted).clip(max=len(sorted_ids) - 1)
    if not np.array_equal(sorted_ids[positions], wanted):
        raise ValueError("The saved run has no prediction for at least one row in this batch")
    return torch.from_numpy(run.arrays["logits"][order[positions]])
