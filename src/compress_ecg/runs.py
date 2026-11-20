"""
Record what produced each training run, so results can be traced and compared.

Purpose:
- Write config.json when a run starts and result.json when it ends, inside the run folder.
- Collect every run folder into one table of settings and aggregate scores.

Special Notes:
- Records hold settings, hashes, and aggregate scores only, never patient values.
- The run folder is the source of truth. The table is rebuilt from it at any time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import find_metadata_file, resolve_data_dir
from .predictions import sha256_file

CONFIG_NAME = "config.json"
RESULT_NAME = "result.json"
# Settings shown as table columns. Every setting stays in config.json.
TABLE_SETTINGS = (
    "lr", "head_lr", "head_dropout", "lead_mask_prob", "balance_patients", "use_tabular", "ema_decay",
    "evals_per_epoch", "select_by", "seed", "batch_size", "max_epochs", "patience",
)


def code_version() -> dict:
    """Return the git commit and whether the checkout has uncommitted changes."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def settings_hash(settings: dict) -> str:
    """Return a short fingerprint of a settings dictionary, identical for identical settings."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def write_run_record(
    output_dir: str | Path, *, run_id: str, settings: dict, command: list[str],
    files: dict[str, str | Path | None], data_dir: str | Path | None = None,
) -> dict:
    """
    Write config.json for a run that is about to start.

    Args:
        output_dir: The run folder. It must not already hold a record.
        run_id: Name that links this folder to its prediction files.
        settings: Every setting that changes the result, such as learning rates and the seed.
        command: The command line, so the run can be repeated.
        files: Input files to hash, by name. A value of None is skipped.
        data_dir: Folder with the EchoNext files, or None for the default.

    Returns:
        The record that was written, including config_sha256.

    Raises:
        FileExistsError: When the folder already holds a record. Choose a new folder.
    """
    output_dir = Path(output_dir)
    path = output_dir / CONFIG_NAME
    if path.exists():
        raise FileExistsError(f"{path} already exists. Choose a new run folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = resolve_data_dir(data_dir)
    record = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "config_sha256": settings_hash(settings),
        "command": command,
        "code": code_version(),
        "files": {
            name: {"path": str(file), "sha256": sha256_file(file)} for name, file in files.items() if file is not None
        },
        "data": {"dir": str(root), "metadata_sha256": sha256_file(find_metadata_file(root))},
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "machine": {
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        },
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_result(output_dir: str | Path, result: dict) -> Path:
    """Write result.json for a run that finished, and return its path."""
    path = Path(output_dir) / RESULT_NAME
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def collect_runs(root: str | Path) -> pd.DataFrame:
    """
    Build one table from every run folder under root.

    Args:
        root: Folder whose subfolders each hold a config.json.

    Returns:
        One row per run with its settings, code version, status, and aggregate scores.
        Runs without a result.json have status incomplete. Rows are sorted by Validation
        mean AUROC, best first, with incomplete runs last.
    """
    rows = []
    for config_path in sorted(Path(root).glob(f"*/{CONFIG_NAME}")):
        config = json.loads(config_path.read_text())
        result_path = config_path.parent / RESULT_NAME
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        rows.append({
            "run_id": config["run_id"],
            "status": "finished" if result else "incomplete",
            **{name: config["settings"].get(name) for name in TABLE_SETTINGS},
            "commit": (config["code"]["commit"] or "")[:7],
            "dirty": config["code"]["dirty"],
            "gpu": config["machine"]["gpu"],
            "best_epoch": result.get("best_epoch"),
            "epochs_run": result.get("epochs_run"),
            "val_mean_auroc": result.get("val_mean_auroc"),
            "val_shd_auroc": result.get("val_shd_auroc"),
            "minutes": result.get("minutes"),
            "checkpoint_sha256": (result.get("checkpoint_sha256") or "")[:12],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("val_mean_auroc", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


def round_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Round the scores and run time for display, and leave learning rates exact."""
    return table.round({"val_mean_auroc": 4, "val_shd_auroc": 4, "minutes": 1})
