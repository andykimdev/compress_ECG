"""
Plan a sweep of Teacher runs and choose idle GPUs for them.

Purpose:
- Expand a grid config into one settings dictionary per run.
- Build the command line for one run.
- Pick GPUs that no one is using, so a sweep never lands on another job's card.

Special Notes:
- Everything here is pure, so it is tested without GPUs. scripts/sweep_teacher.py does the I/O.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Settings that become command line flags of scripts/run_teacher.py.
FLAGS = {
    "lr": "--lr", "head_lr": "--head-lr", "seed": "--seed", "batch_size": "--batch-size",
    "max_epochs": "--max-epochs", "patience": "--patience", "num_workers": "--num-workers",
    "probe_head": "--probe-head", "log_every": "--log-every",
}


def run_id_for(settings: dict) -> str:
    """Name a run from its two learning rates and its seed."""
    return f"lr{settings['lr']:g}_hlr{settings['head_lr']:g}_s{settings['seed']}"


def expand_runs(config: dict) -> list[dict]:
    """
    Turn a sweep config into one settings dictionary per run.

    Args:
        config: Has base (settings every run shares), grid (lists to combine), and optional
            extra_runs (settings that override base for single extra runs, such as a second seed).

    Returns:
        Settings in grid order, then the extra runs. A run that repeats earlier settings is dropped.

    Raises:
        ValueError: When a run has no lr, head_lr, or seed.
    """
    grid = config.get("grid", {})
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] if grid else [{}]
    runs, seen = [], set()
    for override in combos + list(config.get("extra_runs", [])):
        settings = {**config["base"], **override}
        missing = {"lr", "head_lr", "seed"} - settings.keys()
        if missing:
            raise ValueError(f"Every run needs {sorted(missing)}")
        key = tuple(sorted(settings.items()))
        if key not in seen:
            seen.add(key)
            runs.append(settings)
    return runs


def build_command(settings: dict, *, run_id: str, runs_root: str, python: str = sys.executable) -> list[str]:
    """
    Build the command for one run.

    Args:
        settings: One entry of expand_runs.
        run_id: Name of the run. It also names the run folder and the prediction folder.
        runs_root: Parent folder for this sweep's prediction files.
        python: Interpreter that runs the script.

    Returns:
        The argument list for subprocess. Only the Validation ECGs are scored, to save time.
    """
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_teacher.py"
    command = [python, str(script), "--device", "cuda", "--run-id", run_id,
               "--runs-root", f"{runs_root}/{run_id}", "--score-splits", "val"]
    for name, flag in FLAGS.items():
        if settings.get(name) is not None:
            command += [flag, str(settings[name])]
    return command


def free_gpu_indices(
    gpus: list[dict], busy_bus_ids: set[str], *, max_used_mib: int = 1000, max_util: int = 10,
) -> list[int]:
    """
    List the GPUs that no one is using.

    Args:
        gpus: One dictionary per GPU with index, bus_id, used_mib, and util, in percent.
        busy_bus_ids: Bus IDs of every GPU that has a compute process, from any user.
        max_used_mib: A GPU using more memory than this is busy.
        max_util: A GPU busier than this, in percent, is busy.

    Returns:
        The index of each GPU with no process, little memory in use, and low load.
    """
    return [
        gpu["index"] for gpu in gpus
        if gpu["bus_id"] not in busy_bus_ids and gpu["used_mib"] <= max_used_mib and gpu["util"] <= max_util
    ]
