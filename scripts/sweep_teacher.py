#!/usr/bin/env python3
"""Run a sweep of Teacher fine-tunes, each on its own idle GPU. Install first: pip install -e ."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from compress_ecg.runs import CONFIG_NAME, collect_runs, round_scores
from compress_ecg.sweep import build_command, expand_runs, free_gpu_indices, run_id_for
from compress_ecg.teacher import check_probe_head

RUNS_DIR = Path("checkpoints/teacher/runs")


def read_gpus() -> tuple[list[dict], set[str]]:
    """Read every GPU's memory and load, and the bus IDs of GPUs with a compute process."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"], text=True)
    gpus = []
    for line in rows.strip().splitlines():
        index, bus_id, used, util = (field.strip() for field in line.split(","))
        gpus.append({"index": int(index), "bus_id": bus_id, "used_mib": int(used), "util": int(util)})
    return gpus, {line.strip() for line in apps.splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> None:
    """
    Plan the runs of a sweep config and start them on idle GPUs.

    Args:
        argv: Command line arguments without the program name. None reads them from sys.argv.

    Raises:
        ValueError: When a run holds patients out but would start from an output layer that
            was fitted on them.
        FileNotFoundError: When a run's starting output layer does not exist.

    Notes:
        Every pending run is checked before anything starts, in a dry run too.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher_sweep.json")
    parser.add_argument("--runs-root", default="runs/sweep", help="Prediction files, one folder per run")
    parser.add_argument("--log-dir", default="runs/sweep_logs")
    parser.add_argument("--table", default="results/teacher/runs.csv", help="Aggregate scores only")
    parser.add_argument("--max-parallel", type=int, help="Overrides the config")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--give-up-minutes", type=int, default=120, help="Stop if no GPU frees up for this long")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the idle GPUs, and start nothing")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    max_parallel = args.max_parallel or config.get("max_parallel", 1)
    pending = []
    for settings in expand_runs(config):
        run_id = run_id_for(settings)
        if (RUNS_DIR / run_id / CONFIG_NAME).exists():
            print(f"skip {run_id}: already recorded in {RUNS_DIR / run_id}")
        else:
            pending.append(settings)
    for settings in pending:
        check_probe_head(
            settings.get("probe_head"), settings.get("holdout_fraction", 0.0), settings.get("holdout_seed", 0),
        )
    print(f"{len(pending)} runs, at most {max_parallel} at a time")
    if args.dry_run:
        for settings in pending:
            print(" ".join(build_command(settings, run_id=run_id_for(settings), runs_root=args.runs_root)))
        try:
            gpus, busy = read_gpus()
            print("idle GPUs now:", free_gpu_indices(gpus, busy))
        except FileNotFoundError:
            print("nvidia-smi not found, so idle GPUs cannot be checked on this machine")
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running, failed, idle_since = {}, [], time.time()
    while pending or running:
        for gpu, (process, run_id) in list(running.items()):
            code = process.poll()
            if code is not None:
                del running[gpu]
                print(f"{time.strftime('%H:%M:%S')} {run_id} on GPU {gpu} exited with code {code}", flush=True)
                if code != 0:
                    failed.append(run_id)
        if pending and len(running) < max_parallel:
            gpus, busy = read_gpus()
            for gpu in [g for g in free_gpu_indices(gpus, busy) if g not in running]:
                if not pending or len(running) >= max_parallel:
                    break
                settings = pending.pop(0)
                run_id = run_id_for(settings)
                command = build_command(settings, run_id=run_id, runs_root=args.runs_root)
                env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
                with (log_dir / f"{run_id}.log").open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                running[gpu] = (process, run_id)
                idle_since = time.time()
                print(f"{time.strftime('%H:%M:%S')} started {run_id} on GPU {gpu}", flush=True)
        if running:
            idle_since = time.time()
        elif time.time() - idle_since > args.give_up_minutes * 60:
            print(f"No idle GPU for {args.give_up_minutes} minutes. Giving up on {len(pending)} runs", flush=True)
            break
        time.sleep(args.poll_seconds)

    table = collect_runs(RUNS_DIR)
    if not table.empty:
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        round_scores(table).to_csv(args.table, index=False)
        print(round_scores(table).to_string(index=False))
    print("failed runs:", failed or "none")


if __name__ == "__main__":
    main()
