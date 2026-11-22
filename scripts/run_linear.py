#!/usr/bin/env python3
"""Fit the logistic regression baseline and score it on Validation. Install first: pip install -e ."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compress_ecg.data import SHD_INDEX, TABULAR_COLUMNS, EchoNextDataset
from compress_ecg.linear import (
    C_VALUES, FLAG_COLUMNS, best_c, build_inputs, fill_value, fit_models, input_names,
    lead_summaries, missing_flags, predict_logits, prediction_arrays, training_scale,
)
from compress_ecg.predictions import fresh_run_dir, run_manifest, save_run

# Model name and whether it gets the missing flags. Each name owns one run folder.
VERSIONS = (("logistic_regression", True), ("logistic_regression_no_flags", False))


def training_fills(train: EchoNextDataset, tabular: np.ndarray, missing_table: str | None) -> dict[str, float]:
    """
    Learn the fill value of each flagged column from the Training ECGs.

    Args:
        train: The Training ECGs, whose metadata holds the raw measurements.
        tabular: [n, 7] the stored 7 numbers of the Training ECGs.
        missing_table: missing_features.csv from the data statistics, or None to skip the check.

    Returns:
        The fill value of every column in FLAG_COLUMNS.

    Raises:
        SystemExit: When a fill value or a missing count differs from the recorded table.
    """
    fills = {
        name: fill_value(tabular[:, TABULAR_COLUMNS.index(name)], train.metadata[name].to_numpy(dtype=float))
        for name in FLAG_COLUMNS
    }
    if missing_table:
        table = pd.read_csv(missing_table).query("split == 'train'").set_index("feature")
        for name, value in fills.items():
            found = (round(value, 3), round(value, 3), int(train.metadata[name].isna().sum()))
            recorded = (table.at[name, "stored_min"], table.at[name, "stored_max"], table.at[name, "missing"])
            if found != recorded:
                raise SystemExit(f"{name}: fill and missing count {found} differ from the recorded {recorded}")
    return fills


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Folder with the EchoNext files. Defaults to ECHONEXT_DIR")
    parser.add_argument("--runs-root", default="runs", help="Parent folder of the run folders, never committed")
    parser.add_argument("--results-dir", default="results/baselines", help="Folder for the summary numbers")
    parser.add_argument("--c-values", type=float, nargs="+", default=C_VALUES, help="Values of C to try")
    parser.add_argument("--chunk", type=int, default=512, help="ECGs read at a time for the lead summaries")
    parser.add_argument("--missing-table", help="missing_features.csv to check the fill values against")
    args = parser.parse_args(argv)

    # Test is never opened here. The settings are chosen on Validation, so Test waits for the final pass.
    train, val = EchoNextDataset(args.data_dir, "train"), EchoNextDataset(args.data_dir, "val")
    tabular = {group.split: np.load(group.tabular_path)[group.indices] for group in (train, val)}
    fills = training_fills(train, tabular["train"], args.missing_table)
    for group in (train, val):
        unmeasured = group.metadata[list(FLAG_COLUMNS)].isna().to_numpy()
        if not missing_flags(tabular[group.split], fills)[unmeasured].all():
            raise SystemExit(f"A missing measurement in {group.split} is not stored as the Training fill value")
    started = time.perf_counter()
    summaries = {
        group.split: lead_summaries(np.load(group.waveform_path, mmap_mode="r"), chunk=args.chunk)
        for group in (train, val)
    }
    summary_seconds = time.perf_counter() - started
    scale = training_scale(summaries["train"])
    flagged = missing_flags(tabular["train"], fills).sum(axis=0).astype(int).tolist()
    val_shd = val.labels[:, SHD_INDEX]

    for name, with_flags in VERSIONS:
        inputs = {
            split: build_inputs(tabular[split], summaries[split], scale, fills if with_flags else None)
            for split in ("train", "val")
        }
        started = time.perf_counter()
        logits = {
            c: predict_logits(fit_models(inputs["train"], train.labels, train.label_mask, c), inputs["val"])
            for c in args.c_values
        }
        fit_seconds = time.perf_counter() - started
        scores = {c: float(roc_auc_score(val_shd, logits[c][:, SHD_INDEX])) for c in args.c_values}
        chosen = best_c(scores)
        record = {
            "n_inputs": inputs["val"].shape[1], "inputs": input_names(summaries["val"].shape[1] // 3, with_flags),
            "fitted_on": "train", "c": chosen, "val_shd_auroc_by_c": {str(c): scores[c] for c in scores},
            "fill_values": fills,
            "seconds": {"lead_summaries": round(summary_seconds, 1), "fit_all_c": round(fit_seconds, 1)},
        }
        manifest = run_manifest(name, val, **record)
        destination = fresh_run_dir(Path(args.runs_root) / name / val.group, val.root)
        metrics = save_run(
            destination, prediction_arrays(val, logits[chosen]),
            val.selected_metadata.assign(split=val.group), manifest,
        )
        aurocs = [entry["auroc"] for entry in metrics.values() if entry["auroc"] is not None]
        summary = {
            "model": name, "split": val.group, "n_records": len(val), "code": manifest["code"],
            "data_version": manifest["data_version"], **record,
            "flagged_training_ecgs": dict(zip(FLAG_COLUMNS, flagged)),
            "mean_auroc": float(np.mean(aurocs)), "metrics": metrics,
        }
        path = Path(args.results_dir) / f"{name}_{val.group}_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"model": name, "n_inputs": record["n_inputs"], "c": chosen,
                          "val_shd_auroc_by_c": record["val_shd_auroc_by_c"], "mean_auroc": summary["mean_auroc"],
                          "seconds": record["seconds"], "run": str(destination), "summary": str(path)}, indent=2))


if __name__ == "__main__":
    main()
