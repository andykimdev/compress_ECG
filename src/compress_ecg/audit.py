"""
Score every model on accuracy, calibration, and fairness.

Purpose:
- Turn saved prediction runs into one comparison across models.
- Report overall and per patient group results with confidence intervals.

Special Notes:
- Owner: Role E. Every function here is still to be written.
- Inputs are only saved runs, from predictions.load_run, so no model or GPU is needed.
- Only the aggregate tables and figures written here may be committed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from .data import AUDIT_COLUMNS, LABEL_COLUMNS, SHD_INDEX
from .predictions import PredictionRun, load_run, basic_metrics


def score_run(run: PredictionRun) -> pd.DataFrame:
    """
    Score one run overall.

    Args:
        run: A saved run.

    Returns:
        One row per label with AUROC, calibration error, and Brier score, each with a
        bootstrap confidence interval that resamples patients.
    """
    raise NotImplementedError("Role E: compute overall metrics with bootstrap intervals")


def score_groups(run: PredictionRun) -> pd.DataFrame:
    """
    Score one run inside each patient group.

    Args:
        run: A saved run, whose metadata holds race and ethnicity, sex, age, and care setting.

    Returns:
        One row per group, with the same metrics as score_run plus the group's positive count.
    """
    raise NotImplementedError("Role E: compute per-group metrics and the worst group")


def recalibrate(validation_run: PredictionRun, test_run: PredictionRun) -> PredictionRun:
    """
    Fit a calibration correction on Validation and apply it to Test.

    Args:
        validation_run: The run used to fit the correction.
        test_run: The run that receives it, unchanged in every other way.

    Returns:
        The Test run with corrected logits and probabilities.
    """
    raise NotImplementedError("Role E: fit on Validation only, then apply to Test")


def write_scorecard(runs_root: str | Path, output_dir: str | Path) -> Path:
    """
    Compare every model found under a runs folder.

    Args:
        runs_root: Folder laid out by predictions.run_dir.
        output_dir: Where to write the aggregate tables and figures.

    Returns:
        The output folder.
    """
    prediction_run = load_run(runs_root)
    outputs = prediction_run.arrays
    group_metrics = {}
    for col in AUDIT_COLUMNS:
        group_partition = {name: group for name, group in prediction_run.metadata.groupby(col)}
        partition_metrics = {}
        for group_name, partition in group_partition.items():
            group_mask = np.isin(outputs["row_id"], partition["row_id"])
            group_outputs = {output_name: output_val[group_mask] for output_name, output_val in outputs.items() if output_name != "label_columns"}
            group_outputs["label_columns"] = outputs["label_columns"]

            metrics = basic_metrics(group_outputs)
            partition_metrics[group_name] = metrics
        group_metrics[col] = partition_metrics
    (Path(output_dir) / "group_metrics.json").write_text(json.dumps(group_metrics, indent=2, allow_nan=False) + "\n")
    plot_roc_auc(prediction_run, output_path=Path(output_dir) / "roc_auc.png")
    plot_shd_calibration(prediction_run, output_path=Path(output_dir) / "shd_calibration.png")

def measure_model_size(model) -> int:
    """Return the size of a PyTorch model in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())

def plot_roc_auc(
    run: PredictionRun,
    *,
    ax=None,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the ROC curve for the SHD label only.

    Args:
        run: A loaded PredictionRun.
        ax: Optional matplotlib axis to draw on.
        title: Optional plot title. Defaults to the model name and split.
        output_path: Optional file path to save the image to.

    Returns:
        The matplotlib Figure and Axes used for the plot.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    if title is None:
        model = run.manifest.get("model", "model")
        split = run.manifest.get("split", "split")
        title = f"ROC AUC — SHD ({model}, {split})"
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Chance")

    index = SHD_INDEX
    name = LABEL_COLUMNS[index]
    mask = run.arrays["label_mask"][:, index]
    y = run.arrays["labels"][mask, index]
    p = run.arrays["probabilities"][mask, index]
    if len(y) == 0 or np.unique(y).size < 2:
        ax.text(0.5, 0.5, "No observed SHD labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        fpr, tpr, _ = roc_curve(y, p)
        auc_value = roc_auc_score(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_value:.3f})")
        ax.legend(loc="lower right", fontsize="small")

    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_shd_calibration(
    run: PredictionRun,
    *,
    ax=None,
    title: str | None = None,
    bins: int = 10,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration for the SHD label only.

    Args:
        run: A loaded PredictionRun.
        ax: Optional matplotlib axis.
        title: Optional plot title. Defaults to the model name and split.
        bins: Number of probability bins used for the calibration curve.
        output_path: Optional file path to save the image to.

    Returns:
        The matplotlib Figure and Axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    if title is None:
        model = run.manifest.get("model", "model")
        split = run.manifest.get("split", "split")
        title = f"Calibration — SHD ({model}, {split})"
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")

    mask = run.arrays["label_mask"][:, SHD_INDEX]
    y = run.arrays["labels"][mask, SHD_INDEX]
    p = run.arrays["probabilities"][mask, SHD_INDEX]
    if len(y) == 0:
        ax.text(0.5, 0.5, "No observed SHD labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return fig, ax

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed = []
    predicted = []

    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (p >= left) & (p < right)
        if right == 1.0:
            in_bin = (p >= left) & (p <= right)
        if not in_bin.any():
            observed.append(np.nan)
            predicted.append(np.nan)
            continue
        observed.append(y[in_bin].mean())
        predicted.append(p[in_bin].mean())

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~np.isnan(observed) & ~np.isnan(predicted)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
    ax.plot(predicted[valid], observed[valid], marker="o", linewidth=2, label="SHD calibration")
    ax.scatter(bin_centers[valid], observed[valid], s=20, alpha=0.8)
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, ax

