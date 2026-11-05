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

from .predictions import PredictionRun


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
    raise NotImplementedError("Role E: load each run, score it, and write the comparison")
