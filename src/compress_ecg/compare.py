"""
Compare two models that were scored on the same ECGs.

Purpose:
- Measure the AUROC gap between two saved runs.
- Put an interval on the gap by redrawing patients, the same patients for both models.

Special Notes:
- Inputs are saved runs from predictions.load_run, so no model or GPU is needed.
- The result holds summary numbers only and is safe to commit.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import LABEL_COLUMNS, SHD_INDEX
from .predictions import PredictionRun

# Percentiles that bound the middle 95 percent of the resampled values.
INTERVAL = (2.5, 97.5)


def paired_auroc_gap(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, patient_keys: np.ndarray, *,
    n_resamples: int = 2000, seed: int = 0,
) -> dict:
    """
    Measure the AUROC of two models on the same ECGs and the interval of their gap.

    Args:
        labels: [n] true answers, each 0 or 1.
        scores_a: [n] scores of model a, higher for a more likely yes.
        scores_b: [n] scores of model b for the same ECGs in the same order.
        patient_keys: [n] patient of each ECG. A patient's ECGs are drawn together.
        n_resamples: Number of redraws of the patients.
        seed: Seed of the redraws.

    Returns:
        auroc_a, auroc_b and gap (a minus b), each with a low and high end of its 95 percent
        interval, plus share_a_ahead (the share of redraws where a scored higher) and the
        counts n_ecgs, n_patients and n_resamples.

    Raises:
        ValueError: When the arrays differ in length, or labels are not 0 or 1, or one
            class is absent.

    Notes:
        A redraw that holds one class only has no AUROC, so it is drawn again.
    """
    labels, scores_a, scores_b = np.asarray(labels), np.asarray(scores_a), np.asarray(scores_b)
    if not len(labels) == len(scores_a) == len(scores_b) == len(patient_keys):
        raise ValueError("labels, scores and patient keys must have the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Every label must be 0 or 1")
    if len(np.unique(labels)) != 2:
        raise ValueError("AUROC needs both classes among the labels")
    patients, patient_of_ecg = np.unique(patient_keys, return_inverse=True)
    rng = np.random.default_rng(seed)
    resampled = np.empty((n_resamples, 2))
    for index in range(n_resamples):
        while True:
            # A patient drawn k times counts k times, which a weight expresses without copying rows.
            weights = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[patient_of_ecg]
            if weights[labels == 1].any() and weights[labels == 0].any():
                break
        resampled[index] = [roc_auc_score(labels, scores, sample_weight=weights) for scores in (scores_a, scores_b)]
    gaps = resampled[:, 0] - resampled[:, 1]
    auroc_a, auroc_b = float(roc_auc_score(labels, scores_a)), float(roc_auc_score(labels, scores_b))
    result = {"auroc_a": auroc_a, "auroc_b": auroc_b, "gap": auroc_a - auroc_b}
    for name, values in (("auroc_a", resampled[:, 0]), ("auroc_b", resampled[:, 1]), ("gap", gaps)):
        result[f"{name}_low"], result[f"{name}_high"] = (float(value) for value in np.percentile(values, INTERVAL))
    result.update(
        share_a_ahead=float((gaps > 0).mean()), n_ecgs=len(labels), n_patients=len(patients),
        n_resamples=n_resamples,
    )
    return result


def compare_runs(
    run_a: PredictionRun, run_b: PredictionRun, *, label: str = LABEL_COLUMNS[SHD_INDEX],
    n_resamples: int = 2000, seed: int = 0,
) -> dict:
    """
    Compare two saved runs on one output, after lining their rows up by row ID.

    Args:
        run_a: The first model's run.
        run_b: The second model's run on the same ECGs, in any row order.
        label: The output to compare, one of LABEL_COLUMNS.
        n_resamples: Number of redraws of the patients.
        seed: Seed of the redraws.

    Returns:
        The result of paired_auroc_gap for the ECGs whose label is observed, plus the label name.

    Raises:
        ValueError: When the label is unknown, or the runs cover different ECGs or disagree
            on the labels.
    """
    if label not in LABEL_COLUMNS:
        raise ValueError(f"Unknown label {label!r}. Choose one of {LABEL_COLUMNS}")
    column = LABEL_COLUMNS.index(label)
    order_a, order_b = (np.argsort(run.arrays["row_id"]) for run in (run_a, run_b))
    if not np.array_equal(run_a.arrays["row_id"][order_a], run_b.arrays["row_id"][order_b]):
        raise ValueError("The two runs must cover the same ECGs")
    observed = run_a.arrays["label_mask"][order_a, column]
    labels = run_a.arrays["labels"][order_a, column][observed]
    if not np.array_equal(observed, run_b.arrays["label_mask"][order_b, column]) or not np.array_equal(
        labels, run_b.arrays["labels"][order_b, column][observed]
    ):
        raise ValueError(f"The two runs disagree on the {label} labels")
    result = paired_auroc_gap(
        labels, run_a.arrays["probabilities"][order_a, column][observed],
        run_b.arrays["probabilities"][order_b, column][observed],
        run_a.arrays["patient_key"][order_a][observed], n_resamples=n_resamples, seed=seed,
    )
    return {"label": label, **result}
