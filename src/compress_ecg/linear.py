"""
Fit a logistic regression baseline on the 7 numbers and per-lead waveform summaries.

Purpose:
- Turn each lead into 3 numbers that do not depend on where the heartbeats fall.
- Flag the atrial rate and PR interval values that the dataset makers filled in.
- Fit one logistic regression per output, on the ECGs whose label is known.

Special Notes:
- The inputs come from the same waveform and 7 numbers that every other model receives.
- A linear model reads a filled value as a real, very low measurement, hence the flags.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .data import LABEL_COLUMNS, TABULAR_COLUMNS, EchoNextDataset

# The two measurements the dataset makers set to 0 when missing, in flag column order.
FLAG_COLUMNS = ("atrial_rate", "pr_interval")
# Summaries per lead, in column order: spread, 95th percentile, 5th percentile.
SUMMARY_NAMES = ("std", "p95", "p5")
# Values of scikit-learn's C to try. A smaller C keeps the weights smaller.
C_VALUES = (0.00001, 0.0001, 0.001, 0.01, 0.1)


def lead_summaries(waveforms: np.ndarray, *, chunk: int = 512) -> np.ndarray:
    """
    Summarize every lead of every ECG by its spread and its high and low points.

    Args:
        waveforms: [ecg, 1, time, lead] in the stored layout. A memory map is read in chunks.
        chunk: ECGs read at a time.

    Returns:
        [ecg, 3 x lead] with the standard deviation, 95th percentile and 5th percentile over
        time, each for every lead in order.

    Raises:
        ValueError: When the layout is not the stored one or a value is not finite.
    """
    if waveforms.ndim != 4 or waveforms.shape[1] != 1:
        raise ValueError(f"Waveforms must be [ecg, 1, time, lead], got {waveforms.shape}")
    summaries = np.empty((len(waveforms), len(SUMMARY_NAMES) * waveforms.shape[3]))
    for start in range(0, len(waveforms), chunk):
        block = np.asarray(waveforms[start:start + chunk, 0], dtype=np.float64)
        low, high = np.percentile(block, (5, 95), axis=1)
        summaries[start:start + chunk] = np.concatenate([block.std(axis=1), high, low], axis=1)
    if not np.isfinite(summaries).all():
        raise ValueError("A waveform holds a value that is not finite")
    return summaries


def fill_value(stored: np.ndarray, raw: np.ndarray) -> float:
    """
    Find the number the stored array holds where the raw measurement is missing.

    Args:
        stored: [n] one column of the stored 7 numbers.
        raw: [n] the same measurement from the metadata, NaN where it was not measured.

    Returns:
        The one stored value shared by every row whose raw value is missing.

    Raises:
        ValueError: When the lengths differ, no raw value is missing, or the rows with a
            missing raw value do not share one fixed value.
    """
    if len(stored) != len(raw):
        raise ValueError("stored and raw must have the same length")
    values = np.unique(np.asarray(stored)[np.isnan(raw)])
    if not len(values):
        raise ValueError("The raw column has no missing value to learn the fill from")
    if len(values) != 1:
        raise ValueError(f"Missing raw values must be stored as one fixed value, got {values[:5]}")
    return float(values[0])


def missing_flags(tabular: np.ndarray, fills: dict[str, float]) -> np.ndarray:
    """
    Mark the ECGs whose atrial rate or PR interval holds the fill value.

    Args:
        tabular: [n, 7] the stored 7 numbers.
        fills: The fill value of each column in FLAG_COLUMNS, from fill_value.

    Returns:
        [n, 2] with 1 where the column equals its fill value and 0 elsewhere.

    Notes:
        A raw measurement of exactly 0 is stored as the same number, so it is flagged too.
    """
    return np.stack(
        [tabular[:, TABULAR_COLUMNS.index(name)] == fills[name] for name in FLAG_COLUMNS], axis=1,
    ).astype(np.float64)


def training_scale(summaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure the mean and spread of each summary column on the Training ECGs.

    Args:
        summaries: [n, columns] lead summaries of the Training ECGs.

    Returns:
        The mean and the standard deviation of every column.

    Raises:
        ValueError: When a column never varies, so it cannot be scaled.
    """
    mean, spread = summaries.mean(axis=0), summaries.std(axis=0)
    if (spread == 0).any():
        raise ValueError(f"Summary column {int(np.flatnonzero(spread == 0)[0])} never varies in Training")
    return mean, spread


def build_inputs(
    tabular: np.ndarray, summaries: np.ndarray, scale: tuple[np.ndarray, np.ndarray],
    fills: dict[str, float] | None,
) -> np.ndarray:
    """
    Join the model inputs of each ECG into one row.

    Args:
        tabular: [n, 7] the stored 7 numbers, used as they are.
        summaries: [n, columns] lead summaries from lead_summaries.
        scale: Mean and spread from training_scale, applied to the summaries.
        fills: Fill values for the missing flags, or None to leave the flags out.

    Returns:
        [n, 7 + flags + columns] in the order of input_names.
    """
    mean, spread = scale
    flags = [] if fills is None else [missing_flags(tabular, fills)]
    return np.concatenate([tabular, *flags, (summaries - mean) / spread], axis=1)


def input_names(n_leads: int, flags: bool) -> list[str]:
    """
    Name the columns that build_inputs returns.

    Args:
        n_leads: Number of leads in the waveform.
        flags: Whether the missing flags are included.

    Returns:
        The 7 numbers, then the flags when included, then each summary for every lead.
    """
    names = list(TABULAR_COLUMNS) + ([f"{name}_filled" for name in FLAG_COLUMNS] if flags else [])
    return names + [f"lead{lead}_{summary}" for summary in SUMMARY_NAMES for lead in range(n_leads)]


def fit_models(inputs: np.ndarray, labels: np.ndarray, label_mask: np.ndarray, c: float) -> list[LogisticRegression]:
    """
    Fit one logistic regression per output on the ECGs whose label is known.

    Args:
        inputs: [n, features] from build_inputs.
        labels: [n, 12] with 0 or 1. Values where label_mask is False are ignored.
        label_mask: [n, 12] True where the label is known.
        c: scikit-learn's C, the inverse of the regularization strength.

    Returns:
        The fitted models in the order of LABEL_COLUMNS.

    Raises:
        ValueError: When an output has only one class among its known labels.
    """
    models = []
    for column, name in enumerate(LABEL_COLUMNS):
        known = label_mask[:, column]
        if len(np.unique(labels[known, column])) != 2:
            raise ValueError(f"{name} needs both classes among its known labels")
        models.append(LogisticRegression(C=c, max_iter=1000).fit(inputs[known], labels[known, column]))
    return models


def predict_logits(models: list[LogisticRegression], inputs: np.ndarray) -> np.ndarray:
    """
    Score ECGs with the fitted models.

    Args:
        models: One model per output, from fit_models.
        inputs: [n, features] from build_inputs.

    Returns:
        [n, 12] logits, higher for a more likely yes.
    """
    return np.stack([model.decision_function(inputs) for model in models], axis=1)


def best_c(scores: dict[float, float]) -> float:
    """
    Pick the C with the highest score.

    Args:
        scores: The Validation score reached with each C.

    Returns:
        The C with the highest score. A tie goes to the smallest C, the simpler model.
    """
    return min(scores, key=lambda c: (-scores[c], c))


def prediction_arrays(dataset: EchoNextDataset, logits: np.ndarray) -> dict[str, np.ndarray]:
    """
    Pack logits for a whole split into the arrays that predictions.save_run writes.

    Args:
        dataset: The ECGs that were scored, in file order.
        logits: [n, 12] from predict_logits, one row per ECG of the dataset.

    Returns:
        The arrays in predictions.ARRAY_NAMES. Unknown labels are NaN.

    Raises:
        ValueError: When the logits do not have one row per ECG.
    """
    if logits.shape != (len(dataset), len(LABEL_COLUMNS)):
        raise ValueError(f"Expected logits {(len(dataset), len(LABEL_COLUMNS))}, got {logits.shape}")
    selected = dataset.selected_metadata
    label_mask = dataset.label_mask[dataset.indices]
    labels = dataset.labels[dataset.indices].astype(np.float32)
    labels[~label_mask] = np.nan
    return {
        "logits": logits.astype(np.float32), "probabilities": expit(logits).astype(np.float32),
        "labels": labels, "label_mask": label_mask,
        "row_id": selected["row_id"].to_numpy(), "split_row": selected["split_row"].to_numpy(),
        "patient_key": selected["patient_key"].to_numpy(dtype=str),
        "label_columns": np.asarray(LABEL_COLUMNS, dtype=str),
    }
