"""
Check the paired AUROC comparison of two models scored on the same ECGs.

Purpose:
- Pin the scores and the gap to values worked out by hand.
- Show that resampling draws patients, and that mismatched inputs are refused.
"""

import numpy as np
import pandas as pd
import pytest

from compress_ecg.compare import compare_runs, paired_auroc_gap
from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX
from compress_ecg.predictions import PredictionRun

# Four ECGs with a known answer. Scores a rank 3 of the 4 positive and negative pairs
# correctly, which is an AUROC of 0.75. Scores b rank all 4 correctly, which is 1.0.
LABELS = np.array([0, 0, 1, 1])
SCORES_A = np.array([0.1, 0.4, 0.35, 0.8])
SCORES_B = np.array([0.1, 0.2, 0.7, 0.9])
PATIENTS = np.array(["p0", "p1", "p2", "p3"])


def separated_scores(count: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, strong scores, uninformative scores and one patient key per ECG."""
    rng = np.random.default_rng(3)
    labels = np.arange(count) % 2
    strong = labels + rng.normal(0, 0.5, count)
    weak = rng.normal(0, 1, count)
    return labels, strong, weak, np.arange(count).astype(str)


def make_run(row_ids: np.ndarray, labels: np.ndarray, scores: np.ndarray, patients: np.ndarray) -> PredictionRun:
    """Build an in-memory run whose SHD column holds the given labels and scores."""
    shape = (len(row_ids), len(LABEL_COLUMNS))
    probabilities = np.zeros(shape, dtype=np.float32)
    probabilities[:, SHD_INDEX] = scores
    label_array = np.zeros(shape, dtype=np.float32)
    label_array[:, SHD_INDEX] = labels
    arrays = {
        "probabilities": probabilities, "labels": label_array,
        "label_mask": np.ones(shape, dtype=bool), "row_id": row_ids, "patient_key": patients,
    }
    return PredictionRun(arrays=arrays, metadata=pd.DataFrame({"row_id": row_ids}), manifest={}, metrics={})


def test_scores_and_gap_match_the_hand_worked_values():
    result = paired_auroc_gap(LABELS, SCORES_A, SCORES_B, PATIENTS, n_resamples=50, seed=0)

    assert result["auroc_a"] == pytest.approx(0.75)
    assert result["auroc_b"] == pytest.approx(1.0)
    assert result["gap"] == pytest.approx(-0.25)
    assert (result["n_ecgs"], result["n_patients"], result["n_resamples"]) == (4, 4, 50)


def test_identical_scores_give_a_zero_gap_with_a_zero_width_interval():
    labels, strong, _, patients = separated_scores()

    result = paired_auroc_gap(labels, strong, strong, patients, n_resamples=40, seed=1)

    assert (result["gap"], result["gap_low"], result["gap_high"]) == (0.0, 0.0, 0.0)


def test_interval_sits_above_zero_when_one_model_is_far_better():
    labels, strong, weak, patients = separated_scores()

    result = paired_auroc_gap(labels, strong, weak, patients, n_resamples=100, seed=1)

    assert 0 < result["gap_low"] < result["gap"] < result["gap_high"] < 1
    assert result["share_a_ahead"] == 1.0


def test_same_seed_gives_the_same_interval():
    labels, strong, weak, patients = separated_scores()

    first = paired_auroc_gap(labels, strong, weak, patients, n_resamples=40, seed=7)
    second = paired_auroc_gap(labels, strong, weak, patients, n_resamples=40, seed=7)

    assert first == second


def test_resampling_draws_patients_so_a_repeated_ecg_does_not_narrow_the_interval():
    labels, strong, weak, patients = separated_scores(60)
    once = paired_auroc_gap(labels, strong, weak, patients, n_resamples=60, seed=5)

    # Every patient now has the same ECG twice. Drawing patients gives the same resamples as before.
    twice = paired_auroc_gap(
        np.tile(labels, 2), np.tile(strong, 2), np.tile(weak, 2), np.tile(patients, 2), n_resamples=60, seed=5,
    )

    assert (twice["n_ecgs"], twice["n_patients"]) == (120, 60)
    assert twice["gap_low"] == pytest.approx(once["gap_low"])
    assert twice["gap_high"] == pytest.approx(once["gap_high"])


@pytest.mark.parametrize("labels, scores_b, message", [
    (LABELS, SCORES_B[:3], "same length"),
    (np.array([1, 1, 1, 1]), SCORES_B, "both classes"),
    (np.array([0, 0, 1, np.nan]), SCORES_B, "0 or 1"),
])
def test_bad_inputs_are_refused(labels, scores_b, message):
    with pytest.raises(ValueError, match=message):
        paired_auroc_gap(labels, SCORES_A, scores_b, PATIENTS, n_resamples=10)


def test_runs_are_lined_up_by_row_id_before_comparing():
    row_ids = np.array([10, 11, 12, 13])
    run_a = make_run(row_ids, LABELS, SCORES_A, PATIENTS)
    shuffled = np.array([2, 0, 3, 1])
    run_b = make_run(row_ids[shuffled], LABELS[shuffled], SCORES_B[shuffled], PATIENTS[shuffled])

    result = compare_runs(run_a, run_b, n_resamples=50, seed=0)

    assert result["label"] == LABEL_COLUMNS[SHD_INDEX]
    assert (result["auroc_a"], result["auroc_b"]) == (pytest.approx(0.75), pytest.approx(1.0))


def test_runs_on_different_ecgs_are_refused():
    run_a = make_run(np.array([10, 11, 12, 13]), LABELS, SCORES_A, PATIENTS)
    run_b = make_run(np.array([10, 11, 12, 99]), LABELS, SCORES_B, PATIENTS)

    with pytest.raises(ValueError, match="same ECGs"):
        compare_runs(run_a, run_b, n_resamples=10)


def test_command_writes_the_summary_of_two_written_runs(data_dir, tmp_path, load_script):
    import json

    from torch import nn

    from compress_ecg.data import EchoNextDataset
    from compress_ecg.predictions import write_run

    class ByFirstFeature(nn.Module):
        """Score every output by the first tabular number times a fixed sign."""

        def __init__(self, sign: float):
            super().__init__()
            self.sign = sign

        def forward(self, waveform, tabular):
            return (self.sign * tabular[:, :1] * 10).repeat(1, len(LABEL_COLUMNS))

    val = EchoNextDataset(data_dir, "val")
    folder_a = write_run(ByFirstFeature(1.0), "teacher", val, runs_root=tmp_path / "runs")
    folder_b = write_run(ByFirstFeature(-1.0), "mini", val, runs_root=tmp_path / "runs")
    output = tmp_path / "results" / "gap.json"

    load_script("compare_runs").main(
        ["--a", str(folder_a), "--b", str(folder_b), "--resamples", "20", "--output", str(output)]
    )

    saved = json.loads(output.read_text())
    assert (saved["split"], saved["model_a"], saved["model_b"]) == ("val", "teacher", "mini")
    assert (saved["label"], saved["n_ecgs"], saved["n_resamples"]) == (LABEL_COLUMNS[SHD_INDEX], 6, 20)
    # The 6 Validation ECGs have labels 0, 1, 0, 1, 0, 1 and a first number that rises row by row.
    # Model a ranks 6 of the 9 positive and negative pairs correctly, and model b the other 3.
    assert (saved["auroc_a"], saved["auroc_b"], saved["gap"]) == (0.6667, 0.3333, 0.3333)
