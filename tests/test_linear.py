"""
Check the logistic regression baseline that reads the 7 numbers and per-lead waveform summaries.

Purpose:
- Pin the lead summaries, the missing flags and the scaling to values worked out by hand.
- Show that fitting skips ECGs whose label is unknown, and that the command never opens Test.
"""

import json

import numpy as np
import pandas as pd
import pytest

from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX, TABULAR_COLUMNS
from compress_ecg.linear import (
    best_c, build_inputs, fill_value, fit_models, input_names, lead_summaries, missing_flags,
    predict_logits, training_scale,
)
from compress_ecg.predictions import load_run

# Two ECGs, 5 time points, 2 leads, in the stored layout [ecg, 1, time, lead].
# ECG 0: lead 0 is 0, 1, 2, 3, 4 and lead 1 is flat at 7.
# ECG 1: lead 0 is flat at 0 and lead 1 is 0, 10, 20, 30, 40.
WAVEFORMS = np.zeros((2, 1, 5, 2))
WAVEFORMS[0, 0, :, 0] = [0, 1, 2, 3, 4]
WAVEFORMS[0, 0, :, 1] = 7
WAVEFORMS[1, 0, :, 1] = [0, 10, 20, 30, 40]
# Columns are the spread, 95th percentile and 5th percentile, each for lead 0 then lead 1.
# The spread of 0, 1, 2, 3, 4 is the square root of 2. Its 95th percentile sits 0.8 of the way
# from 3 to 4, and its 5th percentile 0.2 of the way from 0 to 1.
SUMMARIES = np.array([
    [np.sqrt(2), 0, 3.8, 7, 0.2, 7],
    [0, 10 * np.sqrt(2), 0, 38, 0, 2],
])
ATRIAL, PR = TABULAR_COLUMNS.index("atrial_rate"), TABULAR_COLUMNS.index("pr_interval")
FILLS = {"atrial_rate": -2.0, "pr_interval": -2.5}


def tabular_rows(atrial: list[float], pr: list[float]) -> np.ndarray:
    """Build stored 7-number rows that are 0 except for the atrial rate and PR interval."""
    rows = np.zeros((len(atrial), len(TABULAR_COLUMNS)))
    rows[:, ATRIAL], rows[:, PR] = atrial, pr
    return rows


def test_lead_summaries_match_the_hand_worked_values():
    assert lead_summaries(WAVEFORMS) == pytest.approx(SUMMARIES)


def test_lead_summaries_do_not_depend_on_the_chunk_size():
    assert lead_summaries(WAVEFORMS, chunk=1) == pytest.approx(SUMMARIES)


def test_lead_summaries_of_no_ecgs_are_empty():
    assert lead_summaries(WAVEFORMS[:0]).shape == (0, 6)


def test_lead_summaries_refuse_a_layout_other_than_the_stored_one():
    with pytest.raises(ValueError, match="ecg, 1, time, lead"):
        lead_summaries(WAVEFORMS[:, 0])


def test_fill_value_is_the_stored_number_on_rows_whose_raw_value_is_missing():
    stored = np.array([0.3, -2.5, 1.1, -2.5])
    raw = np.array([160.0, np.nan, 210.0, np.nan])

    assert fill_value(stored, raw) == -2.5


@pytest.mark.parametrize("stored, raw, message", [
    (np.array([0.3, -2.5, -2.4]), np.array([160.0, np.nan, np.nan]), "one fixed value"),
    (np.array([0.3, 1.1]), np.array([160.0, 210.0]), "no missing"),
    (np.array([0.3, 1.1]), np.array([160.0]), "same length"),
])
def test_fill_value_refuses_inputs_without_one_fixed_fill(stored, raw, message):
    with pytest.raises(ValueError, match=message):
        fill_value(stored, raw)


def test_missing_flags_mark_exactly_the_filled_values():
    tabular = tabular_rows(atrial=[-2.0, 0.4, -2.0], pr=[0.1, -2.5, -2.5])

    assert missing_flags(tabular, FILLS).tolist() == [[1, 0], [0, 1], [1, 1]]


def test_training_scale_is_the_mean_and_spread_of_each_column():
    mean, spread = training_scale(np.array([[0.0, 10.0], [2.0, 30.0]]))

    assert (mean.tolist(), spread.tolist()) == ([1.0, 20.0], [1.0, 10.0])


def test_training_scale_refuses_a_column_that_never_varies():
    with pytest.raises(ValueError, match="column 1"):
        training_scale(np.array([[0.0, 5.0], [2.0, 5.0]]))


def test_inputs_join_the_numbers_the_flags_and_the_scaled_summaries():
    tabular = tabular_rows(atrial=[-2.0, 0.4], pr=[0.1, -2.5])
    summaries = np.array([[3.0, 30.0], [1.0, 20.0]])
    scale = (np.array([1.0, 20.0]), np.array([1.0, 10.0]))

    inputs = build_inputs(tabular, summaries, scale, FILLS)

    assert inputs.tolist() == [
        [0, 0, 0, -2.0, 0.1, 0, 0, 1, 0, 2.0, 1.0],
        [0, 0, 0, 0.4, -2.5, 0, 0, 0, 1, 0.0, 0.0],
    ]


def test_inputs_without_fills_have_no_flag_columns():
    tabular = tabular_rows(atrial=[-2.0, 0.4], pr=[0.1, -2.5])
    scale = (np.zeros(2), np.ones(2))

    inputs = build_inputs(tabular, np.array([[3.0, 30.0], [1.0, 20.0]]), scale, None)

    assert inputs.tolist() == [[0, 0, 0, -2.0, 0.1, 0, 0, 3.0, 30.0], [0, 0, 0, 0.4, -2.5, 0, 0, 1.0, 20.0]]


def test_input_names_follow_the_column_order():
    names = input_names(n_leads=12, flags=True)

    assert len(names) == 45
    assert names[:9] == [*TABULAR_COLUMNS, "atrial_rate_filled", "pr_interval_filled"]
    assert (names[9], names[20], names[21], names[33], names[44]) == (
        "lead0_std", "lead11_std", "lead0_p95", "lead0_p5", "lead11_p5",
    )
    assert len(input_names(n_leads=12, flags=False)) == 43


def test_fitting_skips_ecgs_whose_label_is_unknown():
    # Observed ECGs: a higher input means yes. The unknown ones are stored as 0 at the highest
    # inputs, so a model that counted them as no would rank the two test inputs the other way.
    inputs = np.array([[-2.0], [-1.0], [1.0], [2.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
    labels = np.zeros((10, len(LABEL_COLUMNS)))
    labels[2:4] = 1
    label_mask = np.zeros_like(labels, dtype=bool)
    label_mask[:4] = True

    models = fit_models(inputs, labels, label_mask, c=1.0)
    logits = predict_logits(models, np.array([[-3.0], [3.0]]))

    assert logits.shape == (2, len(LABEL_COLUMNS))
    assert (logits[1] > 0).all() and (logits[0] < 0).all()


def test_fitting_refuses_an_output_with_one_class_among_the_known_labels():
    labels = np.zeros((4, len(LABEL_COLUMNS)))
    labels[2:, :SHD_INDEX] = 1

    with pytest.raises(ValueError, match=LABEL_COLUMNS[SHD_INDEX]):
        fit_models(np.arange(4.0)[:, None], labels, np.ones_like(labels, dtype=bool), c=1.0)


def test_best_c_has_the_highest_score_and_ties_go_to_the_smallest():
    assert best_c({1.0: 0.80, 0.1: 0.80, 0.01: 0.70, 10.0: 0.79}) == 0.1


@pytest.fixture
def linear_dir(data_dir):
    """
    The small dataset with raw measurements, two filled values, and ECGs of different sizes.

    The shared fixture gives every ECG the same wave, so each lead's spread would never vary.
    Test files are deleted, so a command that opened Test would fail.
    """
    path = data_dir / "EchoNext_metadata_100k.csv"
    meta = pd.read_csv(path)
    rng = np.random.default_rng(0)
    for column in ("atrial_rate", "pr_interval"):
        meta[column] = 100.0
    for split in ("train", "val"):
        rows = meta.index[meta["split"].eq(split)]
        tabular = rng.normal(size=(len(rows), 7))
        tabular[0, [ATRIAL, PR]] = [-2.0, -2.5]
        meta.loc[rows[0], ["atrial_rate", "pr_interval"]] = np.nan
        waveforms = rng.normal(size=(len(rows), 1, 2500, 12)) * (1 + np.arange(len(rows)))[:, None, None, None]
        np.save(data_dir / f"EchoNext_{split}_tabular_features.npy", tabular)
        np.save(data_dir / f"EchoNext_{split}_waveforms.npy", waveforms.astype(np.float32))
    meta.to_csv(path, index=False)
    for name in ("waveforms", "tabular_features"):
        (data_dir / f"EchoNext_test_{name}.npy").unlink()
    return data_dir


def test_command_saves_both_models_as_runs_and_writes_their_scores(linear_dir, tmp_path, load_script):
    runs, results = tmp_path / "runs", tmp_path / "results"

    load_script("run_linear").main([
        "--data-dir", str(linear_dir), "--runs-root", str(runs), "--results-dir", str(results),
        "--c-values", "1",
    ])

    for name, n_inputs in (("logistic_regression", 45), ("logistic_regression_no_flags", 43)):
        run = load_run(runs / name / "val")
        saved = json.loads((results / f"{name}_val_metrics.json").read_text())
        assert (run.manifest["model"], run.manifest["split"], len(run.arrays["row_id"])) == (name, "val", 6)
        assert (saved["model"], saved["split"], saved["n_records"], saved["n_inputs"]) == (name, "val", 6, n_inputs)
        assert saved["c"] == 1.0 and list(saved["val_shd_auroc_by_c"]) == ["1.0"]
        assert saved["flagged_training_ecgs"] == {"atrial_rate": 1, "pr_interval": 1}
        assert saved["fill_values"] == {"atrial_rate": -2.0, "pr_interval": -2.5}
        assert saved["metrics"] == run.metrics and list(saved["metrics"]) == list(LABEL_COLUMNS)
        # The fixture leaves one Validation label of the first output unknown.
        assert saved["metrics"][LABEL_COLUMNS[0]]["n_observed"] == 5
        assert np.isnan(run.arrays["labels"]).sum() == 1


def test_command_refuses_fill_values_that_differ_from_the_recorded_table(linear_dir, tmp_path, load_script):
    table = tmp_path / "missing_features.csv"
    pd.DataFrame({
        "feature": ["atrial_rate", "pr_interval"], "missing": [1, 1], "missing_pct": [16.67, 16.67],
        "stored_min": [-2.0, -2.475], "stored_max": [-2.0, -2.475], "split": ["train", "train"],
    }).to_csv(table, index=False)

    with pytest.raises(SystemExit, match="pr_interval"):
        load_script("run_linear").main([
            "--data-dir", str(linear_dir), "--runs-root", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results"), "--missing-table", str(table),
        ])


def test_command_refuses_a_missing_measurement_that_is_not_stored_as_the_fill(linear_dir, tmp_path, load_script):
    path = linear_dir / "EchoNext_val_tabular_features.npy"
    tabular = np.load(path)
    tabular[0, PR] = -2.4
    np.save(path, tabular)

    with pytest.raises(SystemExit, match="val"):
        load_script("run_linear").main([
            "--data-dir", str(linear_dir), "--runs-root", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results"),
        ])
