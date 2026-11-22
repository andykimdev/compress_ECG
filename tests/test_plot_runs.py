"""
Check how the Teacher run figure groups and labels the runs of the run table.

Purpose:
- Pin the plain-word label of each setting.
- Show that runs differing only in the 7 extra numbers or in weight averaging get their own rows.
- Pin the axis limits to hand-worked values, so no run is drawn outside the panel.
"""

import numpy as np
import pandas as pd
import pytest

NAN = np.nan


def run(run_id: str, mean: float, **settings) -> dict:
    """Return one row of the run table. A setting that is not given is blank, as in older runs."""
    row = {
        "run_id": run_id, "lr": 1e-5, "head_lr": 1e-4, "head_dropout": NAN, "lead_mask_prob": NAN,
        "balance_patients": NAN, "use_tabular": NAN, "ema_decay": NAN, "evals_per_epoch": NAN,
        "select_by": NAN, "seed": 42, "val_mean_auroc": mean, "val_shd_auroc": mean + 0.01,
    }
    return {**row, **settings}


AVERAGED = {"head_dropout": 0.5, "use_tabular": True, "ema_decay": 0.999, "evals_per_epoch": 4, "select_by": "shd"}


@pytest.mark.parametrize("settings, label", [
    ({}, "No control (baseline)"),
    ({"lr": 3e-5}, "ECG-FM rate 3e-05, output rate 0.0001"),
    ({"head_dropout": 0.5}, "Dropout 0.5"),
    ({"lead_mask_prob": 0.2}, "Lead masking 0.2"),
    ({"head_dropout": 0.3, "lead_mask_prob": 0.2, "balance_patients": 0.5}, "All three controls"),
    ({"head_dropout": 0.5, "use_tabular": True}, "Dropout 0.5, 7 extra numbers"),
    ({"head_dropout": 0.5, "use_tabular": False}, "Dropout 0.5"),
    (AVERAGED, "Dropout 0.5, 7 extra numbers, weight averaging"),
])
def test_setting_label_names_each_setting_in_plain_words(load_script, settings, label):
    plot_runs = load_script("plot_runs")
    table = plot_runs.fill_blank_settings(pd.DataFrame([run("a", 0.8, **settings)]))

    assert plot_runs.setting_label(table.iloc[0]) == label


def test_build_rows_gives_the_extra_numbers_and_weight_averaging_their_own_rows(load_script):
    table = pd.DataFrame([
        run("baseline", 0.81),
        run("dropout", 0.83, head_dropout=0.5),
        run("numbers", 0.832, head_dropout=0.5, use_tabular=True),
        run("averaged_s42", 0.835, **AVERAGED),
        run("averaged_s43", 0.836, **AVERAGED, seed=43),
        run("fast_rate", 0.82, lr=3e-5),
    ])

    rows = load_script("plot_runs").build_rows(table)

    assert [(group, label, sorted(runs["run_id"])) for group, label, runs in rows] == [
        ("Learning rates", "ECG-FM rate 3e-05, output rate 0.0001", ["fast_rate"]),
        ("Learning rates", "No control (baseline)", ["baseline"]),
        ("Overfitting controls", "Dropout 0.5", ["dropout"]),
        ("Overfitting controls", "No control (baseline)", ["baseline"]),
        ("Extra inputs and weight averaging", "Dropout 0.5, 7 extra numbers, weight averaging",
         ["averaged_s42", "averaged_s43"]),
        ("Extra inputs and weight averaging", "Dropout 0.5, 7 extra numbers", ["numbers"]),
    ]


def test_build_rows_refuses_a_run_that_no_group_would_draw(load_script):
    table = pd.DataFrame([run("baseline", 0.81), run("odd", 0.82, lr=3e-5, head_dropout=0.5)])

    with pytest.raises(ValueError, match="odd"):
        load_script("plot_runs").build_rows(table)


@pytest.mark.parametrize("values, limits", [
    ([79.21, 83.68], (78.5, 84.5)),
    ([82.0], (81.5, 82.5)),
    ([81.48, 85.16, 83.0], (80.5, 86.0)),
])
def test_axis_limits_hold_every_value_with_half_a_point_of_room(load_script, values, limits):
    assert load_script("plot_runs").axis_limits(values) == limits


def test_axis_limits_refuse_an_empty_list(load_script):
    with pytest.raises(ValueError, match="no values"):
        load_script("plot_runs").axis_limits([])
