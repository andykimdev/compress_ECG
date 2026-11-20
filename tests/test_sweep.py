import json
from pathlib import Path

import pytest

from compress_ecg.sweep import build_command, expand_runs, free_gpu_indices, run_id_for

CONFIG = {
    "base": {"batch_size": 64, "max_epochs": 20, "seed": 42},
    "grid": {"lr": [1e-6, 1e-5], "head_lr": [1e-4, 1e-3]},
    "extra_runs": [{"lr": 1e-5, "head_lr": 1e-4, "seed": 43}],
}


def test_grid_expands_in_order_with_extra_runs_last():
    runs = expand_runs(CONFIG)
    assert runs == [
        {"batch_size": 64, "max_epochs": 20, "seed": 42, "lr": 1e-6, "head_lr": 1e-4},
        {"batch_size": 64, "max_epochs": 20, "seed": 42, "lr": 1e-6, "head_lr": 1e-3},
        {"batch_size": 64, "max_epochs": 20, "seed": 42, "lr": 1e-5, "head_lr": 1e-4},
        {"batch_size": 64, "max_epochs": 20, "seed": 42, "lr": 1e-5, "head_lr": 1e-3},
        {"batch_size": 64, "max_epochs": 20, "seed": 43, "lr": 1e-5, "head_lr": 1e-4},
    ]


def test_a_run_that_repeats_earlier_settings_is_dropped():
    config = {"base": {"lr": 1e-5, "head_lr": 1e-4, "seed": 42}, "grid": {"lr": [1e-5]}, "extra_runs": [{"seed": 42}]}
    assert expand_runs(config) == [{"lr": 1e-5, "head_lr": 1e-4, "seed": 42}]


def test_config_without_a_grid_gives_the_base_run():
    assert expand_runs({"base": {"lr": 1e-5, "head_lr": 1e-4, "seed": 1}}) == [{"lr": 1e-5, "head_lr": 1e-4, "seed": 1}]


def test_a_run_without_a_seed_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        expand_runs({"base": {"lr": 1e-5, "head_lr": 1e-4}})


def test_run_id_names_both_rates_and_the_seed():
    assert run_id_for({"lr": 1e-5, "head_lr": 0.0001, "seed": 43}) == "lr1e-05_hlr0.0001_s43"


def test_command_names_the_run_and_scores_only_validation():
    settings = {"lr": 3e-6, "head_lr": 1e-4, "seed": 42, "batch_size": 64, "probe_head": None}
    command = build_command(settings, run_id="r1", runs_root="runs/sweep", python="py")
    assert command[0] == "py"
    assert command[1].endswith("scripts/run_teacher.py")
    assert command[2:] == [
        "--device", "cuda", "--run-id", "r1", "--runs-root", "runs/sweep/r1", "--score-splits", "val",
        "--lr", "3e-06", "--head-lr", "0.0001", "--seed", "42", "--batch-size", "64",
    ]


def test_only_gpus_with_no_process_little_memory_and_low_load_are_free():
    gpus = [
        {"index": 0, "bus_id": "A", "used_mib": 56000, "util": 99},
        {"index": 1, "bus_id": "B", "used_mib": 14, "util": 0},
        {"index": 2, "bus_id": "C", "used_mib": 14, "util": 0},
        {"index": 3, "bus_id": "D", "used_mib": 2800, "util": 0},
        {"index": 4, "bus_id": "E", "used_mib": 14, "util": 60},
    ]
    assert free_gpu_indices(gpus, {"C"}) == [1]


def test_run_id_adds_a_tag_for_each_overfitting_control_that_is_on():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "head_dropout": 0.3, "lead_mask_prob": 0.0, "balance_patients": 0.5}
    assert run_id_for(settings) == "lr1e-05_hlr0.0001_do0.3_bp0.5_s42"


def test_command_passes_the_overfitting_controls():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "head_dropout": 0.3, "balance_patients": 0.5}
    command = build_command(settings, run_id="r1", runs_root="runs/sweep", python="py")
    assert command[-10:] == [
        "--lr", "1e-05", "--head-lr", "0.0001", "--seed", "42", "--head-dropout", "0.3", "--balance-patients", "0.5",
    ]


def test_regularization_config_gives_nine_runs_with_unique_names():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "teacher_regularization.json").read_text())
    runs = expand_runs(config)
    names = [run_id_for(run) for run in runs]
    assert names == [
        "lr1e-05_hlr0.0001_s42",
        "lr1e-05_hlr0.0001_do0.3_s42", "lr1e-05_hlr0.0001_do0.3_s43",
        "lr1e-05_hlr0.0001_lm0.2_s42", "lr1e-05_hlr0.0001_lm0.2_s43",
        "lr1e-05_hlr0.0001_bp0.5_s42", "lr1e-05_hlr0.0001_bp0.5_s43",
        "lr1e-05_hlr0.0001_do0.3_lm0.2_bp0.5_s42", "lr1e-05_hlr0.0001_do0.3_lm0.2_bp0.5_s43",
    ]


def test_run_id_tags_a_run_that_reads_the_7_numbers():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "head_dropout": 0.5, "use_tabular": True}
    assert run_id_for(settings) == "lr1e-05_hlr0.0001_do0.5_tab_s42"


def test_run_id_has_no_tag_when_the_7_numbers_are_off():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "head_dropout": 0.5, "use_tabular": False}
    assert run_id_for(settings) == "lr1e-05_hlr0.0001_do0.5_s42"


def test_command_passes_an_on_switch_without_a_value():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "use_tabular": True, "tabular_lr": 0.001}
    command = build_command(settings, run_id="r1", runs_root="runs/sweep", python="py")
    assert command[-9:] == [
        "--lr", "1e-05", "--head-lr", "0.0001", "--seed", "42", "--use-tabular", "--tabular-lr", "0.001",
    ]


def test_command_leaves_out_a_switch_that_is_off():
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "use_tabular": False}
    command = build_command(settings, run_id="r1", runs_root="runs/sweep", python="py")
    assert command[-6:] == ["--lr", "1e-05", "--head-lr", "0.0001", "--seed", "42"]


def test_tabular_config_gives_five_runs_with_unique_names():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "teacher_tabular.json").read_text())
    names = [run_id_for(run) for run in expand_runs(config)]
    # The first name is the base run. The sweep skips it when its folder already exists.
    assert names == [
        "lr1e-05_hlr0.0001_do0.5_s42",
        "lr1e-05_hlr0.0001_do0.5_tab_s42", "lr1e-05_hlr0.0001_do0.5_tab_s43",
        "lr1e-05_hlr0.0001_do0.5_tab_s44", "lr1e-05_hlr0.0001_do0.5_s44",
    ]
