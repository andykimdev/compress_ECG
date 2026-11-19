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
