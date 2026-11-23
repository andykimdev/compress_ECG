"""Run scripts/run_student.py on the small dataset, with a stand-in Student and a saved Teacher run."""

import json

import pytest
import torch
from torch import nn

from compress_ecg.data import EchoNextDataset
from compress_ecg.predictions import write_run


class TinyStudent(nn.Module):
    """Stand-in for the Student, in the shared call format, with random starting weights."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(7, 12)

    def forward(self, waveform, tabular):
        return self.layer(tabular)


def run_once(script, data_dir, tmp_path, seed: int, folder: str, extra: list[str] = ()) -> dict:
    """Run the script with one seed and return the starting weights and the seed training saw."""
    seen = {}

    def build(device="cpu"):
        seen["start"] = TinyStudent()
        return seen["start"]

    def train(student, teacher_run, train, val, output_dir, *, device="cpu", seed=42, **settings):
        seen["seed"] = seed
        seen["settings"] = settings
        seen["start_weight"] = student.layer.weight.detach().clone()
        return "checkpoint"

    script.build_student = build
    script.train_student = train
    script.load_student = lambda checkpoint, device="cpu": TinyStudent()
    script.main([
        "--data-dir", str(data_dir), "--teacher-run", str(tmp_path / "runs" / "teacher" / "train"),
        "--runs-root", str(tmp_path / folder), "--seed", str(seed), *extra,
    ])
    return seen


def test_seed_reaches_training_and_repeats_the_starting_weights(data_dir, tmp_path, load_script):
    write_run(TinyStudent(), "teacher", EchoNextDataset(data_dir, "train"), runs_root=tmp_path / "runs")
    script = load_script("run_student")

    first = run_once(script, data_dir, tmp_path, 43, "a")
    again = run_once(script, data_dir, tmp_path, 43, "b")
    other = run_once(script, data_dir, tmp_path, 44, "c")

    assert (first["seed"], other["seed"]) == (43, 44)
    torch.testing.assert_close(first["start_weight"], again["start_weight"])
    assert not torch.equal(first["start_weight"], other["start_weight"])


def test_config_settings_reach_training(data_dir, tmp_path, load_script):
    write_run(TinyStudent(), "teacher", EchoNextDataset(data_dir, "train"), runs_root=tmp_path / "runs")
    config = tmp_path / "student.json"
    config.write_text(json.dumps({"temperature": 4.0, "hard_label_weight": 1.0, "num_workers": 2}))
    script = load_script("run_student")

    seen = run_once(script, data_dir, tmp_path, 42, "a", ["--config", str(config)])

    assert seen["settings"]["temperature"] == 4.0
    assert seen["settings"]["hard_label_weight"] == 1.0
    assert seen["settings"]["num_workers"] == 2
    assert seen["settings"]["on_epoch"] is None


def test_unknown_config_setting_fails_before_training(data_dir, tmp_path, load_script):
    write_run(TinyStudent(), "teacher", EchoNextDataset(data_dir, "train"), runs_root=tmp_path / "runs")
    config = tmp_path / "student.json"
    config.write_text(json.dumps({"temprature": 4.0}))
    script = load_script("run_student")

    with pytest.raises(ValueError, match="temprature"):
        run_once(script, data_dir, tmp_path, 42, "a", ["--config", str(config)])


def test_effective_settings_fill_in_training_defaults(load_script):
    script = load_script("run_student")

    assert script.effective_settings({"temperature": 4.0, "num_workers": 4}) == {
        "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "max_epochs": 50, "patience": 7,
        "min_delta": 0.001, "temperature": 4.0, "hard_label_weight": 0.5, "num_workers": 4,
    }
