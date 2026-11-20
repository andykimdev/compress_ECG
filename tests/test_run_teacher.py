"""Run scripts/run_teacher.py end to end on the small dataset, with a stand-in for ECG-FM."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from compress_ecg.teacher import Teacher, TeacherHead
from test_teacher import FakeEncoder

# ECGs per Training patient in the holdout_dir fixture.
ECGS_OF = {"a": 1, "b": 2, "c": 2, "d": 1}
# One ECG per update, a scoring after every update, and no early stop, so the step column of
# the log counts the Training ECGs that were used.
COUNT_TRAINING_ECGS = ("--batch-size", "1", "--evals-per-epoch", "6", "--patience", "10")


@pytest.fixture
def run_teacher(holdout_dir, tmp_path, monkeypatch, load_script):
    """
    Return a function that runs the script in tmp_path with extra arguments.

    ECG-FM is replaced by the fake encoder. Run folders go to tmp_path/checkpoints/<run id>
    and prediction files to tmp_path/runs/<run id>.
    """
    script = load_script("run_teacher")

    def fake_build(path, *, device="cpu", **options):
        return Teacher(FakeEncoder(), **options)

    monkeypatch.setattr(script, "build_teacher", fake_build)
    monkeypatch.setattr("compress_ecg.teacher.build_teacher", fake_build)
    pretrained = tmp_path / "ecg_fm.pt"
    pretrained.write_bytes(b"stand-in")

    def run(run_id: str, *extra: str) -> None:
        script.main([
            "--data-dir", str(holdout_dir), "--pretrained", str(pretrained), "--run-id", run_id,
            "--output-dir", str(tmp_path / "checkpoints" / run_id), "--runs-root", str(tmp_path / "runs" / run_id),
            "--max-epochs", "1", *extra,
        ])

    return run


def read_json(path: Path) -> dict:
    """Read one of the JSON records a run wrote."""
    return json.loads(path.read_text())


def test_holdout_run_trains_on_fit_and_scores_dev_without_opening_validation(run_teacher, holdout_dir, tmp_path):
    # Without its arrays, any attempt to open Validation fails.
    (holdout_dir / "EchoNext_val_waveforms.npy").unlink()
    (holdout_dir / "EchoNext_val_tabular_features.npy").unlink()
    run_teacher("pilot", "--holdout-fraction", "0.5", "--holdout-seed", "0", *COUNT_TRAINING_ECGS)
    assert sorted(folder.name for folder in (tmp_path / "runs" / "pilot" / "teacher").iterdir()) == ["dev"]
    dev = pd.read_csv(tmp_path / "runs" / "pilot" / "teacher" / "dev" / "metadata.csv")
    assert len(dev) == 2
    assert dev["patient_key"].nunique() == 2
    assert set(dev["row_id"]) <= {1, 9, 17, 21}
    assert dev["split"].tolist() == ["dev", "dev"]
    log = pd.read_csv(tmp_path / "checkpoints" / "pilot" / "log.csv")
    assert log["step"].tolist() == list(range(1, 7 - sum(ECGS_OF[patient] for patient in dev["patient_key"])))
    result = read_json(tmp_path / "checkpoints" / "pilot" / "result.json")
    assert (result["scored_on"], result["scored"]) == ("dev", ["dev"])
    settings = read_json(tmp_path / "checkpoints" / "pilot" / "config.json")["settings"]
    assert (settings["holdout_fraction"], settings["holdout_seed"]) == (0.5, 0)


def test_run_without_a_holdout_trains_on_train_and_scores_validation(run_teacher, tmp_path):
    run_teacher("plain", *COUNT_TRAINING_ECGS)
    assert sorted(folder.name for folder in (tmp_path / "runs" / "plain" / "teacher").iterdir()) == ["train", "val"]
    assert pd.read_csv(tmp_path / "checkpoints" / "plain" / "log.csv")["step"].tolist() == [1, 2, 3, 4, 5, 6]
    result = read_json(tmp_path / "checkpoints" / "plain" / "result.json")
    assert (result["scored_on"], result["scored"]) == ("val", ["train", "val"])


def test_holdout_run_scores_validation_too_when_asked(run_teacher, tmp_path):
    run_teacher("both", "--holdout-fraction", "0.5", "--score-splits", "dev", "val")
    assert sorted(folder.name for folder in (tmp_path / "runs" / "both" / "teacher").iterdir()) == ["dev", "val"]
    assert read_json(tmp_path / "checkpoints" / "both" / "result.json")["scored_on"] == "dev"


@pytest.mark.parametrize("extra", [
    ("--score-splits", "dev"), ("--holdout-fraction", "1.0"), ("--holdout-fraction", "-0.2"),
])
def test_invalid_holdout_arguments_stop_before_anything_is_written(run_teacher, tmp_path, extra):
    with pytest.raises(SystemExit):
        run_teacher("bad", *extra)
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "runs").exists()


def saved_output_layer(tmp_path: Path, run_id: str) -> torch.Tensor:
    """Return the output layer weights in the checkpoint a run saved."""
    checkpoint = torch.load(tmp_path / "checkpoints" / run_id / "best.pt", weights_only=True)
    return checkpoint["state_dict"]["head.proj.weight"]


def test_seed_sets_the_starting_output_layer_when_no_probe_head_is_given(run_teacher, tmp_path):
    # A learning rate of 0 keeps the output layer at its starting weights.
    run_teacher("first", "--seed", "7", "--head-lr", "0")
    run_teacher("again", "--seed", "7", "--head-lr", "0")
    run_teacher("other", "--seed", "8", "--head-lr", "0")
    assert torch.equal(saved_output_layer(tmp_path, "first"), saved_output_layer(tmp_path, "again"))
    assert not torch.equal(saved_output_layer(tmp_path, "first"), saved_output_layer(tmp_path, "other"))


def save_probe_head(folder: Path) -> str:
    """Save a new output layer the way run_teacher_probe.py does, and return its path."""
    folder.mkdir(parents=True)
    torch.save(TeacherHead().state_dict(), folder / "rescaled_lr0.001.pt")
    return str(folder / "rescaled_lr0.001.pt")


def test_holdout_run_refuses_an_output_layer_fitted_on_every_training_patient(run_teacher, tmp_path):
    whole_split_head = save_probe_head(tmp_path / "probe")
    with pytest.raises(ValueError, match="_ho0.5_hs0"):
        run_teacher("leak", "--holdout-fraction", "0.5", "--probe-head", whole_split_head)
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "runs").exists()


def test_holdout_run_starts_from_the_output_layer_of_the_same_holdout(run_teacher, tmp_path):
    holdout_head = save_probe_head(tmp_path / "probe_ho0.5_hs0")
    run_teacher("clean", "--holdout-fraction", "0.5", "--probe-head", holdout_head, "--head-lr", "0")
    starting_weights = torch.load(holdout_head, weights_only=True)["proj.weight"]
    assert torch.equal(saved_output_layer(tmp_path, "clean"), starting_weights)


def test_run_with_a_missing_output_layer_stops_before_anything_is_written(run_teacher, tmp_path):
    with pytest.raises(FileNotFoundError, match="run_teacher_probe.py"):
        run_teacher("missing", "--probe-head", str(tmp_path / "probe" / "rescaled_lr0.001.pt"))
    assert not (tmp_path / "checkpoints").exists()
