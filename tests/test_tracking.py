"""Check what a Student run sends to the experiment tracker, without the network."""

import sys
import types

from compress_ecg.tracking import WandbRun, epoch_log, wandb_config


def test_wandb_config_keeps_the_teacher_identity_and_drops_the_manifest():
    settings = {
        "seed": 42, "temperature": 2.0, "hard_label_weight": 0.5,
        "teacher_manifest": {
            "model": "teacher", "run_id": "soup6", "split": "train", "checkpoint_sha256": "1db5",
            "n_records": 72475, "label_columns": ["a", "b"],
        },
    }

    assert wandb_config(settings) == {
        "seed": 42, "temperature": 2.0, "hard_label_weight": 0.5,
        "teacher_model": "teacher", "teacher_run_id": "soup6", "teacher_split": "train",
        "teacher_checkpoint_sha256": "1db5", "teacher_records": 72475,
    }


def test_wandb_config_marks_missing_teacher_fields_as_none():
    settings = {"seed": 1, "teacher_manifest": {"model": "mini", "split": "train"}}

    assert wandb_config(settings) == {
        "seed": 1, "teacher_model": "mini", "teacher_run_id": None, "teacher_split": "train",
        "teacher_checkpoint_sha256": None, "teacher_records": None,
    }


def test_epoch_log_drops_the_epoch_and_empty_values():
    record = {"epoch": 3, "train_loss": 0.5, "val_shd_auc": 0.8, "peak_gpu_memory_bytes": None}

    assert epoch_log(record) == {"train_loss": 0.5, "val_shd_auc": 0.8}


def test_teacher_group_drops_only_the_seed():
    from compress_ecg.tracking import teacher_group

    assert teacher_group("lr1e-05_hlr0.0001_do0.5_s42") == "lr1e-05_hlr0.0001_do0.5"
    assert teacher_group("soup6_lr1e-05_hlr0.0001") == "soup6_lr1e-05_hlr0.0001"


def test_teacher_history_turns_log_rows_into_numbers():
    from compress_ecg.tracking import teacher_history

    rows = [
        {"epoch": "1", "step": "283", "train_loss": "0.32781", "val_shd_auroc": "0.83792", "seconds": "42.3"},
        {"epoch": "1", "step": "566", "train_loss": "", "val_shd_auroc": "0.84494", "seconds": "40.7"},
    ]

    assert teacher_history(rows) == [
        {"epoch": 1.0, "train_step": 283.0, "train_loss": 0.32781, "val_shd_auroc": 0.83792, "seconds": 42.3},
        {"epoch": 1.0, "train_step": 566.0, "val_shd_auroc": 0.84494, "seconds": 40.7},
    ]


def test_teacher_config_keeps_settings_and_identity():
    from compress_ecg.tracking import teacher_config

    config = {
        "run_id": "lr1e-05_s42", "started_at": "2026-11-19T18:32:08+00:00", "command": ["x"],
        "settings": {"lr": 1e-05, "seed": 42, "checkpoint": None},
        "code": {"commit": "428a049", "dirty": False}, "machine": {"gpu": "RTX PRO 6000", "host": "h"},
    }

    assert teacher_config(config) == {
        "lr": 1e-05, "seed": 42, "checkpoint": None, "started_at": "2026-11-19T18:32:08+00:00",
        "code_commit": "428a049", "gpu": "RTX PRO 6000",
    }


def test_teacher_summary_names_the_saved_scores_as_best():
    from compress_ecg.tracking import teacher_summary

    result = {
        "run_id": "lr1e-05_s42", "best_epoch": 2, "epochs_run": 4, "val_mean_auroc": 0.8195,
        "val_shd_auroc": 0.83505, "minutes": 12.5, "scored": ["val"],
    }

    assert teacher_summary(result) == {
        "best_epoch": 2, "epochs_run": 4, "best_val_mean_auroc": 0.8195, "best_val_shd_auroc": 0.83505,
        "minutes": 12.5,
    }


class FakeRun:
    """A W&B run that keeps its summary and closed state in memory."""

    entity, project, id = "team", "compress-ecg", "abc123"

    def __init__(self):
        self.summary = {}
        self.closed = False

    def finish(self):
        self.closed = True


def fake_wandb(run: FakeRun):
    """Return a stand-in wandb module whose public API is unreachable, as on an offline machine."""

    class NoApi:
        def __init__(self):
            raise ConnectionError("The W&B server cannot be reached")

    return types.SimpleNamespace(init=lambda **_: run, Api=NoApi)


def test_offline_run_keeps_final_scores_without_the_server(monkeypatch, tmp_path):
    run = FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run))
    tracker = WandbRun(
        project="compress-ecg", entity=None, name="student_distill_s42", group=None, mode="offline",
        settings={"seed": 42}, directory=tmp_path / "wandb",
    )

    tracker.finish({"best_val_shd_auroc": 0.83})

    assert run.summary == {"best_val_shd_auroc": 0.83}
    assert run.closed
