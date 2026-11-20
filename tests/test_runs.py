import hashlib
import json

import pytest

from compress_ecg.runs import collect_runs, round_scores, settings_hash, write_result, write_run_record


def make_run(root, run_id, settings, result=None, data_dir=None):
    """Write the records of one run, finished when a result is given."""
    write_run_record(root / run_id, run_id=run_id, settings=settings, command=["run"], files={}, data_dir=data_dir)
    if result is not None:
        write_result(root / run_id, {"run_id": run_id, **result})


def test_record_holds_settings_command_and_file_hashes(data_dir, tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"abc")
    record = write_run_record(
        tmp_path / "run", run_id="r1", settings={"lr": 1e-5, "seed": 42}, command=["python", "run.py", "--lr", "1e-5"],
        files={"pretrained": weights, "probe_head": None}, data_dir=data_dir,
    )
    written = json.loads((tmp_path / "run" / "config.json").read_text())
    assert written == record
    assert written["run_id"] == "r1"
    assert written["settings"] == {"lr": 1e-5, "seed": 42}
    assert written["command"] == ["python", "run.py", "--lr", "1e-5"]
    assert written["files"] == {"pretrained": {"path": str(weights), "sha256": hashlib.sha256(b"abc").hexdigest()}}
    metadata = data_dir / "EchoNext_metadata_100k.csv"
    assert written["data"]["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert written["config_sha256"] == hashlib.sha256(b'{"lr": 1e-05, "seed": 42}').hexdigest()
    assert set(written["code"]) == {"commit", "dirty"}


def test_second_record_in_the_same_folder_is_refused(data_dir, tmp_path):
    make_run(tmp_path, "r1", {"lr": 1e-5}, data_dir=data_dir)
    with pytest.raises(FileExistsError, match="config.json"):
        make_run(tmp_path, "r1", {"lr": 3e-5}, data_dir=data_dir)


def test_settings_hash_ignores_order_and_notices_changes():
    assert settings_hash({"a": 1, "b": 2}) == settings_hash({"b": 2, "a": 1})
    assert settings_hash({"a": 1, "b": 2}) != settings_hash({"a": 1, "b": 3})


def test_table_lists_finished_runs_best_first_and_incomplete_runs_last(data_dir, tmp_path):
    make_run(tmp_path, "b", {"lr": 3e-6, "head_lr": 1e-4, "seed": 42},
             {"best_epoch": 5, "epochs_run": 7, "val_mean_auroc": 0.79, "val_shd_auroc": 0.82}, data_dir)
    make_run(tmp_path, "c", {"lr": 1e-5, "head_lr": 1e-3, "seed": 42}, None, data_dir)
    make_run(tmp_path, "a", {"lr": 1e-5, "head_lr": 1e-4, "seed": 42},
             {"best_epoch": 2, "epochs_run": 4, "val_mean_auroc": 0.81, "val_shd_auroc": 0.83}, data_dir)
    table = collect_runs(tmp_path)
    assert list(table["run_id"]) == ["a", "b", "c"]
    finished = table[table["status"] == "finished"]
    assert finished[["run_id", "lr", "head_lr", "seed", "best_epoch", "val_mean_auroc"]].to_dict("records") == [
        {"run_id": "a", "lr": 1e-5, "head_lr": 1e-4, "seed": 42, "best_epoch": 2, "val_mean_auroc": 0.81},
        {"run_id": "b", "lr": 3e-6, "head_lr": 1e-4, "seed": 42, "best_epoch": 5, "val_mean_auroc": 0.79},
    ]
    assert list(table[table["status"] == "incomplete"]["run_id"]) == ["c"]
    assert table[table["status"] == "incomplete"]["val_mean_auroc"].isna().all()


def test_table_of_a_folder_without_runs_is_empty(tmp_path):
    assert collect_runs(tmp_path / "none").empty


def test_rounding_scores_keeps_small_learning_rates(data_dir, tmp_path):
    make_run(tmp_path, "a", {"lr": 3e-6, "head_lr": 1e-4, "seed": 42},
             {"val_mean_auroc": 0.812345678, "val_shd_auroc": 0.8312349, "minutes": 12.345}, data_dir)
    row = round_scores(collect_runs(tmp_path)).iloc[0]
    assert (row["lr"], row["head_lr"]) == (3e-6, 1e-4)
    assert (row["val_mean_auroc"], row["val_shd_auroc"], row["minutes"]) == (0.8123, 0.8312, 12.3)


def test_table_shows_whether_a_run_read_the_7_numbers(data_dir, tmp_path):
    make_run(tmp_path, "a", {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "use_tabular": True}, None, data_dir)
    make_run(tmp_path, "b", {"lr": 1e-5, "head_lr": 1e-4, "seed": 42}, None, data_dir)
    table = collect_runs(tmp_path)
    assert table[["run_id", "use_tabular"]].to_dict("records") == [
        {"run_id": "a", "use_tabular": True}, {"run_id": "b", "use_tabular": None},
    ]


def test_table_shows_the_weight_averaging_settings(data_dir, tmp_path):
    settings = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42, "ema_decay": 0.999, "evals_per_epoch": 4, "select_by": "shd"}
    make_run(tmp_path, "a", settings, None, data_dir)
    table = collect_runs(tmp_path)
    assert table[["run_id", "ema_decay", "evals_per_epoch", "select_by"]].to_dict("records") == [
        {"run_id": "a", "ema_decay": 0.999, "evals_per_epoch": 4, "select_by": "shd"},
    ]


def test_table_says_which_ecgs_each_run_was_scored_on(data_dir, tmp_path):
    scores = {"val_mean_auroc": 0.8, "val_shd_auroc": 0.8}
    rates = {"lr": 1e-5, "head_lr": 1e-4, "seed": 42}
    make_run(tmp_path, "pilot", {**rates, "holdout_fraction": 0.2}, {**scores, "scored_on": "dev"}, data_dir)
    make_run(tmp_path, "final", {**rates, "holdout_fraction": 0.0}, {**scores, "scored_on": "val"}, data_dir)
    # A run from before the holdout option has neither entry. It was scored on Validation.
    make_run(tmp_path, "older", rates, scores, data_dir)
    make_run(tmp_path, "unfinished", rates, None, data_dir)
    table = collect_runs(tmp_path).set_index("run_id")
    assert table["scored_on"].to_dict() == {"pilot": "dev", "final": "val", "older": "val", "unfinished": None}
    assert table.loc[["pilot", "final"], "holdout_fraction"].to_dict() == {"pilot": 0.2, "final": 0.0}
