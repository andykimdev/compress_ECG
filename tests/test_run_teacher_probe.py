"""Run scripts/run_teacher_probe.py end to end on the small dataset, with a stand-in for ECG-FM."""

import numpy as np
import pandas as pd
import pytest
import torch

from compress_ecg.data import LABEL_COLUMNS
from compress_ecg.teacher import Teacher, TeacherHead
from test_data import edit_meta
from test_teacher import FakeEncoder


@pytest.fixture
def run_probe(holdout_dir, tmp_path, load_script):
    """
    Return a function that runs the script with extra arguments.

    ECG-FM is replaced by the fake encoder. Features go to tmp_path/features, output layers to
    tmp_path/probe, and scores to tmp_path/probe.csv, each with the holdout tag when there is one.
    """
    script = load_script("run_teacher_probe")
    script.build_teacher = lambda path, device="cpu": Teacher(FakeEncoder())

    def run(*extra: str) -> None:
        script.main([
            "--data-dir", str(holdout_dir), "--features-dir", str(tmp_path / "features"),
            "--heads-dir", str(tmp_path / "probe"), "--results", str(tmp_path / "probe.csv"),
            "--learning-rates", "0.01", *extra,
        ])

    return run


def names_in(folder) -> list[str]:
    """List the file and folder names inside a folder, sorted."""
    return sorted(path.name for path in folder.iterdir())


def saved_row_ids(path) -> list[int]:
    """Return the row IDs of the ECGs whose features a file holds."""
    with np.load(path) as saved:
        return saved["row_id"].tolist()


def test_holdout_probe_uses_only_fit_and_dev_and_never_opens_validation(run_probe, holdout_dir, tmp_path):
    # Without its arrays, any attempt to open Validation fails.
    (holdout_dir / "EchoNext_val_waveforms.npy").unlink()
    (holdout_dir / "EchoNext_val_tabular_features.npy").unlink()
    run_probe("--holdout-fraction", "0.5", "--holdout-seed", "0")
    # Holdout seed 0 holds out patients c and d. Their most recent ECGs are CSV rows 17 and 21,
    # and the other patients own rows 1, 5, 9. No other ECG had its features computed.
    assert names_in(tmp_path / "features") == ["dev_ho0.5_hs0.npz", "fit_ho0.5_hs0.npz"]
    assert saved_row_ids(tmp_path / "features" / "fit_ho0.5_hs0.npz") == [1, 5, 9]
    assert saved_row_ids(tmp_path / "features" / "dev_ho0.5_hs0.npz") == [17, 21]


def test_holdout_probe_is_saved_apart_from_the_whole_split_probe(run_probe, tmp_path):
    run_probe("--holdout-fraction", "0.5", "--holdout-seed", "0")
    assert names_in(tmp_path / "probe_ho0.5_hs0") == ["raw_lr0.01.pt", "rescaled_lr0.01.pt"]
    assert not (tmp_path / "probe").exists()
    assert not (tmp_path / "probe.csv").exists()
    table = pd.read_csv(tmp_path / "probe_ho0.5_hs0.csv")
    assert table[["model", "picked_on"]].to_dict("records") == [
        {"model": "probe_rescaled", "picked_on": "dev"}, {"model": "probe_raw", "picked_on": "dev"},
    ]
    saved = torch.load(tmp_path / "probe_ho0.5_hs0" / "rescaled_lr0.01.pt", weights_only=True)
    assert set(saved) == set(TeacherHead().state_dict())


def test_holdout_probe_reports_dev_scores_by_name(run_probe, capsys):
    run_probe("--holdout-fraction", "0.5", "--holdout-seed", "0")
    printed = capsys.readouterr().out
    assert printed.count("dev SHD AUROC") == 2
    assert "Validation" not in printed


def test_probe_without_a_holdout_writes_the_same_files_and_columns_as_before(run_probe, tmp_path, capsys):
    run_probe()
    assert names_in(tmp_path / "features") == ["train.npz", "val.npz"]
    assert saved_row_ids(tmp_path / "features" / "train.npz") == [1, 5, 9, 13, 17, 21]
    assert saved_row_ids(tmp_path / "features" / "val.npz") == [2, 6, 10, 14, 18, 22]
    assert names_in(tmp_path / "probe") == ["raw_lr0.01.pt", "rescaled_lr0.01.pt"]
    table = pd.read_csv(tmp_path / "probe.csv")
    assert list(table.columns) == ["model", "learning_rate", "seed", "best_epoch", "epochs_run", *LABEL_COLUMNS]
    assert table["model"].tolist() == ["probe_rescaled", "probe_raw"]
    assert table["seed"].tolist() == [42, 42]
    assert capsys.readouterr().out.count("Validation SHD AUROC") == 2


def test_each_seed_gets_its_own_row_and_its_own_output_layer(run_probe, tmp_path):
    run_probe("--seeds", "42", "43")
    table = pd.read_csv(tmp_path / "probe.csv")
    assert table[["model", "seed"]].values.tolist() == [
        ["probe_rescaled", 42], ["probe_rescaled", 43], ["probe_raw", 42], ["probe_raw", 43],
    ]
    # Seed 42 keeps the plain name, because fine-tuning starts from that file.
    assert names_in(tmp_path / "probe") == [
        "raw_lr0.01.pt", "raw_lr0.01_s43.pt", "rescaled_lr0.01.pt", "rescaled_lr0.01_s43.pt",
    ]


def test_a_seed_changes_the_fitted_layer_and_the_default_seed_is_42(run_probe, tmp_path):
    run_probe("--seeds", "42", "43")
    seed_42 = torch.load(tmp_path / "probe" / "rescaled_lr0.01.pt", weights_only=True)
    seed_43 = torch.load(tmp_path / "probe" / "rescaled_lr0.01_s43.pt", weights_only=True)
    assert any(not torch.equal(seed_42[name], seed_43[name]) for name in seed_42)

    run_probe("--heads-dir", str(tmp_path / "default"))
    default = torch.load(tmp_path / "default" / "rescaled_lr0.01.pt", weights_only=True)
    assert all(torch.equal(seed_42[name], default[name]) for name in seed_42)


def test_another_holdout_seed_gets_its_own_patients_and_files(run_probe, tmp_path):
    run_probe("--holdout-fraction", "0.5", "--holdout-seed", "9")
    # Holdout seed 9 holds out patients b and d, whose most recent ECGs are CSV rows 9 and 21.
    assert names_in(tmp_path / "features") == ["dev_ho0.5_hs9.npz", "fit_ho0.5_hs9.npz"]
    assert saved_row_ids(tmp_path / "features" / "dev_ho0.5_hs9.npz") == [9, 21]
    assert saved_row_ids(tmp_path / "features" / "fit_ho0.5_hs9.npz") == [1, 13, 17]
    assert names_in(tmp_path / "probe_ho0.5_hs9") == ["raw_lr0.01.pt", "rescaled_lr0.01.pt"]
    assert (tmp_path / "probe_ho0.5_hs9.csv").is_file()


@pytest.mark.parametrize("fraction", ["1.0", "-0.2"])
def test_probe_with_an_invalid_holdout_fraction_stops_before_anything_is_written(run_probe, tmp_path, fraction):
    with pytest.raises(SystemExit):
        run_probe("--holdout-fraction", fraction)
    assert names_in(tmp_path) == ["data"]


def test_subsampled_probe_fits_on_a_share_of_the_fit_patients_and_saves_apart(run_probe, holdout_dir, tmp_path):
    # Patient b loses its SHD label, so fit patients a and b are both without SHD and half of them is 1.
    edit_meta(holdout_dir, lambda meta: meta.loc.__setitem__((5, list(LABEL_COLUMNS)), 0))
    run_probe("--holdout-fraction", "0.5", "--patient-fraction", "0.5", "--patient-seed", "0")
    # Dev features do not depend on the share, so their file keeps the holdout name alone.
    assert names_in(tmp_path / "features") == ["dev_ho0.5_hs0.npz", "fit_ho0.5_hs0_pf0.5_ps0.npz"]
    assert saved_row_ids(tmp_path / "features" / "fit_ho0.5_hs0_pf0.5_ps0.npz") in ([1], [5, 9])
    assert saved_row_ids(tmp_path / "features" / "dev_ho0.5_hs0.npz") == [17, 21]
    assert names_in(tmp_path / "probe_ho0.5_hs0_pf0.5_ps0") == ["raw_lr0.01.pt", "rescaled_lr0.01.pt"]
    assert (tmp_path / "probe_ho0.5_hs0_pf0.5_ps0.csv").is_file()
    assert not (tmp_path / "probe_ho0.5_hs0").exists()


@pytest.mark.parametrize("extra", [("--patient-fraction", "0.5"), ("--holdout-fraction", "0.5", "--patient-fraction", "0")])
def test_probe_with_an_invalid_patient_fraction_stops_before_anything_is_written(run_probe, tmp_path, extra):
    with pytest.raises(SystemExit):
        run_probe(*extra)
    assert names_in(tmp_path) == ["data"]
