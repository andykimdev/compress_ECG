import json

import numpy as np
import pytest
import torch
from torch import nn

from compress_ecg.data import EchoNextDataset, LABEL_COLUMNS
from compress_ecg.predictions import aligned_logits, load_run, predict, run_dir, run_manifest, save_run, write_run
from compress_ecg.profiling import synthetic_inputs


class RowDependent(nn.Module):
    """Give every ECG different logits, so rows that swap places are noticed."""

    def forward(self, waveform, tabular):
        return tabular[:, :1] * torch.arange(1., 13., device=tabular.device)


def test_predict_saves_logits_with_matching_probabilities(data_dir):
    dataset = EchoNextDataset(data_dir, "val")
    outputs = predict(RowDependent(), dataset, batch_size=2)
    tabular = np.load(dataset.tabular_path)
    expected = tabular[:, :1] * np.arange(1., 13.)
    np.testing.assert_allclose(outputs["logits"], expected, rtol=1e-6)
    np.testing.assert_allclose(outputs["probabilities"], 1 / (1 + np.exp(-expected)), rtol=1e-6)


def test_model_with_wrong_output_shape_is_rejected(data_dir):
    class ElevenOutputs(nn.Module):
        def forward(self, waveform, tabular):
            return torch.zeros(len(waveform), 11)

    with pytest.raises(ValueError, match=r"\[batch,12\]"):
        predict(ElevenOutputs(), EchoNextDataset(data_dir, "val"))


def test_written_run_loads_back_unchanged(data_dir, tmp_path):
    dataset = EchoNextDataset(data_dir, "val")
    folder = write_run(RowDependent(), "student", dataset, runs_root=tmp_path, batch_size=4, seed=9)
    assert folder == (tmp_path / "student" / "val").resolve()
    run = load_run(folder)
    np.testing.assert_array_equal(run.arrays["logits"], predict(RowDependent(), dataset)["logits"])
    assert run.metadata["row_id"].tolist() == [2, 6, 10, 14, 18, 22]
    assert run.manifest["model"] == "student"
    assert run.manifest["seed"] == 9
    assert set(run.metrics) == set(LABEL_COLUMNS)


def test_second_write_to_same_run_is_refused(data_dir, tmp_path):
    dataset = EchoNextDataset(data_dir, "val")
    write_run(RowDependent(), "student", dataset, runs_root=tmp_path)
    with pytest.raises(FileExistsError):
        write_run(RowDependent(), "student", dataset, runs_root=tmp_path)


def test_failed_artifact_write_leaves_run_retryable(data_dir, tmp_path):
    dataset = EchoNextDataset(data_dir, "val")
    destination = tmp_path / "run"
    destination.mkdir()
    outputs = predict(RowDependent(), dataset)
    manifest = run_manifest("student", dataset)

    def fail(staging):
        (staging / "checkpoint").write_text("partial")
        raise OSError("checkpoint failed")

    with pytest.raises(OSError, match="checkpoint failed"):
        save_run(destination, outputs, dataset.selected_metadata, manifest, write_artifacts=fail)
    assert not list(destination.iterdir())
    save_run(destination, outputs, dataset.selected_metadata, manifest)
    assert load_run(destination).manifest["n_records"] == len(dataset)


def test_load_rejects_predictions_that_do_not_match_metadata(data_dir, tmp_path):
    folder = write_run(RowDependent(), "student", EchoNextDataset(data_dir, "val"), runs_root=tmp_path)
    metadata = folder / "metadata.csv"
    lines = metadata.read_text().splitlines()
    metadata.write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n")
    with pytest.raises(ValueError, match="line up"):
        load_run(folder)


def test_aligned_logits_follow_shuffled_batch_order(data_dir, tmp_path):
    dataset = EchoNextDataset(data_dir, "val")
    run = load_run(write_run(RowDependent(), "teacher", dataset, runs_root=tmp_path))
    logits = aligned_logits(run, torch.tensor([14, 2, 22]))
    np.testing.assert_array_equal(logits.numpy(), run.arrays["logits"][[3, 0, 5]])


@pytest.mark.parametrize("rows", [[3], [999], [2, 3]])
def test_aligned_logits_reject_rows_without_predictions(data_dir, tmp_path, rows):
    dataset = EchoNextDataset(data_dir, "val")
    run = load_run(write_run(RowDependent(), "teacher", dataset, runs_root=tmp_path))
    with pytest.raises(ValueError, match="no prediction"):
        aligned_logits(run, np.array(rows))


@pytest.mark.parametrize("model,split", [("resnet", "val"), ("student", "no_split")])
def test_run_dir_rejects_unknown_names(model, split):
    with pytest.raises(ValueError):
        run_dir(model, split)


def test_synthetic_inputs_follow_call_format_and_repeat():
    waveform, tabular = synthetic_inputs(3, seed=1)
    assert waveform.shape == (3, 12, 2500)
    assert tabular.shape == (3, 7)
    assert waveform.dtype == tabular.dtype == torch.float32
    torch.testing.assert_close(waveform, synthetic_inputs(3, seed=1)[0])
