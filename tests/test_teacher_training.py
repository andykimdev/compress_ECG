"""Check how the Teacher is trained, scored, and reloaded, mostly with a stand-in for ECG-FM that has weights."""

import csv

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX, EchoNextDataset
from compress_ecg.profiling import synthetic_inputs
from compress_ecg.teacher import Teacher, TeacherHead, load_frozen_teacher, restore_teacher, train_teacher
from test_teacher import FakeEncoder


class TrainableEncoder(torch.nn.Module):
    """
    Stands in for ECG-FM with weights to train.

    Like ECG-FM it has a convolution part that runs without gradients when feature_grad_mult
    is 0, dropout that is active in training mode only, and a layer after both. It also notes
    whether it was in training mode at every call that tracked gradients, which is every update.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.feature_grad_mult = 1.0
        self.feature_extractor = torch.nn.Conv1d(12, 4, kernel_size=250, stride=250)
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.Linear(4, 768)
        self.update_modes = []

    def extract_features(self, source, padding_mask, mask):
        if torch.is_grad_enabled():
            self.update_modes.append(self.training)
        if self.feature_grad_mult > 0:
            steps = self.feature_extractor(source)
        else:
            with torch.no_grad():
                steps = self.feature_extractor(source)
        return {"x": self.layers(self.dropout(torch.relu(steps).transpose(1, 2)))}


def splits(data_dir) -> tuple[EchoNextDataset, EchoNextDataset]:
    """Open the Training and Validation ECGs of the small dataset."""
    return EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val")


def write_labels(data_dir, split: str, labels) -> None:
    """Replace the labels of one split. labels holds one row per ECG and one value or 12 per row."""
    path = data_dir / "EchoNext_metadata_100k.csv"
    metadata = pd.read_csv(path)
    rows = metadata.index[metadata["split"].eq(split)]
    values = np.asarray(labels, dtype=float).reshape(len(rows), -1)
    metadata.loc[rows, list(LABEL_COLUMNS)] = np.broadcast_to(values, (len(rows), 12))
    metadata.to_csv(path, index=False)


def read_log(folder) -> list[dict]:
    """Read the log.csv that train_teacher wrote in a run folder."""
    with (folder / "log.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_output_layer_rescales_the_features_before_weighting_them():
    head = TeacherHead().eval()
    head.feature_mean.fill_(2.0)
    head.feature_std.fill_(4.0)
    with torch.no_grad():
        head.proj.weight.fill_(1.0)
        # Every feature is 6, which rescales to (6 - 2) / 4 = 1, and the 768 weights of 1 add them up.
        logits = head(torch.full((2, 768), 6.0))
    torch.testing.assert_close(logits, torch.full((2, 12), 768.0))


def test_prediction_of_an_ecg_does_not_depend_on_the_other_ecgs_in_its_batch():
    torch.manual_seed(0)
    teacher = Teacher(TrainableEncoder()).eval()
    waveform, tabular = synthetic_inputs(3)
    with torch.no_grad():
        together = teacher(waveform, tabular)
        alone = torch.cat([teacher(waveform[i:i + 1], tabular[i:i + 1]) for i in range(3)])
    # The ECGs must give different logits, or mixing them up would go unnoticed.
    assert (together[0] - together[1]).abs().max() > 1e-3
    torch.testing.assert_close(together, alone, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("half, readings", [("first", slice(0, 1249)), ("second", slice(1251, 2500))])
def test_prediction_changes_when_only_one_half_of_the_ecg_changes(half, readings):
    torch.manual_seed(0)
    teacher = Teacher(TrainableEncoder()).eval()
    waveform, tabular = synthetic_inputs(1)
    changed = waveform.clone()
    # Reversing these readings in time keeps each lead's mean and spread, so the 5-second
    # window of the other half reaches ECG-FM unchanged and only this half's window differs.
    changed[..., readings] = waveform[..., readings].flip(-1)
    with torch.no_grad():
        difference = (teacher(waveform, tabular) - teacher(changed, tabular)).abs().max()
    assert difference > 1e-3


def parts_changed_by_one_update(data_dir, tmp_path, **rates) -> dict[str, bool]:
    """Train for one update and report, for each group of weights, whether any value changed."""
    torch.manual_seed(0)
    teacher = Teacher(TrainableEncoder())
    before = {name: weights.detach().clone() for name, weights in teacher.named_parameters()}
    train_teacher(teacher, *splits(data_dir), tmp_path / "t", max_epochs=1, **rates)
    changed = {name: not torch.equal(weights, before[name]) for name, weights in teacher.named_parameters()}
    return {
        "convolution": changed["encoder.feature_extractor.weight"] or changed["encoder.feature_extractor.bias"],
        "later_layers": changed["encoder.layers.weight"] and changed["encoder.layers.bias"],
        "output_layer": changed["head.proj.weight"] and changed["head.proj.bias"],
    }


def test_one_update_trains_ecg_fm_and_the_output_layer_but_not_the_convolution_part(data_dir, tmp_path):
    assert parts_changed_by_one_update(data_dir, tmp_path) == {
        "convolution": False, "later_layers": True, "output_layer": True,
    }


@pytest.mark.parametrize("rates, expected", [
    ({"lr": 0.0, "head_lr": 1e-4}, {"convolution": False, "later_layers": False, "output_layer": True}),
    ({"lr": 1e-5, "head_lr": 0.0}, {"convolution": False, "later_layers": True, "output_layer": False}),
])
def test_each_learning_rate_reaches_its_own_part_of_the_model(rates, expected, data_dir, tmp_path):
    # A rate of 0 leaves weights exactly as they were, so it shows which part each rate drives.
    assert parts_changed_by_one_update(data_dir, tmp_path, **rates) == expected


def test_every_update_runs_in_training_mode_even_after_a_scoring(data_dir, tmp_path):
    # The model arrives in evaluation mode, and each scoring switches it back to that mode.
    teacher = Teacher(TrainableEncoder()).eval()
    # 6 Training ECGs in batches of 2 give 3 updates per epoch, each followed by a scoring.
    train_teacher(
        teacher, *splits(data_dir), tmp_path / "t", batch_size=2, max_epochs=2, patience=10, evals_per_epoch=3,
    )
    assert teacher.encoder.update_modes == [True] * 6


def test_training_starts_from_the_probe_head_it_is_given(data_dir, tmp_path):
    probe = TeacherHead()
    probe.feature_mean.fill_(2.0)
    probe.feature_std.fill_(4.0)
    torch.nn.init.constant_(probe.proj.weight, 0.25)
    torch.save(probe.state_dict(), tmp_path / "probe.pt")
    teacher = Teacher(FakeEncoder())
    # A learning rate of 0 keeps the output layer at its starting values.
    train_teacher(
        teacher, *splits(data_dir), tmp_path / "t", max_epochs=1, head_lr=0.0, probe_head=tmp_path / "probe.pt",
    )
    assert torch.equal(teacher.head.proj.weight, torch.full((12, 768), 0.25))
    assert torch.equal(teacher.head.feature_mean, torch.full((768,), 2.0))
    assert torch.equal(teacher.head.feature_std, torch.full((768,), 4.0))


def test_model_learns_from_the_training_labels_and_not_the_validation_labels(data_dir, tmp_path):
    # The 7 numbers of the fixture rise from one ECG to the next. In Training the last 3 ECGs
    # are positive, and in Validation the first 3 are, so a model that learned the Training
    # rule ranks every Validation ECG the wrong way round.
    write_labels(data_dir, "train", [0, 0, 0, 1, 1, 1])
    write_labels(data_dir, "val", [1, 1, 1, 0, 0, 0])
    train_teacher(Teacher(FakeEncoder(), use_tabular=True), *splits(data_dir), tmp_path / "t", max_epochs=1)
    row = read_log(tmp_path / "t")[0]
    assert (float(row["val_mean_auroc"]), float(row["val_shd_auroc"])) == (0.0, 0.0)


def test_logged_scores_match_the_saved_checkpoint_scored_from_scratch(data_dir, tmp_path):
    # Every output gets its own Validation labels, so the 12 AUROCs differ from one another.
    labels = np.random.default_rng(3).permuted(np.tile([0, 0, 0, 1, 1, 1], (12, 1)), axis=1).T
    write_labels(data_dir, "val", labels)
    train, val = splits(data_dir)
    best = train_teacher(Teacher(FakeEncoder(), use_tabular=True), train, val, tmp_path / "t", max_epochs=1)
    reloaded = restore_teacher(Teacher(FakeEncoder(), use_tabular=True), best).eval()
    items = [val[index] for index in range(len(val))]
    with torch.no_grad():
        waveform, tabular = (torch.stack([item[name] for item in items]) for name in ("waveform", "tabular"))
        logits = reloaded(waveform, tabular)
    aurocs = [roc_auc_score(labels[:, column], logits[:, column].numpy()) for column in range(12)]
    assert len(set(np.round(aurocs, 5))) > 1
    row = read_log(tmp_path / "t")[0]
    assert float(row["val_mean_auroc"]) == pytest.approx(np.mean(aurocs), abs=1e-5)
    assert float(row["val_shd_auroc"]) == pytest.approx(aurocs[SHD_INDEX], abs=1e-5)


def test_frozen_teacher_runs_with_dropout_switched_off(data_dir, tmp_path, monkeypatch):
    trained = Teacher(TrainableEncoder())
    best = train_teacher(trained, *splits(data_dir), tmp_path / "t", max_epochs=1)
    monkeypatch.setattr(
        "compress_ecg.teacher.build_teacher", lambda path, **options: Teacher(TrainableEncoder(), **options),
    )
    frozen = load_frozen_teacher(best)
    inputs = synthetic_inputs(4)
    with torch.no_grad():
        expected = trained.eval()(*inputs)
    # With dropout left on, the encoder would blank half of its values at random on every call.
    torch.testing.assert_close(frozen(*inputs), expected)
    torch.testing.assert_close(frozen(*inputs), expected)
