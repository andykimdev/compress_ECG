from dataclasses import replace
import json
import shutil

import pandas as pd
import pytest
import torch
from torch import nn

import compress_ecg.distill as distill
from compress_ecg.data import EchoNextDataset, find_metadata_file
from compress_ecg.predictions import load_run, write_run


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(7, 12)

    def forward(self, waveform, tabular):
        return self.layer(tabular)


@pytest.fixture
def training_inputs(data_dir, tmp_path):
    train = EchoNextDataset(data_dir, "train")
    val = EchoNextDataset(data_dir, "val")
    teacher_run = load_run(write_run(
        TinyStudent(), "teacher", train, runs_root=tmp_path / "runs",
    ))
    return train, val, teacher_run


@pytest.mark.parametrize("metadata_hash", [None, "different-dataset"])
def test_teacher_metadata_must_match(training_inputs, tmp_path, metadata_hash):
    train, val, teacher_run = training_inputs
    manifest = dict(teacher_run.manifest)
    if metadata_hash is None:
        manifest.pop("metadata_sha256")
    else:
        manifest["metadata_sha256"] = metadata_hash
    teacher_run = replace(teacher_run, manifest=manifest)
    output = tmp_path / "student"

    with pytest.raises(ValueError, match="metadata_sha256"):
        distill.train_student(TinyStudent(), teacher_run, train, val, output)

    assert not output.exists()


def test_validation_metadata_must_match(training_inputs, tmp_path):
    train, _, teacher_run = training_inputs
    other_root = tmp_path / "other_data"
    shutil.copytree(train.root, other_root)
    metadata_path = find_metadata_file(other_root)
    metadata = pd.read_csv(metadata_path)
    # Preserve row IDs and valid splits while changing which patient each row represents.
    metadata["patient_key"] = "other-" + metadata["patient_key"]
    metadata.to_csv(metadata_path, index=False)
    val = EchoNextDataset(other_root, "val")
    output = tmp_path / "student"

    with pytest.raises(ValueError, match="same metadata"):
        distill.train_student(TinyStudent(), teacher_run, train, val, output)

    assert not output.exists()


@pytest.mark.parametrize("copy_validation", [False, True])
def test_matching_metadata_reaches_training(training_inputs, tmp_path, monkeypatch, copy_validation):
    train, val, teacher_run = training_inputs
    if copy_validation:
        other_root = tmp_path / "identical_data"
        shutil.copytree(train.root, other_root)
        val = EchoNextDataset(other_root, "val")

    class ReachedTraining(Exception):
        pass

    def stop_at_loss(*args, **kwargs):
        raise ReachedTraining()

    # Stop at the loss function; this tests provenance checks, not distillation.
    monkeypatch.setattr(distill, "distillation_loss", stop_at_loss)
    with pytest.raises(ReachedTraining):
        distill.train_student(TinyStudent(), teacher_run, train, val, tmp_path / "student")


def test_training_uses_notebook_settings_and_records_them(training_inputs, tmp_path):
    train, val, teacher_run = training_inputs
    student = TinyStudent()
    before = student.layer.weight.detach().clone()
    output = tmp_path / "student"
    settings = {
        "batch_size": 2, "learning_rate": 0.002, "weight_decay": 0.,
        "max_epochs": 2, "patience": 2, "min_delta": 0.,
        "temperature": 3., "hard_label_weight": 0.25, "num_workers": 0,
    }

    checkpoint = distill.train_student(student, teacher_run, train, val, output, **settings)

    history = json.loads((output / "history.json").read_text())
    saved_settings = json.loads((output / "training_config.json").read_text())
    assert len(history) == 2
    assert checkpoint.is_file()
    assert not torch.equal(before, student.layer.weight)
    for name, value in settings.items():
        assert saved_settings[name] == value
    assert saved_settings["teacher_manifest"] == teacher_run.manifest


@pytest.mark.parametrize("settings", [
    {"max_epochs": 0}, {"batch_size": 0}, {"patience": 0},
    {"learning_rate": -1.}, {"temperature": 0.}, {"hard_label_weight": 1.1},
])
def test_invalid_training_settings_fail_before_writing(training_inputs, tmp_path, settings):
    train, val, teacher_run = training_inputs
    output = tmp_path / "student"
    with pytest.raises(ValueError):
        distill.train_student(TinyStudent(), teacher_run, train, val, output, **settings)
    assert not output.exists()


def test_distillation_gradient_combines_binary_targets_and_freezes_teacher():
    student = torch.linspace(-3, 3, 24, dtype=torch.float64).reshape(2, 12).requires_grad_()
    teacher = torch.linspace(2, -2, 24, dtype=torch.float64).reshape(2, 12).requires_grad_()
    labels = (torch.arange(24).reshape(2, 12) % 2).double()
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, 0] = False
    labels[~mask] = float("nan")
    temperature, alpha = 2.5, 0.3

    loss = distill.distillation_loss(
        student, teacher, labels, mask,
        temperature=temperature, hard_label_weight=alpha,
    )
    loss.backward()

    # Analytical derivatives check masking, averaging, mixture, and T^2 scaling.
    hard_gradient = torch.zeros_like(student)
    hard_gradient[mask] = (student.detach().sigmoid()[mask] - labels[mask]) / mask.sum()
    soft_gradient = temperature * (
        (student.detach() / temperature).sigmoid() - (teacher.detach() / temperature).sigmoid()
    ) / student.numel()
    torch.testing.assert_close(student.grad, alpha * hard_gradient + (1 - alpha) * soft_gradient)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert teacher.grad is None


def test_hard_only_loss_ignores_masked_labels():
    student = torch.zeros(1, 12, requires_grad=True)
    teacher = torch.full_like(student, -5)
    labels = torch.full_like(student, float("nan"))
    mask = torch.zeros_like(student, dtype=torch.bool)
    labels[0, 0] = 1
    mask[0, 0] = True

    loss = distill.distillation_loss(
        student, teacher, labels, mask, temperature=2, hard_label_weight=1,
    )
    loss.backward()

    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.))))
    assert student.grad[0, 0].item() == pytest.approx(-0.5)
    assert torch.count_nonzero(student.grad[~mask]) == 0


@pytest.mark.parametrize("label_dtype", [torch.int64, torch.bool])
def test_nonfloating_labels_match_float_targets(label_dtype):
    labels = (torch.arange(24).reshape(2, 12) % 2).to(label_dtype)
    teacher = torch.zeros(2, 12)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, 0] = False
    student = torch.linspace(-2, 2, 24).reshape(2, 12).requires_grad_()
    reference_student = student.detach().clone().requires_grad_()

    loss = distill.distillation_loss(
        student, teacher, labels, mask, temperature=2, hard_label_weight=0.5,
    )
    reference_loss = distill.distillation_loss(
        reference_student, teacher, labels.float(), mask, temperature=2, hard_label_weight=0.5,
    )
    loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(student.grad, reference_student.grad)


@pytest.mark.parametrize("alpha", [0., 0.5, 1.])
def test_no_observed_labels_keeps_teacher_supervision(alpha):
    student = torch.zeros(1, 12, requires_grad=True)
    teacher = torch.full_like(student, 4)
    labels = torch.full_like(student, float("nan"))
    mask = torch.zeros_like(student, dtype=torch.bool)

    loss = distill.distillation_loss(
        student, teacher, labels, mask, temperature=2, hard_label_weight=alpha,
    )
    loss.backward()

    assert loss.item() == pytest.approx((1 - alpha) * 4 * float(torch.log(torch.tensor(2.))))
    assert torch.isfinite(student.grad).all()
    if alpha < 1:
        assert (student.grad < 0).all()  # Gradient descent moves toward the positive teacher.
    else:
        assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature_is_rejected(temperature):
    logits = torch.zeros(1, 12)
    with pytest.raises(ValueError, match="temperature"):
        distill.distillation_loss(
            logits, logits, logits, logits.bool(), temperature=temperature, hard_label_weight=0.5,
        )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_loss_weight_is_rejected(alpha):
    logits = torch.zeros(1, 12)
    with pytest.raises(ValueError, match="hard_label_weight"):
        distill.distillation_loss(
            logits, logits, logits, logits.bool(), temperature=2, hard_label_weight=alpha,
        )


def test_loss_rejects_broadcasting_teacher_targets():
    logits = torch.zeros(2, 12)
    with pytest.raises(ValueError, match="match student_logits shape"):
        distill.distillation_loss(
            logits, logits[:1], logits, logits.bool(), temperature=2, hard_label_weight=0.5,
        )
