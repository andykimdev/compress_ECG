"""Run all five roles end to end on synthetic ECGs, using stand-ins for unwritten steps.

When a role's real code exists, replace its stand-in here with the real call.
"""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from compress_ecg.baseline import run_gradient_boosting
from compress_ecg.data import EchoNextDataset, LABEL_COLUMNS, SHD_INDEX
from compress_ecg.distill import train_student
from compress_ecg.predictions import aligned_logits, basic_metrics, load_run, run_dir, write_run


class TinyModel(nn.Module):
    """Stand-in for any role's model, in the shared call format."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(7, 12)

    def forward(self, waveform, tabular):
        return self.layer(tabular) + waveform.mean(dim=(1, 2), keepdim=False).unsqueeze(1)


def test_handoffs_connect_from_teacher_to_scorecard(data_dir, tmp_path):
    torch.manual_seed(0)
    train = EchoNextDataset(data_dir, "train")
    val = EchoNextDataset(data_dir, "val")
    test = EchoNextDataset(data_dir, "test")

    # Role A: baseline predictions.
    for dataset in (val, test):
        destination = run_dir("gradient_boosting", dataset.split, tmp_path)
        run_gradient_boosting(data_dir, destination, split=dataset.split)

    # Role B: frozen Teacher predictions.
    teacher = TinyModel()
    for dataset in (train, val, test):
        write_run(teacher, "teacher", dataset, runs_root=tmp_path)

    # Role C: Student learns only from the saved Teacher file.
    student = TinyModel()
    before = student.layer.weight.detach().clone()
    teacher_run = load_run(run_dir("teacher", "train", tmp_path))
    checkpoint = train_student(
        student, teacher_run, train, val, tmp_path / "checkpoints" / "student", batch_size=2, max_epochs=1,
    )
    assert checkpoint.is_file()
    assert not torch.equal(before, student.layer.weight)
    for dataset in (val, test):
        write_run(student, "student", dataset, runs_root=tmp_path)

    # Role D: each small model goes through the same call format and files.
    for name in ("student_fp16", "student_int8_post", "student_int8_qat"):
        small = copy.deepcopy(student)
        for dataset in (val, test):
            write_run(small, name, dataset, runs_root=tmp_path)

    # Role E: everything is read back from files alone.
    expected = {
        "gradient_boosting", "teacher", "student", "student_fp16", "student_int8_post", "student_int8_qat",
    }
    assert {path.parent.name for path in tmp_path.glob("*/test")} == expected
    scorecard = {}
    for name in expected:
        run = load_run(run_dir(name, "test", tmp_path))
        assert run.arrays["logits"].shape == (6, 12)
        assert run.metadata["row_id"].tolist() == test.selected_metadata["row_id"].tolist()
        assert list(run.metadata["sex"]) == list(test.selected_metadata["sex"])
        scorecard[name] = basic_metrics(run.arrays)[LABEL_COLUMNS[SHD_INDEX]]
    assert all(entry["n_observed"] == 6 and entry["n_positive"] == 3 for entry in scorecard.values())


def test_student_sees_the_teacher_prediction_for_its_own_row(data_dir, tmp_path):
    # A wrong lookup here would train the Student against another patient's outputs.
    train = EchoNextDataset(data_dir, "train")
    teacher = TinyModel().eval()
    run = load_run(write_run(teacher, "teacher", train, runs_root=tmp_path))
    loader = DataLoader(train, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(3))
    batch = next(iter(loader))
    with torch.inference_mode():
        direct = teacher(batch["waveform"], batch["tabular"])
    assert aligned_logits(run, batch["row_id"]).numpy() == pytest.approx(direct.numpy())
