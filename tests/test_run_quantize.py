"""Run scripts/run_quantize.py on the small dataset, with stand-ins for the source model."""

import copy

import pytest
import torch
from torch import nn


class TinyModel(nn.Module):
    """Stand-in for EchoNext-Mini or the Student, in the shared call format."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(7, 12)

    def forward(self, waveform, tabular):
        return self.layer(tabular)


def test_unwritten_versions_are_skipped_and_finished_ones_scored(data_dir, tmp_path, load_script, capsys):
    script = load_script("run_quantize")
    script.load_mini = lambda device="cpu": TinyModel()
    script.to_fp16 = copy.deepcopy

    script.main(["--data-dir", str(data_dir), "--runs-root", str(tmp_path / "runs")])

    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == ["student_fp16"]
    output = capsys.readouterr().out
    assert "Skipped student_int8_post: Role D: convert to INT8 with calibration" in output
    assert "Skipped student_int8_qat: Role D: run quant-aware training" in output


def test_a_failing_conversion_still_stops_the_run(data_dir, tmp_path, load_script):
    script = load_script("run_quantize")
    script.load_mini = lambda device="cpu": TinyModel()

    def broken(model):
        raise ValueError("bad conversion")

    script.to_fp16 = broken

    with pytest.raises(ValueError, match="bad conversion"):
        script.main(["--data-dir", str(data_dir), "--runs-root", str(tmp_path / "runs")])


def test_a_conversion_that_changes_its_model_leaves_the_next_one_untouched(data_dir, tmp_path, load_script):
    script = load_script("run_quantize")
    source = TinyModel()
    original = source.layer.weight.detach().clone()
    script.load_mini = lambda device="cpu": source
    seen = {}

    def zero_in_place(model):
        model.layer.weight.data.zero_()
        return model

    def record(model, calibration):
        seen["weight"] = model.layer.weight.detach().clone()
        raise NotImplementedError("stop here")

    script.to_fp16 = zero_in_place
    script.to_int8_post_training = record
    script.main(["--data-dir", str(data_dir), "--runs-root", str(tmp_path / "runs")])

    assert torch.equal(seen["weight"], original)
