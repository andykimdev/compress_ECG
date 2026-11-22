"""
Convert the Student to smaller number formats.

Purpose:
- Produce FP16, post-training INT8, and quant-aware INT8 versions of the Student.
- Keep every version in the shared call format, so predictions.write_run can score it.

Special Notes:
- Owner: Role D. Every function here is still to be written.
- During Phase 1, baseline.load_mini gives a model with the same layout to test on.
- Hosted devices may only receive synthetic inputs, from profiling.synthetic_inputs.
"""

from __future__ import annotations

import copy #new: so we can copy a model w/out modifying original

from pathlib import Path

from torch import nn

from .data import EchoNextDataset

class FP16Inference(nn.Module):
    """Run an FP16 model using the team's existing input/output interface."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = copy.deepcopy(model).eval().half()

    def forward(self, waveform, tabular):
        logits = self.model(waveform.half(), tabular.half())
        return logits.float()


def to_fp16(student: nn.Module) -> nn.Module:
    """
    Convert a trained Student to 16-bit numbers.

    Args:
        student: The trained model.

    Returns:
        The FP16 model, in the shared call format.
    """
    return FP16Inference(student)


def to_int8_post_training(student: nn.Module, calibration: EchoNextDataset) -> nn.Module:
    """
    Convert a trained Student to 8-bit numbers after training.

    Args:
        student: The trained model.
        calibration: ECGs used to measure value ranges, from the Training ECGs.

    Returns:
        The INT8 model, in the shared call format.
    """
    raise NotImplementedError("Role D: convert to INT8 with calibration")


def train_int8_quant_aware(
    student: nn.Module, train: EchoNextDataset, val: EchoNextDataset, output_dir: str | Path,
    *, device: str = "cpu", seed: int = 42,
) -> nn.Module:
    """
    Fine-tune a Student while simulating 8-bit numbers, then convert it.

    Args:
        student: The trained model.
        train: The Training ECGs.
        val: The Validation ECGs, used for tuning.
        output_dir: Folder for checkpoints.
        device: Where training runs.
        seed: Random seed.

    Returns:
        The INT8 model, in the shared call format.
    """
    raise NotImplementedError("Role D: run quant-aware training")


def export_for_device(model: nn.Module, path: str | Path) -> Path:
    """
    Save a model in the format the deployment device runs.

    Args:
        model: Any of the converted models.
        path: File to write.

    Returns:
        The written path.
    """
    raise NotImplementedError("Role D: choose the device, then export for it")
