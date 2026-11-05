"""
Fine-tune ECG-FM into the Teacher.

Purpose:
- Add a 12-output layer to ECG-FM and fine-tune it with mixed precision.
- Produce a frozen Teacher that follows the shared call format.

Special Notes:
- Owner: Role B. Every function here is still to be written.
- Any input conversion ECG-FM needs, such as resampling, stays inside the Teacher.
- predictions.write_run saves the Teacher's outputs once the model follows the call format.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from .data import EchoNextDataset


def build_teacher(pretrained_path: str | Path, *, device: str = "cpu") -> nn.Module:
    """
    Load ECG-FM and add a new output layer.

    Args:
        pretrained_path: The ECG-FM checkpoint from scripts/download_weights.sh.
        device: Where the model is placed.

    Returns:
        A model where model(waveform, tabular) returns logits [batch, 12]. It may ignore tabular.
    """
    raise NotImplementedError("Role B: wrap ECG-FM in the shared call format")


def train_teacher(
    teacher: nn.Module, train: EchoNextDataset, val: EchoNextDataset, output_dir: str | Path,
    *, device: str = "cpu", seed: int = 42,
) -> Path:
    """
    Fine-tune the Teacher and save the best checkpoint.

    Args:
        teacher: Output of build_teacher.
        train: The Training ECGs.
        val: The Validation ECGs, used for early stopping and tuning.
        output_dir: Folder for checkpoints. Never choose settings using Test ECGs.
        device: Where training runs.
        seed: Random seed.

    Returns:
        The path of the best checkpoint.
    """
    raise NotImplementedError("Role B: fine-tune with mixed precision, masking missing labels")


def load_frozen_teacher(checkpoint: str | Path, *, device: str = "cpu") -> nn.Module:
    """
    Load the finished Teacher, ready to run.

    Args:
        checkpoint: A file written by train_teacher.
        device: Where the model is placed.

    Returns:
        The model in evaluation mode, in the shared call format.
    """
    raise NotImplementedError("Role B: restore the checkpoint saved by train_teacher")
