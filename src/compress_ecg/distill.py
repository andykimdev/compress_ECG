"""
Train the Student to copy the Teacher.

Purpose:
- Define the distillation loss.
- Train a Student from saved Teacher logits, without loading the Teacher itself.

Special Notes:
- Owner: Role C. Every function here is still to be written.
- During Phase 1, the Teacher run can be EchoNext-Mini's saved run on the Training ECGs.
- Inside the training loop, predictions.aligned_logits(run, batch["row_id"]) finds each
  batch's Teacher logits, including when batches are shuffled.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import EchoNextDataset
from .predictions import PredictionRun


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
    label_mask: torch.Tensor, *, temperature: float, hard_label_weight: float,
) -> torch.Tensor:
    """
    Combine copying the Teacher with matching the true labels.

    Args:
        student_logits: Student outputs [batch, 12].
        teacher_logits: Saved Teacher outputs [batch, 12].
        labels: True labels [batch, 12].
        label_mask: False where a label must be left out of the loss [batch, 12].
        temperature: Softening applied to both sets of logits.
        hard_label_weight: Share of the loss from the true labels, 0 to 1.

    Returns:
        A single loss value.
    """
    raise NotImplementedError("Role C: combine soft Teacher and hard label losses")


def train_student(
    student: nn.Module, teacher_run: PredictionRun, train: EchoNextDataset, val: EchoNextDataset,
    output_dir: str | Path, *, device: str = "cpu", seed: int = 42,
) -> Path:
    """
    Train the Student and save the best checkpoint.

    Args:
        student: Output of student.build_student.
        teacher_run: Saved Teacher outputs on the Training ECGs, from predictions.load_run.
        train: The Training ECGs.
        val: The Validation ECGs, used for early stopping and tuning.
        output_dir: Folder for checkpoints. Save them with student.save_student.
        device: Where training runs.
        seed: Random seed.

    Returns:
        The path of the best checkpoint.
    """
    raise NotImplementedError("Role C: train with distillation_loss and save with save_student")
