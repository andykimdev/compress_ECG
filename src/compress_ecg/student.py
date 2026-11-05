"""
Build, save, and load the Student model.

Purpose:
- Define the Student as the EchoNext-Mini layout at a smaller channel width.
- Fix the checkpoint format that Role C hands to Roles A and D.

Special Notes:
- Owner: Role C. Training lives in distill.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .baseline import DEFAULT_MINI_DIR, EchoNextMini, build_mini_design

STUDENT_WIDTH = 5


def build_student(
    width: int = STUDENT_WIDTH, *, mini_dir: str | Path = DEFAULT_MINI_DIR, device: str = "cpu",
) -> EchoNextMini:
    """
    Build an untrained Student.

    Args:
        width: Base channel count. The plan uses 5, which gives 104,436 parameters.
        mini_dir: Folder holding the pinned EchoNext-Mini code.
        device: Where the model is placed.

    Returns:
        The model, in the shared call format.
    """
    return build_mini_design(width, mini_dir).to(device)


def save_student(student: EchoNextMini, path: str | Path, *, width: int = STUDENT_WIDTH) -> Path:
    """
    Save a trained Student for the other roles.

    Args:
        student: The trained model.
        path: Checkpoint file to write.
        width: The channel width the model was built with.

    Returns:
        The checkpoint path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"width": width, "state_dict": student.state_dict()}, path)
    return path


def load_student(
    path: str | Path, *, mini_dir: str | Path = DEFAULT_MINI_DIR, device: str = "cpu",
) -> EchoNextMini:
    """
    Load a saved Student, ready to run.

    Args:
        path: A file written by save_student.
        mini_dir: Folder holding the pinned EchoNext-Mini code.
        device: Where the model is placed.

    Returns:
        The model, in evaluation mode.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    student = build_student(checkpoint["width"], mini_dir=mini_dir)
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    return student.to(device).eval()
