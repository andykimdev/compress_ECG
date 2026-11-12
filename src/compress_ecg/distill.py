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

import random
from pathlib import Path
import json

from .student import save_student

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from sklearn.metrics import roc_auc_score

from .data import EchoNextDataset, SHD_INDEX
from .predictions import PredictionRun, aligned_logits


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
    student: nn.Module, 
    teacher_run: PredictionRun, 
    train: EchoNextDataset, 
    val: EchoNextDataset,
    output_dir: str | Path, 
    *, 
    device: str = "cpu", 
    seed: int = 42,
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

    # Init Training Settings
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_epochs = 50
    patience = 7
    min_delta = 0.001
    temperature = 2.0
    hard_label_weight = 0.5

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    student = student.to(device)

    if train.split != "train" or val.split != "val":
        raise ValueError("Expected training and validation datasets")

    if teacher_run.manifest["split"] != "train":
        raise ValueError("Teacher predictions must come from the training split")

    train_row_ids = train.selected_metadata["row_id"].to_numpy()
    teacher_row_ids = teacher_run.arrays["row_id"]

    if not np.isin(train_row_ids, teacher_row_ids).all():
        raise ValueError("Teacher predictions are missing training ECGs")


    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    optimizer = AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = output_dir / "best_student.pt"
    if best_checkpoint_path.exists():
        raise FileExistsError(best_checkpoint_path)

    early_stopping_auc = float("-inf")
    best_val_auc = float("-inf")
    epochs_without_improvement = 0
    history = []

    # Training Loop
    for epoch in range(max_epochs):
        student.train()
        
        total_train_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            waveform = batch["waveform"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            teacher_logits = aligned_logits(
                teacher_run,
                batch["row_id"],
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            student_logits = student(waveform, tabular)

            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                label_mask,
                temperature=temperature,
                hard_label_weight=hard_label_weight,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch + 1}"
                )

            loss.backward()
            optimizer.step()

            current_batch_size = waveform.shape[0]
            total_train_loss += loss.item() * current_batch_size
            total_examples += current_batch_size

        train_loss = total_train_loss / total_examples
        
        student.eval()

        total_val_loss = 0.0
        total_observed_labels = 0
        shd_targets = []
        shd_probabilities = []

        # Validation computes predictions without tracking gradients.
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch["waveform"].to(device)
                tabular = batch["tabular"].to(device)
                labels = batch["labels"].to(device)
                label_mask = batch["label_mask"].to(device)

                logits = student(waveform, tabular)

                if not torch.isfinite(logits).all().item():
                    raise RuntimeError(
                        f"Non-finite validation logits at epoch {epoch + 1}"
                    )

                # Calculate hard-label loss for observed targets only.
                losses = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                )

                total_val_loss += losses[label_mask].sum().item()
                total_observed_labels += label_mask.sum().item()

                # SHD labels are guaranteed to be observed by the loader.
                shd_targets.append(labels[:, SHD_INDEX].cpu())
                shd_probabilities.append(
                    torch.sigmoid(logits[:, SHD_INDEX]).cpu()
                )

        # Average loss across all observed validation targets.
        val_loss = total_val_loss / total_observed_labels

        # Compute AUROC once across the entire validation set.
        y_true = torch.cat(shd_targets).numpy()
        y_prob = torch.cat(shd_probabilities).numpy()

        if np.unique(y_true).size != 2:
            raise ValueError(
                "Validation SHD AUROC requires both positive and negative labels"
            )

        val_auc = float(roc_auc_score(y_true, y_prob))

                # Save whenever we achieve a new best validation AUROC.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_student(student, best_checkpoint_path)

        # Reset patience only for a meaningful improvement.
        if val_auc > early_stopping_auc + min_delta:
            early_stopping_auc = val_auc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Record this epoch's results.
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_shd_auc": val_auc,
            "best_val_shd_auc": best_val_auc,
        })

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )

        print(
            f"Best validation AUROC: {best_val_auc:.4f} | "
            f"Patience: {epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch + 1}")
            break


    return best_checkpoint_path
