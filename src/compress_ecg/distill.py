"""
Train the Student to copy the Teacher.

Purpose:
- Define the distillation loss.
- Train a Student from saved Teacher logits, without loading the Teacher itself.

Special Notes:
- Owner: Role C.
- During Phase 1, the Teacher run can be EchoNext-Mini's saved run on the Training ECGs.
- Inside the training loop, predictions.aligned_logits(run, batch["row_id"]) finds each
  batch's Teacher logits, including when batches are shuffled.
"""

from __future__ import annotations

import random
import math
from pathlib import Path
import json

from .student import save_student

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from sklearn.metrics import roc_auc_score

from .baseline import EchoNextMini
from .data import EchoNextDataset, SHD_INDEX, find_metadata_file
from .predictions import PredictionRun, aligned_logits, sha256_file


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
    label_mask: torch.Tensor, *, temperature: float, hard_label_weight: float,
) -> torch.Tensor:
    """
    Combine copying the Teacher with matching the true labels.

    Each output is an independent binary task. The hard loss averages BCE over
    observed labels; the soft loss averages BCE against detached Teacher sigmoid
    probabilities over all outputs, including those without a true label. Both
    logits are divided by temperature for the soft loss, which is scaled by T^2.
    With no observed labels, the hard loss is zero.

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
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(hard_label_weight) or not 0 <= hard_label_weight <= 1:
        raise ValueError("hard_label_weight must be between 0 and 1")
    if student_logits.ndim != 2 or student_logits.shape[0] == 0 or student_logits.shape[1] != 12:
        raise ValueError("student_logits must have shape [batch, 12] with a nonempty batch")
    if any(tensor.shape != student_logits.shape for tensor in (teacher_logits, labels, label_mask)):
        raise ValueError("teacher_logits, labels, and label_mask must match student_logits shape")
    if label_mask.dtype != torch.bool:
        raise ValueError("label_mask must be boolean")

    # Select observed entries before BCE so masked NaNs cannot enter the loss.
    hard_loss = F.binary_cross_entropy_with_logits(
        student_logits[label_mask], labels[label_mask].to(student_logits.dtype), reduction="sum",
    ) / label_mask.sum().clamp_min(1)

    teacher_probabilities = torch.sigmoid(teacher_logits.detach() / temperature)
    soft_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature, teacher_probabilities, reduction="mean",
    ) * temperature ** 2

    return hard_label_weight * hard_loss + (1 - hard_label_weight) * soft_loss



def train_student(
    student: EchoNextMini, 
    teacher_run: PredictionRun, 
    train: EchoNextDataset, 
    val: EchoNextDataset,
    output_dir: str | Path, 
    *, 
    device: str = "cpu", 
    seed: int = 42,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 0.001,
    temperature: float = 2.0,
    hard_label_weight: float = 0.5,
    num_workers: int = 0,
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
        batch_size: ECGs per training or validation batch.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_epochs: Maximum number of training epochs.
        patience: Epochs without meaningful validation AUROC improvement before stopping.
        min_delta: AUROC improvement required to reset patience.
        temperature: Softening used in the distillation loss.
        hard_label_weight: Share of the loss from observed true labels, from 0 to 1.
        num_workers: DataLoader worker processes; zero loads in the main process.

    Returns:
        The path of the best checkpoint.
    """

    for name, value, minimum in (
        ("batch_size", batch_size, 1), ("max_epochs", max_epochs, 1),
        ("patience", patience, 1), ("num_workers", num_workers, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value, strictly_positive in (
        ("learning_rate", learning_rate, True), ("weight_decay", weight_decay, False),
        ("min_delta", min_delta, False), ("temperature", temperature, True),
    ):
        if not math.isfinite(value) or value < 0 or (strictly_positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
    if not math.isfinite(hard_label_weight) or not 0 <= hard_label_weight <= 1:
        raise ValueError("hard_label_weight must be between 0 and 1")

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

    # Row IDs are CSV positions, so they are meaningful only for the same metadata.
    train_metadata_path = find_metadata_file(train.root)
    val_metadata_path = find_metadata_file(val.root)
    if train_metadata_path is None:
        raise FileNotFoundError(f"No EchoNext metadata CSV in {train.root}")
    if val_metadata_path is None:
        raise FileNotFoundError(f"No EchoNext metadata CSV in {val.root}")
    train_metadata_hash = sha256_file(train_metadata_path)
    val_metadata_hash = sha256_file(val_metadata_path)
    if train_metadata_hash != val_metadata_hash:
        raise ValueError("Training and validation datasets must use the same metadata")

    teacher_metadata_hash = teacher_run.manifest.get("metadata_sha256")
    if not teacher_metadata_hash:
        raise ValueError("Teacher run is missing metadata_sha256; regenerate its predictions")
    if teacher_metadata_hash != train_metadata_hash:
        raise ValueError("Teacher run metadata_sha256 does not match the training dataset")

    train_row_ids = train.selected_metadata["row_id"].to_numpy()
    teacher_row_ids = teacher_run.arrays["row_id"]

    if not np.isin(train_row_ids, teacher_row_ids).all():
        raise ValueError("Teacher predictions are missing training ECGs")


    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
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

    settings = {
        "seed": seed, "device": device, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "max_epochs": max_epochs, "patience": patience, "min_delta": min_delta,
        "temperature": temperature, "hard_label_weight": hard_label_weight,
        "num_workers": num_workers, "train_records": len(train),
        "val_records": len(val), "train_seed": train.seed, "val_seed": val.seed,
        "teacher_manifest": teacher_run.manifest,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n"
    )

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
            f"Epoch {epoch + 1}/{max_epochs} | Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f} | "
            f"Best validation AUROC: {best_val_auc:.4f} | "
            f"Patience: {epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch + 1}")
            break


    return best_checkpoint_path
