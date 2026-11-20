"""
Fine-tune ECG-FM into the Teacher.

Purpose:
- Convert our ECGs into the input ECG-FM was pretrained on, inside the model.
- Add a 12-output layer to ECG-FM and fine-tune it with mixed precision.
- Optionally add the 7 routine numbers to the outputs through a second small layer.
- Train the output layer alone on saved ECG-FM features first, as a quick check.
- Produce a frozen Teacher that follows the shared call format.

Special Notes:
- Owner: Role B.
- Loading ECG-FM needs fairseq-signals, which only this role installs. The README shows how.
- predictions.write_run saves the Teacher's outputs once the model follows the call format.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import LABEL_COLUMNS, SHD_INDEX, TABULAR_COLUMNS, EchoNextDataset
from .predictions import basic_metrics, predict

DEFAULT_PRETRAINED = (
    Path(__file__).resolve().parents[2] / "checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
)
# ECG-FM was pretrained on 5-second windows at 500 readings per second. Our ECGs are
# 10 seconds at 250 per second, so each one becomes two windows.
SOURCE_RATE = 250
ECG_FM_RATE = 500
WINDOW_READINGS = 2500
EMBED_DIM = 768
# A lead whose spread is below this is treated as flat and set to zero, as ECG-FM's notebook does.
FLAT_LEAD_STD = 1e-6


def to_ecg_fm_input(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert our ECGs into the windows ECG-FM expects.

    Args:
        waveform: ECGs of shape [batch, 12, 2500], 10 seconds at 250 readings per second.

    Returns:
        Windows of shape [batch * 2, 12, 2500], 5 seconds each at 500 readings per second.
        Window 2i is the first half of ECG i, and window 2i + 1 is the second half.

    Notes:
        Upsampling matches fairseq-signals preprocess.py, which uses scipy interp1d with the
        first and last readings fixed in place. Each lead is then scaled to mean 0 and spread 1
        over the whole 10 seconds before splitting, following the order in ECG-FM's notebook.
    """
    batch, leads, readings = waveform.shape
    upsampled = F.interpolate(
        waveform, size=readings * ECG_FM_RATE // SOURCE_RATE, mode="linear", align_corners=True,
    )
    mean = upsampled.mean(dim=-1, keepdim=True)
    std = upsampled.std(dim=-1, keepdim=True, unbiased=False)
    flat = std < FLAT_LEAD_STD
    scaled = torch.where(flat, torch.zeros_like(upsampled), (upsampled - mean) / std.clamp_min(FLAT_LEAD_STD))
    windows = scaled.reshape(batch, leads, -1, WINDOW_READINGS).transpose(1, 2)
    return windows.reshape(-1, leads, WINDOW_READINGS)


def check_waveform(waveform: torch.Tensor) -> None:
    """Raise ValueError unless waveform has shape [batch, 12, 2500]."""
    if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (12, 2500):
        raise ValueError("Teacher expects waveform [batch,12,2500]")


def patient_weights(patient_keys: np.ndarray, alpha: float) -> np.ndarray:
    """
    Weight each ECG by 1 over its patient's ECG count, raised to alpha.

    Args:
        patient_keys: The patient of every ECG, in dataset order.
        alpha: 0 gives every ECG the same weight, and 1 gives every patient the same total weight.

    Returns:
        One float weight per ECG. Patients with many ECGs get less weight per ECG.
    """
    keys = np.asarray(patient_keys)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.float64) ** -alpha


class TeacherHead(nn.Module):
    """The new output layer, with an optional rescaling of ECG-FM's features in front of it."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        # Active only in training mode. A value of 0 leaves the features untouched.
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(EMBED_DIM, len(LABEL_COLUMNS))
        # Same starting values as ECG-FM's own classification layer.
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)
        # Mean 0 and spread 1 leave the features unchanged until fit_probe sets them.
        self.register_buffer("feature_mean", torch.zeros(EMBED_DIM))
        self.register_buffer("feature_std", torch.ones(EMBED_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout((features - self.feature_mean) / self.feature_std))


class Teacher(nn.Module):
    """ECG-FM plus a 12-output layer, in the shared call format."""

    def __init__(
        self, encoder: nn.Module, *, head_dropout: float = 0.0, lead_mask_prob: float = 0.0,
        use_tabular: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = TeacherHead(head_dropout)
        # In training mode only, each lead of each ECG is set to zero with this probability.
        self.lead_mask_prob = lead_mask_prob
        self.tabular_head = None
        if use_tabular:
            self.tabular_head = nn.Linear(len(TABULAR_COLUMNS), len(LABEL_COLUMNS))
            # Zero weights make a new Teacher start equal to the waveform-only one.
            nn.init.zeros_(self.tabular_head.weight)
            nn.init.zeros_(self.tabular_head.bias)

    def features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return ECG-FM's 768 averaged features for each ECG in waveform [batch, 12, 2500]."""
        check_waveform(waveform)
        if self.training and self.lead_mask_prob > 0:
            # ECG-FM was pretrained with random lead masking, so this input is familiar to it.
            keep = torch.rand(waveform.shape[0], waveform.shape[1], 1, device=waveform.device) >= self.lead_mask_prob
            keep = keep | ~keep.any(dim=1, keepdim=True)  # Never blank every lead of an ECG.
            waveform = waveform * keep.to(waveform.dtype)
        windows = to_ecg_fm_input(waveform)
        steps = self.encoder.extract_features(windows, padding_mask=None, mask=False)["x"]
        # Averaging every time step of both windows matches ECG-FM's classifier for one window.
        return steps.mean(dim=1).reshape(waveform.shape[0], -1, EMBED_DIM).mean(dim=1)

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        check_waveform(waveform)
        if tuple(tabular.shape) != (waveform.shape[0], 7):
            raise ValueError("Teacher expects tabular [batch,7]")
        logits = self.head(self.features(waveform))
        if self.tabular_head is None:
            # ECG-FM only reads the waveform, so the 7 numbers are ignored unless switched on.
            return logits
        return logits + self.tabular_head(tabular)


def build_teacher(
    pretrained_path: str | Path, *, device: str = "cpu", head_dropout: float = 0.0, lead_mask_prob: float = 0.0,
    use_tabular: bool = False,
) -> nn.Module:
    """
    Load ECG-FM and add a new output layer.

    Args:
        pretrained_path: The ECG-FM checkpoint from scripts/download_weights.sh.
        device: Where the model is placed.
        head_dropout: Dropout on the 768 features before the output layer, in training only.
        lead_mask_prob: Chance that each lead is blanked in an ECG, in training only.
        use_tabular: Whether a second output layer reads the 7 routine numbers.

    Returns:
        A model where model(waveform, tabular) returns logits [batch, 12]. It ignores tabular
        unless use_tabular is on.

    Raises:
        FileNotFoundError: When the checkpoint is missing.
        ImportError: When fairseq-signals is not installed.
    """
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"No ECG-FM checkpoint at {pretrained_path}. Run scripts/download_weights.sh")
    try:
        from fairseq_signals.models import build_model_from_checkpoint
    except ImportError as error:
        raise ImportError("Loading ECG-FM needs fairseq-signals. See Teacher setup in the README") from error
    encoder = build_model_from_checkpoint(checkpoint_path=str(pretrained_path))
    # The quantizer and projection layers only served pretraining. Removing them leaves 90,374,016 parameters.
    encoder.remove_pretraining_modules()
    teacher = Teacher(encoder, head_dropout=head_dropout, lead_mask_prob=lead_mask_prob, use_tabular=use_tabular)
    return teacher.to(device)


@torch.inference_mode()
def extract_features(
    teacher: Teacher, dataset: EchoNextDataset, *, batch_size: int = 32,
    num_workers: int = 0, device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Run ECG-FM once over a dataset and keep its averaged features.

    Args:
        teacher: Output of build_teacher. Only its encoder is used.
        dataset: The ECGs to run.
        batch_size: ECGs per step.
        num_workers: Data loader workers.
        device: Where ECG-FM runs.

    Returns:
        features float32 [N, 768], labels [N, 12] with missing labels set to 0,
        label_mask [N, 12], and row_id [N], in the dataset's order.
    """
    teacher = teacher.to(device).eval()
    parts = {key: [] for key in ("features", "labels", "label_mask", "row_id")}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        parts["features"].append(teacher.features(batch["waveform"].to(device)).float().cpu().numpy())
        for key in ("labels", "label_mask", "row_id"):
            parts[key].append(batch[key].numpy())
    return {key: np.concatenate(values) for key, values in parts.items()}


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average the binary cross-entropy over observed labels only."""
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return losses[mask].mean()


def shd_auroc(head: TeacherHead, features: dict[str, np.ndarray]) -> float:
    """Return the overall SHD AUROC of a head on saved features."""
    with torch.no_grad():
        scores = head(torch.from_numpy(features["features"]))[:, SHD_INDEX].numpy()
    return float(roc_auc_score(features["labels"][:, SHD_INDEX], scores))


def fit_probe(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray], *, rescale: bool, lr: float,
    max_epochs: int = 100, patience: int = 5, batch_size: int = 1024, seed: int = 42,
) -> tuple[TeacherHead, list[float]]:
    """
    Train only the output layer on saved ECG-FM features, keeping the best epoch.

    Args:
        train: Output of extract_features on the Training ECGs.
        val: Output of extract_features on the Validation ECGs, used to pick the epoch.
        rescale: Whether to scale each feature to mean 0 and spread 1, using Training values.
        lr: Adam learning rate.
        max_epochs: Upper limit on passes over the Training ECGs.
        patience: Epochs without a better Validation SHD AUROC before stopping.
        batch_size: ECGs per step.
        seed: Random seed for the starting values and batch order.

    Returns:
        The head from the epoch with the best Validation SHD AUROC, and that AUROC for each epoch.
    """
    torch.manual_seed(seed)
    head = TeacherHead()
    features = torch.from_numpy(train["features"])
    if rescale:
        head.feature_mean.copy_(features.mean(dim=0))
        head.feature_std.copy_(features.std(dim=0, unbiased=False).clamp_min(FLAT_LEAD_STD))
    labels = torch.from_numpy(train["labels"]).float()
    mask = torch.from_numpy(train["label_mask"]).bool()
    optimizer = torch.optim.Adam(head.proj.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history, best_state = [], None
    for _ in range(max_epochs):
        head.train()
        for rows in torch.randperm(len(features), generator=generator).split(batch_size):
            optimizer.zero_grad()
            masked_bce(head(features[rows]), labels[rows], mask[rows]).backward()
            optimizer.step()
        head.eval()
        history.append(shd_auroc(head, val))
        if history[-1] >= max(history):
            best_state = {key: value.clone() for key, value in head.state_dict().items()}
        elif len(history) - 1 - int(np.argmax(history)) >= patience:
            break
    head.load_state_dict(best_state)
    return head, history


def mean_auroc(metrics: dict) -> float:
    """Average the AUROC over every output that has one. This score picks the best epoch."""
    return float(np.mean([entry["auroc"] for entry in metrics.values() if entry["auroc"] is not None]))


def train_teacher(
    teacher: nn.Module, train: EchoNextDataset, val: EchoNextDataset, output_dir: str | Path,
    *, device: str = "cpu", seed: int = 42, probe_head: str | Path | None = None,
    lr: float = 1e-5, head_lr: float = 1e-4, batch_size: int = 32, max_epochs: int = 10,
    patience: int = 2, num_workers: int = 0, log_every: int = 200, balance_patients: float = 0.0,
    tabular_lr: float = 1e-3,
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
        probe_head: Output layer saved by scripts/run_teacher_probe.py, used as the starting point.
        lr: AdamW learning rate for ECG-FM.
        head_lr: AdamW learning rate for the output layer.
        batch_size: ECGs per step.
        max_epochs: Upper limit on passes over the Training ECGs.
        patience: Epochs without a better Validation mean AUROC before stopping.
        num_workers: Data loader workers.
        log_every: Steps between progress lines.
        balance_patients: Exponent for patient_weights. Above 0, each epoch draws ECGs with those
            weights, so patients with many ECGs are seen less often per ECG. 0 shuffles as usual.
        tabular_lr: AdamW learning rate for the layer that reads the 7 routine numbers, when the
            Teacher has one. It starts at zero, so it gets a higher rate than the output layer.

    Returns:
        The path of the best checkpoint, chosen by the mean AUROC over all 12 outputs.

    Raises:
        FileExistsError: When output_dir already holds a log from an earlier run.

    Notes:
        Writes best.pt and log.csv, with one row per epoch, to output_dir. On a GPU that
        supports it, the forward pass runs in bfloat16 mixed precision.
    """
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "log.csv").exists():
        raise FileExistsError(f"{output_dir / 'log.csv'} already exists. Choose a new output folder for each run")
    if probe_head is not None:
        teacher.head.load_state_dict(torch.load(probe_head, map_location="cpu", weights_only=True))
    # ECG-FM's own fine-tuning keeps its first convolution layers fixed.
    if hasattr(teacher.encoder, "feature_grad_mult"):
        teacher.encoder.feature_grad_mult = 0.0
    teacher = teacher.to(device)
    groups = [
        {"params": list(teacher.encoder.parameters()), "lr": lr},
        {"params": list(teacher.head.parameters()), "lr": head_lr},
    ]
    if getattr(teacher, "tabular_head", None) is not None:
        groups.append({"params": list(teacher.tabular_head.parameters()), "lr": tabular_lr})
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=0.01)
    on_gpu = str(device).startswith("cuda")
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
    sampler = None
    if balance_patients > 0:
        weights = patient_weights(train.selected_metadata["patient_key"].to_numpy(), balance_patients)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loader = DataLoader(
        train, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers,
        pin_memory=on_gpu, generator=torch.Generator().manual_seed(seed),
    )
    best_path, log_path = output_dir / "best.pt", output_dir / "log.csv"
    history = []
    for epoch in range(1, max_epochs + 1):
        start, total = time.time(), 0.0
        teacher.train()
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if on_gpu else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                logits = teacher(batch["waveform"].to(device), batch["tabular"].to(device))
            loss = masked_bce(logits.float(), batch["labels"].to(device), batch["label_mask"].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {total / step:.4f} "
                      f"{time.time() - start:.0f}s", flush=True)
        metrics = basic_metrics(predict(teacher, val, batch_size=batch_size, num_workers=num_workers, device=device))
        row = {
            "epoch": epoch, "train_loss": round(total / len(loader), 5),
            "val_mean_auroc": round(mean_auroc(metrics), 5),
            "val_shd_auroc": round(metrics[LABEL_COLUMNS[SHD_INDEX]]["auroc"], 5),
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"epoch {epoch} done: {row}", flush=True)
        best = max(range(len(history)), key=lambda i: (history[i]["val_mean_auroc"], i))
        if best == len(history) - 1:
            torch.save({"epoch": epoch, "state_dict": teacher.state_dict(), **row}, best_path)
        elif len(history) - 1 - best >= patience:
            break
    return best_path


def restore_teacher(teacher: Teacher, checkpoint: str | Path) -> Teacher:
    """Load a checkpoint written by train_teacher into a Teacher with the same layout."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    teacher.load_state_dict(saved["state_dict"], strict=True)
    return teacher


def load_frozen_teacher(
    checkpoint: str | Path, *, pretrained_path: str | Path = DEFAULT_PRETRAINED, device: str = "cpu",
) -> nn.Module:
    """
    Load the finished Teacher, ready to run.

    Args:
        checkpoint: A file written by train_teacher.
        pretrained_path: The ECG-FM checkpoint, which supplies the model layout.
        device: Where the model is placed.

    Returns:
        The model in evaluation mode, in the shared call format. It reads the 7 routine
        numbers exactly when the saved Teacher did.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    teacher = build_teacher(pretrained_path, use_tabular="tabular_head.weight" in saved["state_dict"])
    teacher.load_state_dict(saved["state_dict"], strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher.to(device).eval()
