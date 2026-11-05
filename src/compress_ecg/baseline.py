"""Released EchoNext-Mini inference using its pinned architecture and weights.

Only the upstream torch model is imported; Lightning and the upstream training
stack are not required. The released checkpoint stores its state in ['model'].
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import AUDIT_COLUMNS, LABEL_COLUMNS, EchoNextDataset

INTROECG_COMMIT = "15233e9392e9dc10c136a5624abf10199207bce9"
MINI_WEIGHTS_SHA256 = "aac57aad0ec9763021c6264e89c74e3d78f8b711159b1880e5bd4515d5e21cd6"
MINI_SOURCE_SHA256 = "6539e16953f231fe964957464d49d11fec529fc95dbaf845a2721e69e9328d39"
DEFAULT_MINI_DIR = Path(__file__).resolve().parents[2] / "external/IntroECG/7-EchoNext Minimodel"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run bash scripts/download_weights.sh --mini-only")
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path}; restore the pinned Mini release")


class EchoNextMini(nn.Module):
    """Adapter accepting channel-first ECGs. forward returns LOGITS, not risks."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or tuple(waveform.shape[1:]) != (12, 2500):
            raise ValueError("Mini expects waveform [batch,12,2500]")
        if tuple(tabular.shape) != (waveform.shape[0], 7):
            raise ValueError("Mini expects tabular [batch,7]")
        return self.model((waveform.transpose(1, 2).unsqueeze(1), tabular))


def load_mini(mini_dir: str | Path = DEFAULT_MINI_DIR, *, device: str = "cpu") -> EchoNextMini:
    """Verify both files, strictly load the released 12-output model and set eval."""
    root = Path(mini_dir)
    source = root / "cradlenet/models/resnet1d_tabular.py"
    weights = root / "models/echonext_multilabel_minimodel/weights.pt"
    _verify_file(source, MINI_SOURCE_SHA256)
    _verify_file(weights, MINI_WEIGHTS_SHA256)
    spec = importlib.util.spec_from_file_location("_echonext_pinned_resnet1d_tabular", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backbone = module.ResNet1dWithTabular(
        len_tabular_feature_vector=7, filter_size=16, num_classes=12,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    return EchoNextMini(backbone).to(device).eval()


@torch.inference_mode()
def predict_mini(
    model: nn.Module, dataset: EchoNextDataset, *, batch_size: int = 64,
    num_workers: int = 0, device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Return ordered, portable arrays for E; missing labels remain NaN on disk."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    model = model.to(device).eval()
    parts = {key: [] for key in ("probabilities", "labels", "label_mask", "row_id", "split_row")}
    patient_keys = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        logits = model(batch["waveform"].to(device), batch["tabular"].to(device))
        if tuple(logits.shape) != (len(batch["row_id"]), len(LABEL_COLUMNS)):
            raise ValueError("Mini must produce [batch,12] logits")
        if not torch.isfinite(logits).all():
            raise ValueError("Mini produced non-finite logits")
        parts["probabilities"].append(torch.sigmoid(logits).cpu().numpy())
        labels = batch["labels"].numpy().copy()
        mask = batch["label_mask"].numpy()
        labels[~mask] = np.nan
        parts["labels"].append(labels)
        parts["label_mask"].append(mask)
        for key in ("row_id", "split_row"):
            parts[key].append(batch[key].numpy())
        patient_keys.extend(batch["patient_key"])
    outputs = {key: np.concatenate(values) for key, values in parts.items()}
    outputs["patient_key"] = np.asarray(patient_keys, dtype=str)
    outputs["label_columns"] = np.asarray(LABEL_COLUMNS, dtype=str)
    return outputs


def baseline_metrics(outputs: dict[str, np.ndarray]) -> dict:
    """Small sanity scorecard. E owns calibration plots and the full fairness audit."""
    metrics = {}
    for column, name in enumerate(LABEL_COLUMNS):
        mask = outputs["label_mask"][:, column]
        y = outputs["labels"][mask, column]
        p = outputs["probabilities"][mask, column]
        metrics[name] = {
            "n_observed": int(len(y)), "n_positive": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)) if len(y) else None,
        }
    return metrics


def _code_version() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_baseline(
    data_dir: str | Path | None, output_dir: str | Path, *, split: str = "val",
    limit: int | None = 1000, seed: int = 42, batch_size: int = 64,
    num_workers: int = 0, device: str = "cpu", mini_dir: str | Path = DEFAULT_MINI_DIR,
    data_version: str = "1.1.1",
) -> dict:
    """Run the baseline and save private predictions, audit fields and provenance.

    Use a fresh output directory for each run to prevent mixing old/new files.
    """
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    dataset = EchoNextDataset(data_dir, split, limit=limit, seed=seed)
    if destination == dataset.root or dataset.root in destination.parents:
        raise ValueError("Store outputs outside the source dataset directory")
    model = load_mini(mini_dir, device=device)
    outputs = predict_mini(model, dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    selected = dataset.selected_metadata
    np.testing.assert_array_equal(outputs["row_id"], selected["row_id"].to_numpy())
    metrics = baseline_metrics(outputs)
    audit_columns = ["row_id", "split_row", "patient_key", "split"]
    audit_columns += [column for column in AUDIT_COLUMNS if column in selected]
    manifest = {
        "model": "EchoNext-Mini", "introecg_commit": INTROECG_COMMIT,
        "weights_sha256": MINI_WEIGHTS_SHA256, "source_sha256": MINI_SOURCE_SHA256,
        "code": _code_version(), "data_version": data_version,
        "data_dir": str(dataset.root), "metadata_sha256": sha256_file(dataset.root / "EchoNext_metadata_100k.csv"),
        "split": split, "requested_limit": limit, "n_records": len(dataset), "seed": seed,
        "batch_size": batch_size, "num_workers": num_workers, "device": device,
        "label_columns": list(LABEL_COLUMNS), "sampling_rate_hz": 250,
        "preprocessing": "official preprocessed arrays; no additional normalization",
        "waveform_file": {"name": dataset.waveform_path.name, "shape": list(dataset._waveforms.shape),
                          "bytes": dataset.waveform_path.stat().st_size},
        "tabular_file": {"name": dataset.tabular_path.name, "shape": list(dataset._tabular.shape),
                         "bytes": dataset.tabular_path.stat().st_size},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "predictions.npz", **outputs)
    selected[audit_columns].to_csv(destination / "metadata.csv", index=False)
    for filename, value in (("manifest.json", manifest), ("metrics.json", metrics)):
        (destination / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {"output_dir": str(destination), "n_records": len(dataset), "metrics": metrics}
