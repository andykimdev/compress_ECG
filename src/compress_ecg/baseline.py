"""Released EchoNext-Mini inference using its pinned architecture and weights.

Only the upstream torch model is imported; Lightning and the upstream training
stack are not required. The released checkpoint stores its state in ['model'].
Owner: Role A. The gradient boosting baseline at the bottom is still to be written.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
import time

from .data import EchoNextDataset
from .predictions import fresh_run_dir, predict, run_manifest, save_run, sha256_file, basic_metrics
from .audit import measure_model_size

INTROECG_COMMIT = "15233e9392e9dc10c136a5624abf10199207bce9"
MINI_WEIGHTS_SHA256 = "aac57aad0ec9763021c6264e89c74e3d78f8b711159b1880e5bd4515d5e21cd6"
MINI_SOURCE_SHA256 = "6539e16953f231fe964957464d49d11fec529fc95dbaf845a2721e69e9328d39"
DEFAULT_MINI_DIR = Path(__file__).resolve().parents[2] / "external/IntroECG/7-EchoNext Minimodel"
MINI_WIDTH = 16


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


def build_mini_design(width: int, mini_dir: str | Path = DEFAULT_MINI_DIR) -> EchoNextMini:
    """
    Build an untrained model with the released layout at a chosen channel width.

    Args:
        width: Base channel count, 16 for EchoNext-Mini and 5 for the Student.
        mini_dir: Folder holding the pinned EchoNext-Mini code.

    Returns:
        The model, in the shared call format.

    Raises:
        FileNotFoundError: When the pinned code has not been downloaded.
        ValueError: When the code does not match its pinned hash.
    """
    source = Path(mini_dir) / "cradlenet/models/resnet1d_tabular.py"
    _verify_file(source, MINI_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location("_echonext_pinned_resnet1d_tabular", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return EchoNextMini(module.ResNet1dWithTabular(
        len_tabular_feature_vector=7, filter_size=width, num_classes=12,
    ))


def load_mini(mini_dir: str | Path = DEFAULT_MINI_DIR, *, device: str = "cpu") -> EchoNextMini:
    """Verify both files, strictly load the released 12-output model and set eval."""
    mini = build_mini_design(MINI_WIDTH, mini_dir)
    weights = Path(mini_dir) / "models/echonext_multilabel_minimodel/weights.pt"
    _verify_file(weights, MINI_WEIGHTS_SHA256)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    mini.model.load_state_dict(checkpoint["model"], strict=True)
    return mini.to(device).eval()


def run_baseline(
    data_dir: str | Path | None, output_dir: str | Path, *, split: str = "val",
    limit: int | None = 1000, seed: int = 42, batch_size: int = 64,
    num_workers: int = 0, device: str = "cpu", mini_dir: str | Path = DEFAULT_MINI_DIR,
    data_version: str = "1.1.1",
) -> dict:
    """Run the baseline and save private predictions, audit fields and provenance.

    Use a fresh output directory for each run to prevent mixing old/new files.
    """
    dataset = EchoNextDataset(data_dir, split, limit=limit, seed=seed)
    destination = fresh_run_dir(output_dir, dataset.root)
    model = load_mini(mini_dir, device=device)
    start = time.perf_counter()
    outputs = predict(model, dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    predict_time = time.perf_counter() - start
    model_size_bytes = measure_model_size(model)
    manifest = run_manifest(
        "EchoNext-Mini", dataset, data_version=data_version,
        introecg_commit=INTROECG_COMMIT, weights_sha256=MINI_WEIGHTS_SHA256,
        source_sha256=MINI_SOURCE_SHA256, requested_limit=limit, seed=seed,
        batch_size=batch_size, num_workers=num_workers, device=device,
    )

    metrics = save_run(destination, outputs, dataset.selected_metadata, manifest)
    return {"output_dir": str(destination), "n_records": len(dataset), "metrics": metrics, "predict_time": f"{predict_time:.2f}s", "model_size_bytes": f"{model_size_bytes} bytes"}


def fit_gradient_boosting(train: EchoNextDataset, *, seed: int = 42) -> object:
    """
    Fit the non-neural baseline, using only the 7 tabular numbers.

    Args:
        train: The Training ECGs.
        seed: Random seed.

    Returns:
        A fitted model that predict_gradient_boosting accepts.
    """
    raise NotImplementedError("Role A: fit one gradient boosting model per output on tabular features")


def predict_gradient_boosting(model: object, dataset: EchoNextDataset) -> dict[str, np.ndarray]:
    """
    Score ECGs with the fitted baseline.

    Args:
        model: Output of fit_gradient_boosting.
        dataset: The ECGs to score.

    Returns:
        The arrays in predictions.ARRAY_NAMES, ready for predictions.save_run. Fill logits
        with the log odds of the clipped probabilities.
    """
    raise NotImplementedError("Role A: return the same arrays as predictions.predict")
