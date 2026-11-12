"""Released EchoNext-Mini inference and the seven-feature gradient boosting baseline.

Only the upstream torch model is imported; Lightning and the upstream training
stack are not required. The released checkpoint stores its state in ['model'].
Owner: Role A.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from .data import LABEL_COLUMNS, TABULAR_COLUMNS, EchoNextDataset
from .predictions import fresh_run_dir, predict, run_manifest, save_run, sha256_file

INTROECG_COMMIT = "15233e9392e9dc10c136a5624abf10199207bce9"
MINI_WEIGHTS_SHA256 = "aac57aad0ec9763021c6264e89c74e3d78f8b711159b1880e5bd4515d5e21cd6"
MINI_SOURCE_SHA256 = "6539e16953f231fe964957464d49d11fec529fc95dbaf845a2721e69e9328d39"
DEFAULT_MINI_DIR = Path(__file__).resolve().parents[2] / "external/IntroECG/7-EchoNext Minimodel"
MINI_CHECKPOINT = Path("models/echonext_multilabel_minimodel/weights.pt")
MINI_WIDTH = 16
GRADIENT_BOOSTING_CHECKPOINT = "gradient_boosting.joblib"


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
    weights = Path(mini_dir) / MINI_CHECKPOINT
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
    outputs = predict(model, dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    manifest = run_manifest(
        "EchoNext-Mini", dataset, data_version=data_version,
        introecg_commit=INTROECG_COMMIT, weights_sha256=MINI_WEIGHTS_SHA256,
        source_sha256=MINI_SOURCE_SHA256, requested_limit=limit, seed=seed,
        batch_size=batch_size, num_workers=num_workers, device=device,
    )
    metrics = save_run(destination, outputs, dataset.selected_metadata, manifest)
    return {"output_dir": str(destination), "n_records": len(dataset), "metrics": metrics}


@dataclass(frozen=True)
class GradientBoostingBaseline:
    """A fitted binary estimator or constant positive-class probability for each label."""

    estimators: tuple[HistGradientBoostingClassifier | float, ...]
    training_counts: dict[str, dict[str, int]]


def fit_gradient_boosting(train: EchoNextDataset, *, seed: int = 42) -> GradientBoostingBaseline:
    """Fit each output from its observed training targets and the seven supplied features."""
    if train.split != "train":
        raise ValueError("Fit gradient boosting on the official train split")
    features = train.selected_tabular
    labels = train.labels[train.indices]
    label_mask = train.label_mask[train.indices]
    estimators = []
    training_counts = {}
    for column, name in enumerate(LABEL_COLUMNS):
        observed = label_mask[:, column]
        targets = labels[observed, column]
        count = len(targets)
        positives = int(targets.sum())
        training_counts[name] = {"n_observed": count, "n_positive": positives}
        if 0 < positives < count:
            estimator = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
            estimators.append(estimator.fit(features[observed], targets))
        else:
            estimators.append(positives / count if count else 0.5)
    return GradientBoostingBaseline(tuple(estimators), training_counts)


def predict_gradient_boosting(
    model: GradientBoostingBaseline, dataset: EchoNextDataset,
) -> dict[str, np.ndarray]:
    """Score the selected rows and return the shared prediction arrays in dataset order."""
    features = dataset.selected_tabular
    probabilities = np.empty((len(dataset), len(LABEL_COLUMNS)), dtype=np.float64)
    for column, estimator in enumerate(model.estimators):
        if isinstance(estimator, HistGradientBoostingClassifier):
            probabilities[:, column] = estimator.predict_proba(features)[:, 1]
        else:
            probabilities[:, column] = estimator
    # Keep log odds finite for constant models and very confident predictions.
    np.clip(probabilities, 1e-7, 1 - 1e-7, out=probabilities)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    mask = dataset.label_mask[dataset.indices]
    labels = dataset.labels[dataset.indices]
    labels[~mask] = np.nan
    selected = dataset.selected_metadata
    return {
        "logits": logits.astype(np.float32),
        "probabilities": probabilities.astype(np.float32),
        "labels": labels,
        "label_mask": mask,
        "row_id": selected["row_id"].to_numpy(),
        "split_row": selected["split_row"].to_numpy(),
        "patient_key": selected["patient_key"].to_numpy(dtype=str),
        "label_columns": np.asarray(LABEL_COLUMNS, dtype=str),
    }


def run_gradient_boosting(
    data_dir: str | Path | None, output_dir: str | Path, *, split: str = "val",
    limit: int | None = None, train_limit: int | None = None, seed: int = 42,
    data_version: str = "1.1.1",
) -> dict:
    """Train on the official train split and save predictions for the requested split."""
    dataset = EchoNextDataset(data_dir, split, limit=limit, seed=seed, load_waveforms=False)
    destination = fresh_run_dir(output_dir, dataset.root)
    train = EchoNextDataset(dataset.root, "train", limit=train_limit, seed=seed, load_waveforms=False)
    model = fit_gradient_boosting(train, seed=seed)
    outputs = predict_gradient_boosting(model, dataset)
    manifest = run_manifest(
        "gradient_boosting", dataset, data_version=data_version, requested_limit=limit,
        seed=seed, train_limit=train_limit, n_train_records=len(train), device="cpu",
        train_tabular_sha256=sha256_file(train.tabular_path),
        score_tabular_sha256=sha256_file(dataset.tabular_path),
        training_counts=model.training_counts, tabular_columns=list(TABULAR_COLUMNS),
        estimator="HistGradientBoostingClassifier", early_stopping=False, sklearn_version=sklearn.__version__,
        checkpoint=GRADIENT_BOOSTING_CHECKPOINT,
    )

    def save_checkpoint(staging: Path) -> None:
        joblib.dump(model, staging / GRADIENT_BOOSTING_CHECKPOINT, compress=3)

    metrics = save_run(destination, outputs, dataset.selected_metadata, manifest, write_artifacts=save_checkpoint)
    return {"output_dir": str(destination), "n_records": len(dataset), "metrics": metrics}
