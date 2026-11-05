import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

import compress_ecg.baseline as baseline
from compress_ecg.data import EchoNextDataset, LABEL_COLUMNS


class KnownLogits(nn.Module):
    def forward(self, waveform, tabular):
        return torch.zeros((len(waveform), 12), device=waveform.device)


def test_predictions_apply_sigmoid_once_and_preserve_order(data_dir):
    dataset = EchoNextDataset(data_dir, "val", limit=5, seed=4)
    result = baseline.predict_mini(KnownLogits(), dataset, batch_size=2)
    np.testing.assert_array_equal(result["probabilities"], np.full((5, 12), .5))
    np.testing.assert_array_equal(result["row_id"], dataset.selected_metadata.row_id)
    np.testing.assert_array_equal(result["split_row"], dataset.indices)
    np.testing.assert_array_equal(result["label_columns"], LABEL_COLUMNS)
    assert np.isnan(result["labels"][~result["label_mask"]]).all()


def test_metrics_ignore_unobserved_and_handle_single_class():
    result = {
        "labels": np.tile([0., 1., np.nan], (12, 1)).T,
        "probabilities": np.tile([.1, .9, .8], (12, 1)).T,
        "label_mask": np.tile([True, True, False], (12, 1)).T,
    }
    scores = baseline.baseline_metrics(result)
    assert scores[LABEL_COLUMNS[0]]["n_observed"] == 2
    assert scores[LABEL_COLUMNS[0]]["auroc"] == 1.0
    assert scores[LABEL_COLUMNS[0]]["brier"] == pytest.approx(.01)
    result["labels"][1] = 0
    assert baseline.baseline_metrics(result)[LABEL_COLUMNS[0]]["auroc"] is None
    result["label_mask"][:] = False
    assert baseline.baseline_metrics(result)[LABEL_COLUMNS[0]]["brier"] is None


def test_run_saves_private_evaluation_contract(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_mini", lambda *a, **kw: KnownLogits())
    output = tmp_path / "run"
    result = baseline.run_baseline(data_dir, output, limit=4, batch_size=3)
    assert result["n_records"] == 4
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        assert saved["probabilities"].shape == (4, 12)
        assert saved["patient_key"].dtype.kind == "U"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["weights_sha256"] == baseline.MINI_WEIGHTS_SHA256
    assert manifest["split"] == "val"
    assert (output / "metadata.csv").is_file()
    assert (output / "metrics.json").is_file()
    with pytest.raises(FileExistsError):
        baseline.run_baseline(data_dir, output)
    with pytest.raises(ValueError, match="outside the source"):
        baseline.run_baseline(data_dir, data_dir / "output")


def test_checkpoint_hash_required(tmp_path):
    source = tmp_path / "cradlenet/models/resnet1d_tabular.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('must not execute unverified source')")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        baseline.load_mini(tmp_path)


@pytest.fixture
def released_model():
    weights = baseline.DEFAULT_MINI_DIR / "models/echonext_multilabel_minimodel/weights.pt"
    if not weights.is_file():
        pytest.skip("Run bash scripts/download_weights.sh --mini-only for released-model checks")
    torch.set_num_threads(1)
    return baseline.load_mini()


def test_released_checkpoint_matches_upstream_and_is_batch_independent(data_dir, released_model):
    dataset = EchoNextDataset(data_dir, "val", limit=3)
    a = baseline.predict_mini(released_model, dataset, batch_size=1)
    b = baseline.predict_mini(released_model, dataset, batch_size=2)
    np.testing.assert_allclose(a["probabilities"], b["probabilities"], rtol=1e-5, atol=1e-6)
    spec = importlib.util.spec_from_file_location(
        "_upstream_reference", baseline.DEFAULT_MINI_DIR / "cradlenet/models/resnet1d_tabular.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = module.ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    upstream.load_state_dict(released_model.model.state_dict())
    upstream.eval()
    raw_waveforms = np.load(dataset.waveform_path)[dataset.indices]
    raw_tabular = np.load(dataset.tabular_path)[dataset.indices]
    with torch.inference_mode():
        expected = torch.sigmoid(upstream((torch.from_numpy(raw_waveforms), torch.from_numpy(raw_tabular))))
    np.testing.assert_allclose(a["probabilities"], expected.numpy(), rtol=1e-5, atol=1e-6)


def test_cli_with_released_model(data_dir, tmp_path, released_model):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_dir": str(data_dir), "limit": 4, "batch_size": 2}))
    output = tmp_path / "cli"
    completed = subprocess.run([
        sys.executable, "scripts/run_baseline.py", "--config", str(config),
        "--output-dir", str(output), "--limit", "3",
    ], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["n_records"] == 3
    assert json.loads((output / "manifest.json").read_text())["requested_limit"] == 3
