import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from compress_ecg.data import SHD_INDEX, EchoNextDataset
from compress_ecg.profiling import synthetic_inputs
from compress_ecg.teacher import (
    Teacher, TeacherHead, build_teacher, extract_features, fit_probe, load_frozen_teacher, masked_bce,
    mean_auroc, patient_weights, restore_teacher, to_ecg_fm_input, train_teacher,
)

ECG_FM_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
)


class FakeEncoder(torch.nn.Module):
    """Stands in for ECG-FM: returns 3 time steps of 768 numbers per window."""

    def __init__(self):
        super().__init__()
        self.windows = None

    def extract_features(self, source, padding_mask, mask):
        self.windows = source
        steps = source.mean(dim=(1, 2))[:, None, None].expand(-1, 3, 768)
        return {"x": steps}


def reference_windows(ecg: np.ndarray) -> np.ndarray:
    """Rebuild one ECG's windows with scipy, the way fairseq-signals preprocess.py does."""
    readings = ecg.shape[-1] * 2
    upsampled = interp1d(np.linspace(0, readings - 1, ecg.shape[-1]), ecg, kind="linear")(np.arange(readings))
    scaled = (upsampled - upsampled.mean(-1, keepdims=True)) / upsampled.std(-1, keepdims=True)
    return np.stack([scaled[:, :2500], scaled[:, 2500:]])


def test_windows_match_the_ecg_fm_preprocessing():
    ecgs = np.random.default_rng(0).standard_normal((2, 12, 2500))
    windows = to_ecg_fm_input(torch.tensor(ecgs)).numpy()
    expected = np.concatenate([reference_windows(ecg) for ecg in ecgs])
    np.testing.assert_allclose(windows, expected, atol=1e-10)


def test_flat_lead_becomes_zero():
    ecg = torch.randn(1, 12, 2500, dtype=torch.float64)
    ecg[0, 3] = 0.7
    windows = to_ecg_fm_input(ecg)
    assert windows.shape == (2, 12, 2500)
    assert torch.equal(windows[:, 3], torch.zeros(2, 2500, dtype=torch.float64))
    assert torch.isfinite(windows).all()


def test_teacher_follows_the_call_format_and_ignores_tabular():
    teacher = Teacher(FakeEncoder()).eval()
    waveform, tabular = synthetic_inputs(3)
    logits = teacher(waveform, tabular)
    assert logits.shape == (3, 12)
    assert teacher.encoder.windows.shape == (6, 12, 2500)
    torch.testing.assert_close(teacher(waveform, torch.randn_like(tabular)), logits)


@pytest.mark.parametrize("shape", [(2, 12, 1250), (2, 11, 2500), (12, 2500)])
def test_teacher_rejects_wrong_waveform_shape(shape):
    with pytest.raises(ValueError, match="waveform"):
        Teacher(FakeEncoder())(torch.zeros(shape), torch.zeros(2, 7))


def test_missing_checkpoint_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_weights"):
        build_teacher(tmp_path / "missing.pt")


def test_real_ecg_fm_gives_logits_for_each_ecg():
    pytest.importorskip("fairseq_signals", reason="Teacher setup in the README installs fairseq-signals")
    if not ECG_FM_CHECKPOINT.is_file():
        pytest.skip("Run bash scripts/download_weights.sh for the ECG-FM checkpoint")
    teacher = build_teacher(ECG_FM_CHECKPOINT).eval()
    with torch.inference_mode():
        logits = teacher(*synthetic_inputs(2))
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    assert sum(p.numel() for p in teacher.parameters()) == 90_374_016 + 768 * 12 + 12


def separable_features(count: int, seed: int) -> dict[str, np.ndarray]:
    """Features where overall SHD is 1 exactly when the first feature is positive."""
    rng = np.random.default_rng(seed)
    features = (rng.standard_normal((count, 768)) * 12 + 5).astype(np.float32)
    labels = np.zeros((count, 12), dtype=np.float32)
    labels[:, SHD_INDEX] = features[:, 0] > 5
    return {"features": features, "labels": labels, "label_mask": np.ones((count, 12), dtype=bool)}


def test_masked_labels_do_not_change_the_loss():
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    mask = torch.tensor([[True, False], [True, True]])
    first = masked_bce(logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]]), mask)
    second = masked_bce(logits, torch.tensor([[1.0, 1.0], [0.0, 1.0]]), mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0, 0.5, 3.0]), torch.tensor([1.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


@pytest.mark.parametrize("rescale", [True, False])
def test_probe_learns_a_separable_label(rescale):
    train, val = separable_features(2000, 0), separable_features(500, 1)
    head, history = fit_probe(train, val, rescale=rescale, lr=1e-2, max_epochs=20, batch_size=64)
    assert max(history) > 0.8
    expected_mean = train["features"].mean(0) if rescale else np.zeros(768)
    np.testing.assert_allclose(head.feature_mean.numpy(), expected_mean, rtol=1e-4, atol=1e-4)


def test_extracted_features_follow_dataset_order(data_dir):
    dataset = EchoNextDataset(data_dir, "val")
    features = extract_features(Teacher(FakeEncoder()), dataset, batch_size=4)
    assert features["features"].shape == (len(dataset), 768)
    np.testing.assert_array_equal(features["row_id"], dataset.selected_metadata["row_id"].to_numpy())


def test_mean_auroc_skips_outputs_without_a_score():
    metrics = {"a": {"auroc": 0.8}, "b": {"auroc": None}, "c": {"auroc": 0.6}}
    assert mean_auroc(metrics) == pytest.approx(0.7)


def test_training_saves_the_epoch_with_the_best_mean_auroc(data_dir, tmp_path):
    torch.manual_seed(0)
    train, val = EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val")
    # A large step size makes the score move up and down between epochs.
    best = train_teacher(
        Teacher(FakeEncoder()), train, val, tmp_path / "teacher", head_lr=5.0, max_epochs=6, patience=10,
    )
    log = list(csv.DictReader((tmp_path / "teacher" / "log.csv").open()))
    scores = [float(row["val_mean_auroc"]) for row in log]
    assert len(scores) == 6
    expected_epoch = max(range(6), key=lambda i: (scores[i], i)) + 1
    assert torch.load(best, weights_only=True)["epoch"] == expected_epoch


def test_restored_teacher_gives_the_same_logits(data_dir, tmp_path):
    train, val = EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val")
    trained = Teacher(FakeEncoder())
    best = train_teacher(trained, train, val, tmp_path / "teacher", max_epochs=1)
    restored = restore_teacher(Teacher(FakeEncoder()), best).eval()
    inputs = synthetic_inputs(2)
    torch.testing.assert_close(restored(*inputs), trained.eval()(*inputs))


def test_training_refuses_a_folder_that_already_has_a_log(data_dir, tmp_path):
    folder = tmp_path / "teacher"
    folder.mkdir()
    (folder / "log.csv").write_text("epoch\n")
    with pytest.raises(FileExistsError, match="log.csv"):
        train_teacher(Teacher(FakeEncoder()), EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val"), folder)


def test_patient_weights_follow_the_count_to_the_power_alpha():
    keys = np.array(["a", "a", "a", "b"])
    np.testing.assert_allclose(patient_weights(keys, 1.0), [1 / 3, 1 / 3, 1 / 3, 1.0])
    np.testing.assert_allclose(patient_weights(keys, 0.5), [3 ** -0.5, 3 ** -0.5, 3 ** -0.5, 1.0])
    np.testing.assert_allclose(patient_weights(keys, 0.0), [1.0, 1.0, 1.0, 1.0])


def test_head_dropout_changes_outputs_only_in_training_mode():
    features = torch.randn(32, 768)
    head = TeacherHead(dropout=0.5)
    head.train()
    assert not torch.equal(head(features), head(features))
    head.eval()
    assert torch.equal(head(features), head(features))


def test_head_without_dropout_is_the_same_in_training_mode():
    features = torch.randn(32, 768)
    head = TeacherHead().train()
    assert torch.equal(head(features), head(features))


def test_lead_masking_blanks_leads_in_training_mode_only():
    torch.manual_seed(0)
    teacher = Teacher(FakeEncoder(), lead_mask_prob=0.5)
    waveform, tabular = synthetic_inputs(8)
    teacher.eval()(waveform, tabular)
    assert (teacher.encoder.windows.abs().amax(dim=-1) > 0).all()
    teacher.train()(waveform, tabular)
    blank = teacher.encoder.windows.abs().amax(dim=-1) == 0
    assert blank.shape == (16, 12)
    assert blank.any()
    assert not blank.all(dim=1).any()


def test_lead_masking_never_blanks_every_lead_of_an_ecg():
    teacher = Teacher(FakeEncoder(), lead_mask_prob=1.0).train()
    teacher(*synthetic_inputs(4))
    assert (teacher.encoder.windows.abs().amax(dim=-1) > 0).all()


def test_training_with_patient_balancing_saves_a_checkpoint(data_dir, tmp_path):
    best = train_teacher(
        Teacher(FakeEncoder(), head_dropout=0.3, lead_mask_prob=0.2), EchoNextDataset(data_dir, "train"),
        EchoNextDataset(data_dir, "val"), tmp_path / "teacher", max_epochs=1, balance_patients=0.5,
    )
    assert best.is_file()


def tabular_teacher(weight: float = 0.0) -> Teacher:
    """A Teacher that reads the 7 numbers, with every tabular weight set to one value."""
    teacher = Teacher(FakeEncoder(), use_tabular=True).eval()
    with torch.no_grad():
        teacher.tabular_head.weight.fill_(weight)
    return teacher


def test_tabular_teacher_adds_a_weighted_sum_of_the_7_numbers():
    teacher = tabular_teacher(weight=0.5)
    waveform, tabular = synthetic_inputs(3)
    change = teacher(waveform, tabular) - teacher(waveform, torch.zeros_like(tabular))
    expected = (0.5 * tabular.sum(dim=1, keepdim=True)).expand(3, 12)
    torch.testing.assert_close(change, expected, atol=1e-5, rtol=0)


def test_new_tabular_teacher_matches_the_waveform_only_teacher():
    plain, with_tabular = Teacher(FakeEncoder()).eval(), Teacher(FakeEncoder(), use_tabular=True).eval()
    with_tabular.head.load_state_dict(plain.head.state_dict())
    inputs = synthetic_inputs(3)
    torch.testing.assert_close(with_tabular(*inputs), plain(*inputs))


def test_training_learns_tabular_weights(data_dir, tmp_path):
    teacher = Teacher(FakeEncoder(), use_tabular=True)
    train_teacher(teacher, EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val"), tmp_path / "t", max_epochs=1)
    assert teacher.tabular_head.weight.abs().sum() > 0


def test_tabular_rate_of_zero_leaves_tabular_weights_at_zero(data_dir, tmp_path):
    teacher = Teacher(FakeEncoder(), use_tabular=True)
    train_teacher(
        teacher, EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val"), tmp_path / "t",
        max_epochs=1, tabular_lr=0.0,
    )
    assert torch.equal(teacher.tabular_head.weight, torch.zeros(12, 7))


@pytest.mark.parametrize("use_tabular", [True, False])
def test_frozen_teacher_is_rebuilt_with_the_inputs_it_was_trained_on(use_tabular, data_dir, tmp_path, monkeypatch):
    trained = Teacher(FakeEncoder(), use_tabular=use_tabular)
    best = train_teacher(
        trained, EchoNextDataset(data_dir, "train"), EchoNextDataset(data_dir, "val"), tmp_path / "t", max_epochs=1,
    )
    # ECG-FM itself is not needed to check which inputs the loader wires up.
    monkeypatch.setattr(
        "compress_ecg.teacher.build_teacher", lambda path, **options: Teacher(FakeEncoder(), **options),
    )
    frozen = load_frozen_teacher(best)
    inputs = synthetic_inputs(2)
    torch.testing.assert_close(frozen(*inputs), trained.eval()(*inputs))
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
