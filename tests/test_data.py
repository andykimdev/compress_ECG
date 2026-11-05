import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from compress_ecg.data import EchoNextDataset, LABEL_COLUMNS, read_metadata


def edit_meta(root, edit):
    path = root / "EchoNext_metadata_100k.csv"
    meta = pd.read_csv(path)
    edit(meta)
    meta.to_csv(path, index=False)


def test_interleaved_alignment_and_transpose(data_dir):
    dataset = EchoNextDataset(data_dir, "val")
    assert dataset.selected_metadata.row_id.tolist() == [2, 6, 10, 14, 18, 22]
    raw = np.load(data_dir / "EchoNext_val_waveforms.npy")
    for i in range(len(dataset)):
        record = dataset[i]
        assert record["row_id"] == 2 + 4 * i
        assert record["split_row"] == i
        assert record["patient_key"] == f"val-{i}"
        assert record["waveform"].shape == (12, 2500)
        assert record["waveform"].dtype == torch.float32
        np.testing.assert_array_equal(record["waveform"].numpy(), raw[i, 0].T)
        np.testing.assert_array_equal(record["tabular"].numpy(), np.load(dataset.tabular_path)[i])
    assert isinstance(dataset._waveforms, np.memmap)


def test_unknown_component_labels_are_masked(data_dir):
    item = EchoNextDataset(data_dir, "val")[2]
    assert item["labels"][0] == 0
    assert not item["label_mask"][0]
    assert item["label_mask"][-1]


def test_subset_indices_are_repeatable_and_keep_alignment(data_dir):
    a = EchoNextDataset(data_dir, "val", limit=3, seed=7)
    b = EchoNextDataset(data_dir, "val", limit=3, seed=7)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert len(a) == 3
    assert [a[i]["row_id"] for i in range(3)] == a.selected_metadata.row_id.tolist()
    assert len(EchoNextDataset(data_dir, "val", limit=1000)) == 6


def test_pickled_dataset_reopens_mmaps(data_dir):
    original = EchoNextDataset(data_dir, "val")
    restored = pickle.loads(pickle.dumps(original))
    assert restored._waveforms is None
    torch.testing.assert_close(original[1]["waveform"], restored[1]["waveform"])
    assert isinstance(restored._waveforms, np.memmap)


def test_dataloader_collates_fields(data_dir):
    batch = next(iter(DataLoader(EchoNextDataset(data_dir, "train"), batch_size=2)))
    assert batch["waveform"].shape == (2, 12, 2500)
    assert batch["tabular"].shape == (2, 7)
    assert batch["labels"].shape == batch["label_mask"].shape == (2, 12)


def test_spawned_worker_preserves_alignment(data_dir):
    dataset = EchoNextDataset(data_dir, "val")
    loader = DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")
    rows = torch.cat([batch["row_id"] for batch in loader]).tolist()
    assert rows == dataset.selected_metadata.row_id.tolist()


@pytest.mark.parametrize("split", ["no_split", "validation", "oops"])
def test_reject_unofficial_split(data_dir, split):
    with pytest.raises(ValueError, match="Choose train"):
        EchoNextDataset(data_dir, split)


@pytest.mark.parametrize("limit", [0, -1, 2.5, True])
def test_bad_limit(data_dir, limit):
    with pytest.raises(ValueError, match="limit"):
        EchoNextDataset(data_dir, limit=limit)


def test_patient_leakage_rejected_but_no_split_overlap_allowed(data_dir):
    assert len(read_metadata(data_dir)) == 24
    edit_meta(data_dir, lambda meta: meta.loc.__setitem__((1, "patient_key"), "val-0"))
    with pytest.raises(ValueError, match="Patient leakage"):
        EchoNextDataset(data_dir)


def test_duplicate_validation_patient_rejected(data_dir):
    edit_meta(data_dir, lambda meta: meta.loc.__setitem__((6, "patient_key"), "val-0"))
    with pytest.raises(ValueError, match="multiple ECGs"):
        EchoNextDataset(data_dir)


@pytest.mark.parametrize("column,value,message", [
    (LABEL_COLUMNS[0], 2, "Labels must"),
    (LABEL_COLUMNS[-1], np.nan, "SHD labels"),
    ("split", "unknown", "Metadata split"),
])
def test_bad_metadata(data_dir, column, value, message):
    edit_meta(data_dir, lambda meta: meta.loc.__setitem__((2, column), value))
    with pytest.raises(ValueError, match=message):
        EchoNextDataset(data_dir, "val")


@pytest.mark.parametrize("kind,shape", [("waveforms", (5, 1, 2500, 12)), ("tabular_features", (6, 8))])
def test_misaligned_arrays_rejected(data_dir, kind, shape):
    np.save(data_dir / f"EchoNext_val_{kind}.npy", np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="shape must"):
        EchoNextDataset(data_dir, "val")


def test_nonfinite_inputs_rejected(data_dir):
    path = data_dir / "EchoNext_val_tabular_features.npy"
    array = np.load(path)
    array[0, 0] = np.nan
    np.save(path, array)
    with pytest.raises(ValueError, match="Non-finite input"):
        EchoNextDataset(data_dir, "val")[0]


def test_environment_path(data_dir, monkeypatch):
    monkeypatch.setenv("ECHONEXT_DIR", str(data_dir))
    assert len(EchoNextDataset(split="val")) == 6
