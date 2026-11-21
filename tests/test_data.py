import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from compress_ecg import data as data_module
from compress_ecg.data import (
    DEFAULT_DATA_DIR, EchoNextDataset, LABEL_COLUMNS, TABULAR_COLUMNS, held_out_patients, holdout_tag, kept_patients,
    open_group, subsample_tag,
    read_metadata, resolve_data_dir,
)


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


def test_lowercase_metadata_filename_from_physionet_is_found(data_dir):
    # A case-sensitive disk, such as Linux, needs the name exactly as PhysioNet ships it.
    (data_dir / "EchoNext_metadata_100k.csv").rename(data_dir / "echonext_metadata_100k.csv")
    assert (data_dir / "echonext_metadata_100k.csv").is_file()
    assert len(read_metadata(data_dir)) == 24


def test_default_folder_is_used_when_nothing_is_set(data_dir, monkeypatch):
    monkeypatch.delenv("ECHONEXT_DIR", raising=False)
    monkeypatch.setattr(data_module, "DEFAULT_DATA_DIR", data_dir)
    assert resolve_data_dir() == data_dir.resolve()


def test_missing_data_folder_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("ECHONEXT_DIR", raising=False)
    monkeypatch.setattr(data_module, "DEFAULT_DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="ECHONEXT_DIR"):
        resolve_data_dir()


def test_tabular_column_names_match_the_real_array():
    if not (DEFAULT_DATA_DIR / "EchoNext_val_tabular_features.npy").is_file():
        pytest.skip("Real EchoNext files are not on this machine")
    meta = read_metadata(DEFAULT_DATA_DIR).query("split == 'val'").reset_index(drop=True)
    array = np.load(DEFAULT_DATA_DIR / "EchoNext_val_tabular_features.npy")
    for index, name in enumerate(TABULAR_COLUMNS):
        raw = meta[name].eq("male").astype(float) if name == "sex" else meta[name]
        correlation = pd.Series(array[:, index]).corr(raw, method="spearman")
        assert correlation > .99, f"Array column {index} is not {name}: {correlation:.3f}"


# What the holdout_dir fixture holds: the patient of the Training ECGs at CSV rows 1, 5, 9, 13,
# 17, 21, and the row of each patient's most recent ECG.
TRAIN_PATIENTS = ("a", "b", "b", "c", "c", "d")
RECENT_ROW_OF = {"a": 1, "b": 9, "c": 17, "d": 21}


def parts(root, **options):
    """Open the fit and dev parts of the Training ECGs with one holdout."""
    options = {"holdout_fraction": 0.5, "holdout_seed": 0, **options}
    return EchoNextDataset(root, "train", part="fit", **options), EchoNextDataset(root, "train", part="dev", **options)


def test_held_out_patient_count_follows_the_fraction():
    keys = [f"p{index:03d}" for index in range(200) for _ in range(index % 3 + 1)]
    held = held_out_patients(keys, 0.2, seed=0)
    assert len(held) == 40
    assert len(set(held)) == 40
    assert set(held) <= set(keys)
    assert len(held_out_patients(keys, 0.333, seed=0)) == 67


def test_held_out_patients_do_not_depend_on_row_order_or_ecg_count():
    keys = np.array([f"p{index:03d}" for index in range(50)])
    shuffled = np.random.default_rng(1).permutation(np.repeat(keys, 3))
    np.testing.assert_array_equal(held_out_patients(shuffled, 0.3, seed=5), held_out_patients(keys, 0.3, seed=5))


def test_same_holdout_seed_repeats_and_another_seed_differs():
    keys = [f"p{index:03d}" for index in range(100)]
    first = held_out_patients(keys, 0.2, seed=0)
    np.testing.assert_array_equal(first, held_out_patients(keys, 0.2, seed=0))
    assert set(first) != set(held_out_patients(keys, 0.2, seed=1))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_holdout_fraction_outside_zero_to_one_is_rejected(fraction):
    with pytest.raises(ValueError, match="above 0 and below 1"):
        held_out_patients(["a", "b", "c"], fraction, seed=0)


@pytest.mark.parametrize("fraction", [0.1, 0.9])
def test_holdout_that_leaves_one_side_without_patients_is_rejected(fraction):
    with pytest.raises(ValueError, match="no patient"):
        held_out_patients(["a", "b", "c"], fraction, seed=0)


def test_fit_and_dev_share_no_patient_and_cover_every_training_patient(holdout_dir):
    fit, dev = parts(holdout_dir)
    fit_patients = set(fit.selected_metadata["patient_key"])
    dev_patients = set(dev.selected_metadata["patient_key"])
    assert len(dev_patients) == 2
    assert fit_patients & dev_patients == set()
    assert fit_patients | dev_patients == {"a", "b", "c", "d"}


def test_fit_keeps_every_ecg_of_the_patients_that_are_not_held_out(holdout_dir):
    fit, dev = parts(holdout_dir)
    kept = set(TRAIN_PATIENTS) - set(dev.selected_metadata["patient_key"])
    expected_rows = [1 + 4 * local for local, patient in enumerate(TRAIN_PATIENTS) if patient in kept]
    assert expected_rows != []
    assert fit.selected_metadata["row_id"].tolist() == expected_rows


def test_dev_has_exactly_the_most_recent_ecg_of_each_held_out_patient(holdout_dir):
    _, dev = parts(holdout_dir)
    patients = dev.selected_metadata["patient_key"].tolist()
    assert len(patients) == 2
    assert len(set(patients)) == 2
    assert dev.selected_metadata["row_id"].tolist() == sorted(RECENT_ROW_OF[patient] for patient in patients)


def test_parts_repeat_with_the_same_holdout_seed(holdout_dir):
    first_fit, first_dev = parts(holdout_dir, holdout_seed=3)
    second_fit, second_dev = parts(holdout_dir, holdout_seed=3)
    np.testing.assert_array_equal(first_fit.indices, second_fit.indices)
    np.testing.assert_array_equal(first_dev.indices, second_dev.indices)


def test_parts_change_with_the_holdout_seed(holdout_dir):
    held_by_seed = {
        tuple(parts(holdout_dir, holdout_seed=seed)[1].selected_metadata["patient_key"]) for seed in range(6)
    }
    # Six seeds over 4 patients must not all hold out the same pair.
    assert len(held_by_seed) > 1


@pytest.mark.parametrize("part", ["fit", "dev"])
def test_items_of_a_part_come_from_one_original_row(holdout_dir, part):
    dataset = EchoNextDataset(holdout_dir, "train", part=part, holdout_fraction=0.5, holdout_seed=0)
    assert len(dataset) >= 2
    wave = np.sin(np.arange(2500) / 20) * .1
    for index in range(len(dataset)):
        item = dataset[index]
        row = item["row_id"]
        assert row == dataset.selected_metadata["row_id"][index]
        assert item["split_row"] == (row - 1) // 4
        assert item["patient_key"] == TRAIN_PATIENTS[(row - 1) // 4]
        # The fixture writes each original row number into its waveform, its 7 numbers, and its labels.
        expected_waveform = row * .01 + np.arange(12)[:, None] * .001 + wave[None, :]
        np.testing.assert_allclose(item["waveform"].numpy(), expected_waveform, atol=1e-6)
        np.testing.assert_allclose(item["tabular"].numpy(), row * .01 + np.arange(7) * .001, atol=1e-6)
        assert item["labels"].tolist() == [float((row // 4) % 2)] * 12


def test_no_holdout_reproduces_the_whole_training_split(holdout_dir):
    plain = EchoNextDataset(holdout_dir, "train")
    explicit = EchoNextDataset(holdout_dir, "train", part=None, holdout_fraction=0.0)
    assert plain.selected_metadata["row_id"].tolist() == [1, 5, 9, 13, 17, 21]
    assert explicit.selected_metadata["row_id"].tolist() == [1, 5, 9, 13, 17, 21]
    assert plain.group == "train"


def test_limit_without_a_part_draws_the_same_rows_as_before(data_dir):
    dataset = EchoNextDataset(data_dir, "train", limit=3, seed=7)
    # The draw the dataset made before parts existed.
    expected = np.sort(np.random.default_rng(7).choice(6, 3, replace=False))
    np.testing.assert_array_equal(dataset.indices, expected)


def test_limit_applies_inside_the_chosen_part(holdout_dir):
    # Holdout seed 0 holds out patients c and d, whose most recent ECGs are CSV rows 17 and 21.
    fit, dev = parts(holdout_dir)
    assert dev.selected_metadata["row_id"].tolist() == [17, 21]
    limited_dev = EchoNextDataset(holdout_dir, "train", part="dev", holdout_fraction=0.5, holdout_seed=0, limit=1)
    assert len(limited_dev) == 1
    assert limited_dev.selected_metadata["row_id"][0] in (17, 21)
    limited_fit = EchoNextDataset(holdout_dir, "train", part="fit", holdout_fraction=0.5, holdout_seed=0, limit=2)
    assert len(limited_fit) == 2
    assert set(limited_fit.selected_metadata["row_id"]) < {1, 5, 9}


def test_limit_above_the_size_of_a_part_keeps_the_whole_part(holdout_dir):
    # The fit part has 3 of the 6 Training ECGs, so a limit of 4 is above the part but below the split.
    limited = EchoNextDataset(holdout_dir, "train", part="fit", holdout_fraction=0.5, holdout_seed=0, limit=4)
    assert limited.selected_metadata["row_id"].tolist() == [1, 5, 9]


def test_a_part_names_its_group_of_ecgs(holdout_dir):
    fit, dev = parts(holdout_dir)
    assert (fit.split, fit.group, dev.split, dev.group) == ("train", "fit", "train", "dev")
    assert EchoNextDataset(holdout_dir, "val").group == "val"


@pytest.mark.parametrize("split", ["val", "test"])
def test_a_part_of_validation_or_test_is_rejected(holdout_dir, split):
    with pytest.raises(ValueError, match="Only the train split"):
        EchoNextDataset(holdout_dir, split, part="dev", holdout_fraction=0.5)


def test_an_unknown_part_is_rejected(holdout_dir):
    with pytest.raises(ValueError, match="part must be"):
        EchoNextDataset(holdout_dir, "train", part="holdout", holdout_fraction=0.5)


def test_a_part_without_a_holdout_fraction_is_rejected(holdout_dir):
    with pytest.raises(ValueError, match="above 0 and below 1"):
        EchoNextDataset(holdout_dir, "train", part="fit")


def test_a_holdout_fraction_without_a_part_is_rejected(holdout_dir):
    with pytest.raises(ValueError, match="part"):
        EchoNextDataset(holdout_dir, "train", holdout_fraction=0.5)


def test_dev_needs_the_most_recent_ecg_column(data_dir):
    edit_meta(data_dir, lambda meta: meta.loc.__setitem__((meta["split"].eq("train"), "patient_key"), TRAIN_PATIENTS))
    with pytest.raises(ValueError, match="most_recent_ecg"):
        EchoNextDataset(data_dir, "train", part="dev", holdout_fraction=0.5)


@pytest.mark.parametrize("flags", [(1, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 1)])
def test_dev_rejects_a_held_out_patient_without_exactly_one_most_recent_ecg(holdout_dir, flags):
    edit_meta(holdout_dir, lambda meta: meta.loc.__setitem__((meta["split"].eq("train"), "most_recent_ecg"), flags))
    # Patients b and c now have 2 or 0 flagged ECGs. Holding out 3 of the 4 patients includes one of them.
    with pytest.raises(ValueError, match="exactly one"):
        EchoNextDataset(holdout_dir, "train", part="dev", holdout_fraction=0.75)


@pytest.mark.parametrize("fraction, seed, expected", [
    (0.2, 0, "_ho0.2_hs0"), (0.25, 3, "_ho0.25_hs3"), (0.0, 0, ""), (0.0, 5, ""),
])
def test_holdout_tag_names_the_fraction_and_seed_and_is_empty_without_a_holdout(fraction, seed, expected):
    assert holdout_tag(fraction, seed) == expected


@pytest.mark.parametrize("name, expected_rows", [
    ("train", [1, 5, 9, 13, 17, 21]), ("val", [2, 6, 10, 14, 18, 22]), ("fit", [1, 5, 9]), ("dev", [17, 21]),
])
def test_a_group_opens_by_name_as_a_whole_split_or_a_part(holdout_dir, name, expected_rows):
    dataset = open_group(holdout_dir, name, holdout_fraction=0.5, holdout_seed=0)
    assert dataset.group == name
    assert dataset.selected_metadata["row_id"].tolist() == expected_rows


def test_a_group_opened_with_a_limit_has_that_many_ecgs(holdout_dir):
    assert len(open_group(holdout_dir, "fit", limit=2, holdout_fraction=0.5)) == 2
    assert len(open_group(holdout_dir, "val", limit=4)) == 4


# 200 patients with 1 to 3 ECGs each. Patients 0 to 79 have SHD, on their first ECG only.
CURVE_KEYS = [f"p{patient:03d}" for patient in range(200) for _ in range(patient % 3 + 1)]
CURVE_SHD = [patient < 80 and ecg == 0 for patient in range(200) for ecg in range(patient % 3 + 1)]


def test_kept_patient_count_follows_the_fraction_among_patients_with_and_without_shd():
    kept = kept_patients(CURVE_KEYS, CURVE_SHD, 0.25, seed=0)
    assert len(set(kept)) == len(kept) == 50
    assert sum(int(key[1:]) < 80 for key in kept) == 20
    assert set(kept) <= set(CURVE_KEYS)


def test_a_smaller_share_of_patients_lies_inside_every_larger_share():
    eighth, quarter, half, everyone = (
        set(kept_patients(CURVE_KEYS, CURVE_SHD, fraction, seed=0)) for fraction in (0.125, 0.25, 0.5, 1.0)
    )
    assert eighth < quarter < half < everyone
    assert everyone == set(CURVE_KEYS)


def test_kept_patients_do_not_depend_on_row_order():
    order = np.random.default_rng(1).permutation(len(CURVE_KEYS))
    shuffled = kept_patients(np.array(CURVE_KEYS)[order], np.array(CURVE_SHD)[order], 0.25, seed=4)
    np.testing.assert_array_equal(shuffled, kept_patients(CURVE_KEYS, CURVE_SHD, 0.25, seed=4))


def test_same_patient_seed_repeats_and_another_seed_differs():
    first = kept_patients(CURVE_KEYS, CURVE_SHD, 0.25, seed=0)
    np.testing.assert_array_equal(first, kept_patients(CURVE_KEYS, CURVE_SHD, 0.25, seed=0))
    assert set(first) != set(kept_patients(CURVE_KEYS, CURVE_SHD, 0.25, seed=1))


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_patient_fraction_outside_zero_to_one_is_rejected(fraction):
    with pytest.raises(ValueError, match="patient_fraction"):
        kept_patients(["a", "b", "c"], [False, True, False], fraction, seed=0)


def test_patient_fraction_that_keeps_no_patient_is_rejected():
    with pytest.raises(ValueError, match="keeps no patient"):
        kept_patients(["a", "b", "c"], [False, False, False], 0.1, seed=0)


def subsampled_fit(root, **options):
    """Open half of the fit patients of the holdout that keeps patients a and b for training."""
    # Patient b loses its SHD label, so a and b are both without SHD and half of them is 1 patient.
    edit_meta(root, lambda meta: meta.loc.__setitem__((5, list(LABEL_COLUMNS)), 0))
    options = {"holdout_fraction": 0.5, "holdout_seed": 0, "patient_fraction": 0.5, **options}
    return EchoNextDataset(root, "train", part="fit", **options)


def test_subsampled_fit_holds_every_ecg_of_the_kept_patients_and_no_other(holdout_dir):
    # Patient a owns CSV row 1 and patient b owns rows 5 and 9. Each seed keeps one of them whole.
    rows_by_seed = {
        tuple(subsampled_fit(holdout_dir, patient_seed=seed).selected_metadata["row_id"]) for seed in range(8)
    }
    assert rows_by_seed == {(1,), (5, 9)}


def test_subsampled_fit_repeats_with_the_same_patient_seed(holdout_dir):
    first = subsampled_fit(holdout_dir, patient_seed=2).selected_metadata["row_id"].tolist()
    assert subsampled_fit(holdout_dir, patient_seed=2).selected_metadata["row_id"].tolist() == first


def test_patient_fraction_of_one_leaves_fit_whole(holdout_dir):
    assert subsampled_fit(holdout_dir, patient_fraction=1.0).selected_metadata["row_id"].tolist() == [1, 5, 9]


@pytest.mark.parametrize("part, holdout", [("dev", 0.5), (None, 0.0)])
def test_only_the_fit_part_can_be_subsampled(holdout_dir, part, holdout):
    with pytest.raises(ValueError, match="Only the fit part"):
        EchoNextDataset(holdout_dir, "train", part=part, holdout_fraction=holdout, patient_fraction=0.5)


@pytest.mark.parametrize("fraction, seed, expected", [
    (0.125, 0, "_pf0.125_ps0"), (0.5, 3, "_pf0.5_ps3"), (1.0, 0, ""), (1.0, 5, ""),
])
def test_subsample_tag_names_the_fraction_and_seed_and_is_empty_for_all_patients(fraction, seed, expected):
    assert subsample_tag(fraction, seed) == expected


def test_a_group_opened_with_a_patient_fraction_subsamples_fit_and_leaves_dev_whole(holdout_dir):
    edit_meta(holdout_dir, lambda meta: meta.loc.__setitem__((5, list(LABEL_COLUMNS)), 0))
    options = {"holdout_fraction": 0.5, "holdout_seed": 0, "patient_fraction": 0.5}
    assert open_group(holdout_dir, "fit", **options).selected_metadata["row_id"].tolist() in ([1], [5, 9])
    assert open_group(holdout_dir, "dev", **options).selected_metadata["row_id"].tolist() == [17, 21]
