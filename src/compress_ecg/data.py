"""Load the official EchoNext 1.1.1 preprocessed arrays without refitting transforms.

Source: https://physionet.org/content/echonext/1.1.1/#files-panel
Array rows match metadata filtered by split in its ORIGINAL order, not by patient.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test", "no_split")
MODEL_SPLITS = SPLITS[:3]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "echonext" / "1.1.1"
LABEL_COLUMNS = (
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
)
SHD_INDEX = LABEL_COLUMNS.index("shd_moderate_or_greater_flag")
# Order inside the .npy arrays. The dataset README lists age last, but the arrays hold it second.
TABULAR_COLUMNS = (
    "sex", "age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected",
)
# PhysioNet ships the lowercase name, while its README uses the other. Linux disks treat them apart.
METADATA_FILENAMES = ("echonext_metadata_100k.csv", "EchoNext_metadata_100k.csv")
AUDIT_COLUMNS = ("race_ethnicity", "sex", "age_at_ecg", "location_setting")
# Parts of the train split for pilot runs: fit trains a model and dev judges it. See part_rows.
PARTS = ("fit", "dev")
# Metadata flag that is 1 on exactly one ECG per patient in train, val, and test: the latest one.
RECENT_COLUMN = "most_recent_ecg"


def find_metadata_file(root: Path) -> Path | None:
    """Return the metadata CSV in a folder, under either name, or None."""
    for name in METADATA_FILENAMES:
        if (root / name).is_file():
            return root / name
    return None


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Use data_dir, then ECHONEXT_DIR, then the repository's data folder."""
    value = data_dir if data_dir is not None else os.environ.get("ECHONEXT_DIR", DEFAULT_DATA_DIR)
    root = Path(value).expanduser().resolve()
    if find_metadata_file(root) is None:
        raise FileNotFoundError(f"No EchoNext metadata CSV in {root}. Set ECHONEXT_DIR or pass data_dir")
    return root


def read_metadata(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Check official split integrity; retain the CSV's zero-based positional row ID."""
    root = resolve_data_dir(data_dir)
    meta = pd.read_csv(find_metadata_file(root), dtype={"patient_key": "string"})
    required = {"patient_key", "split", *LABEL_COLUMNS}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if "row_id" in meta.columns:
        raise ValueError("Metadata column row_id is reserved for the original CSV row position")
    if meta["split"].isna().any() or not meta["split"].isin(SPLITS).all():
        raise ValueError(f"Metadata split must be one of {SPLITS}")
    if meta["patient_key"].isna().any() or meta["patient_key"].str.strip().eq("").any():
        raise ValueError("Metadata contains missing patient_key values")
    # no_split can contain earlier ECGs from validation/test patients. Never treat
    # those rows as extra training examples or include them in this overlap check.
    official = meta[meta["split"].isin(MODEL_SPLITS)]
    if official.groupby("patient_key")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage between official train/val/test splits")
    for split in ("val", "test"):
        if meta.loc[meta["split"].eq(split), "patient_key"].duplicated().any():
            raise ValueError(f"Official {split} split contains multiple ECGs per patient")
    meta.insert(0, "row_id", np.arange(len(meta), dtype=np.int64))
    return meta


def held_out_patients(patient_keys: Sequence[str] | np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """
    Pick the patients that are held out of training to judge pilot runs.

    Args:
        patient_keys: The patient of every Training ECG. Repeats and row order do not matter.
        fraction: Share of the unique patients to hold out, above 0 and below 1.
        seed: Seed of the draw. The same seed and patients give the same result.

    Returns:
        The held-out patient keys, unique and sorted. Their count is fraction times the
        number of unique patients, rounded to the nearest whole number.

    Raises:
        ValueError: When fraction is not between 0 and 1, or one side would get no patient.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"holdout_fraction must be above 0 and below 1, got {fraction}")
    # Drawing from the sorted unique keys makes the result independent of row order.
    unique = np.unique(np.asarray(patient_keys, dtype=str))
    count = round(fraction * len(unique))
    if count == 0 or count == len(unique):
        raise ValueError(f"holdout_fraction {fraction} leaves no patient on one side of {len(unique)} patients")
    chosen = np.random.default_rng(seed).choice(len(unique), count, replace=False)
    return unique[np.sort(chosen)]


def part_rows(metadata: pd.DataFrame, part: str, fraction: float, seed: int = 0) -> np.ndarray:
    """
    Return the rows of the Training metadata that belong to one part.

    Args:
        metadata: The Training rows, with patient_key and, for dev, the most_recent_ecg flag.
        part: fit for every ECG of the patients that are not held out, or dev for exactly
            one ECG per held-out patient, the one flagged as most recent. Validation and Test
            also hold one ECG per patient, all flagged, so dev scores compare with theirs.
        fraction: Share of the patients to hold out. See held_out_patients.
        seed: Seed of the held-out draw.

    Returns:
        Row positions in metadata, in increasing order.

    Raises:
        ValueError: When part is unknown, the flag column is missing, or a held-out patient
            does not have exactly one flagged ECG.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    keys = metadata["patient_key"].to_numpy(dtype=str)
    held_out = np.isin(keys, held_out_patients(keys, fraction, seed))
    if part == "fit":
        return np.flatnonzero(~held_out)
    if RECENT_COLUMN not in metadata.columns:
        raise ValueError(f"The dev part needs the metadata column {RECENT_COLUMN}")
    rows = np.flatnonzero(held_out & metadata[RECENT_COLUMN].eq(1).to_numpy())
    if not np.array_equal(np.sort(keys[rows]), np.unique(keys[held_out])):
        raise ValueError(f"Every held-out patient needs exactly one ECG with {RECENT_COLUMN} set to 1")
    return rows


def holdout_tag(fraction: float, seed: int = 0) -> str:
    """
    Name a holdout for the files and folders that depend on it.

    Args:
        fraction: Share of the Training patients that is held out. 0 means no holdout.
        seed: Seed of the held-out draw.

    Returns:
        An empty string without a holdout, and otherwise a tag such as _ho0.2_hs0.
    """
    return f"_ho{fraction:g}_hs{seed}" if fraction else ""


def kept_patients(
    patient_keys: Sequence[str] | np.ndarray, shd: Sequence[bool] | np.ndarray, fraction: float, seed: int = 0,
) -> np.ndarray:
    """
    Pick a share of the patients to train on, for one point of a learning curve.

    Args:
        patient_keys: The patient of every ECG. Repeats and row order do not matter.
        shd: Whether each ECG has the SHD label. A patient with SHD on any ECG counts as with SHD.
        fraction: Share of the patients to keep, above 0 and at most 1.
        seed: Seed of the draw. The same seed and patients give the same result.

    Returns:
        The kept patient keys, unique and sorted. The share is taken separately among the
        patients with SHD and those without, each count rounded to the nearest whole number.

    Raises:
        ValueError: When fraction is not above 0 and at most 1, or no patient is kept.

    Notes:
        With one seed, the patients of a smaller share are all inside every larger share.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"patient_fraction must be above 0 and at most 1, got {fraction}")
    # Drawing from the sorted unique keys makes the result independent of row order.
    unique, patient_of = np.unique(np.asarray(patient_keys, dtype=str), return_inverse=True)
    with_shd = np.bincount(patient_of, weights=np.asarray(shd, dtype=float), minlength=len(unique)) > 0
    rng = np.random.default_rng(seed)
    kept = []
    for group in (False, True):
        # The order does not depend on the fraction, so a smaller share is the start of a larger one.
        order = rng.permutation(np.flatnonzero(with_shd == group))
        kept.append(order[:round(fraction * len(order))])
    chosen = np.sort(np.concatenate(kept))
    if not len(chosen):
        raise ValueError(f"patient_fraction {fraction} keeps no patient of {len(unique)}")
    return unique[chosen]


def subsample_tag(fraction: float, seed: int = 0) -> str:
    """
    Name a patient subsample for the files and folders that depend on it.

    Args:
        fraction: Share of the fit patients that is kept. 1 means all of them.
        seed: Seed of the draw of kept patients.

    Returns:
        An empty string for all patients, and otherwise a tag such as _pf0.125_ps0.
    """
    return f"_pf{fraction:g}_ps{seed}" if fraction != 1 else ""


class EchoNextDataset(Dataset):
    """Memory-mapped, map-style dataset shared by teacher, student and Mini.

    Items contain float32 waveform [12,2500], tabular [7], labels [12], a
    boolean label_mask [12], original row_id, split-local split_row, patient_key.
    Missing component labels are zero-filled ONLY alongside a False mask. Never
    include them in a loss/metric as negatives. The SHD label must be observed.
    Use the supplied tabular array; metadata echo measurements are never inputs.
    Set load_waveforms=False to use tabular features and labels without ECG arrays.

    For pilot runs the train split can be served in two parts that share no patient.
    Pass part fit or dev with holdout_fraction and holdout_seed. See part_rows. The
    group attribute names what is served: the split, or the part when one is chosen.
    limit draws its ECGs from inside the chosen part. For a learning curve the fit part can
    be cut to a share of its patients with patient_fraction and patient_seed. See kept_patients.
    """

    def __init__(
        self, data_dir: str | Path | None = None, split: str = "train", *,
        limit: int | None = None, seed: int = 42, load_waveforms: bool = True, part: str | None = None,
        holdout_fraction: float = 0.0, holdout_seed: int = 0, patient_fraction: float = 1.0, patient_seed: int = 0,
    ):
        if split not in MODEL_SPLITS:
            raise ValueError("Choose train, val or test; no_split is excluded from experiments")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        if part is not None and split != "train":
            raise ValueError(f"Only the train split has fit and dev parts. The {split} split is always served whole")
        if part is None and holdout_fraction != 0:
            raise ValueError("Choose part fit or dev when holdout_fraction is not 0")
        if patient_fraction != 1 and part != "fit":
            raise ValueError("Only the fit part can be cut to a share of its patients")
        self.root = resolve_data_dir(data_dir)
        self.split = split
        self.part = part
        self.group = split if part is None else part
        self.holdout_fraction = holdout_fraction
        self.holdout_seed = holdout_seed
        self.seed = seed
        self.metadata = read_metadata(self.root).query("split == @split").reset_index(drop=True)
        self.metadata.insert(1, "split_row", np.arange(len(self.metadata), dtype=np.int64))
        self.waveform_path = self.root / f"EchoNext_{split}_waveforms.npy" if load_waveforms else None
        self.tabular_path = self.root / f"EchoNext_{split}_tabular_features.npy"
        self._waveforms = self._tabular = None
        self._pid = None
        self._open_arrays()
        count = len(self.metadata)
        if not count:
            raise ValueError(f"Official {split} split is empty")
        if self._waveforms is not None and self._waveforms.shape != (count, 1, 2500, 12):
            raise ValueError(f"Waveform shape must be {(count, 1, 2500, 12)}, got {self._waveforms.shape}")
        tabular_shape = (count, len(TABULAR_COLUMNS))
        if self._tabular.shape != tabular_shape:
            raise ValueError(f"Tabular shape must be {tabular_shape}, got {self._tabular.shape}")
        for array in (self._waveforms, self._tabular):
            if array is not None and (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)):
                raise ValueError("Expected real numeric arrays")
        labels = self.metadata[list(LABEL_COLUMNS)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not (np.isnan(labels) | (labels == 0) | (labels == 1)).all():
            raise ValueError("Labels must be 0, 1 or missing")
        if np.isnan(labels[:, SHD_INDEX]).any():
            raise ValueError("Composite SHD labels must not be missing")
        self.label_mask = np.isfinite(labels)
        self.labels = np.nan_to_num(labels, nan=0.0)
        self.indices = np.arange(count)
        if part is not None:
            self.indices = part_rows(self.metadata, part, holdout_fraction, holdout_seed)
        if patient_fraction != 1:
            keys = self.metadata["patient_key"].to_numpy(dtype=str)[self.indices]
            kept = kept_patients(keys, self.labels[self.indices, SHD_INDEX] == 1, patient_fraction, patient_seed)
            self.indices = self.indices[np.isin(keys, kept)]
        if limit is not None and limit < len(self.indices):
            chosen = np.random.default_rng(seed).choice(len(self.indices), limit, replace=False)
            self.indices = self.indices[np.sort(chosen)]

    def _open_arrays(self):
        # Reopen after fork/spawn instead of copying multi-GB arrays to workers.
        if self._pid != os.getpid() or self._tabular is None:
            if self.waveform_path is not None:
                self._waveforms = np.load(self.waveform_path, mmap_mode="r", allow_pickle=False)
            self._tabular = np.load(self.tabular_path, mmap_mode="r", allow_pickle=False)
            self._pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_waveforms=None, _tabular=None, _pid=None)
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        self._open_arrays()
        row = int(self.indices[index])
        # Copy one ECG so torch never shares read-only mmap storage.
        tabular = np.array(self._tabular[row], dtype=np.float32, copy=True)
        if not np.isfinite(tabular).all():
            raise ValueError(f"Non-finite input in tabular features at {self.split} split row {row}")
        record = self.metadata.iloc[row]
        sample = {
            "tabular": torch.from_numpy(tabular),
            "labels": torch.from_numpy(self.labels[row].copy()),
            "label_mask": torch.from_numpy(self.label_mask[row].copy()),
            "row_id": int(record["row_id"]),
            "split_row": row,
            "patient_key": str(record["patient_key"]),
        }
        if self._waveforms is not None:
            waveform = np.array(self._waveforms[row, 0].T, dtype=np.float32, copy=True)
            if not np.isfinite(waveform).all():
                raise ValueError(f"Non-finite input in waveform at {self.split} split row {row}")
            sample["waveform"] = torch.from_numpy(waveform)
        return sample

    @property
    def selected_tabular(self) -> np.ndarray:
        """Return the seven features in dataset order for tabular model training or scoring."""
        self._open_arrays()
        features = self._tabular[self.indices].astype(np.float32, copy=False)
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite input in {self.split} tabular features")
        return features

    @property
    def selected_metadata(self) -> pd.DataFrame:
        """Rows in dataset order; filtering never changes the array lookup indices."""
        return self.metadata.iloc[self.indices].copy().reset_index(drop=True)


def open_group(
    data_dir: str | Path | None, name: str, *, limit: int | None = None,
    holdout_fraction: float = 0.0, holdout_seed: int = 0, patient_fraction: float = 1.0, patient_seed: int = 0,
) -> EchoNextDataset:
    """
    Open one group of ECGs by the name its run folder carries.

    Args:
        data_dir: Folder with the EchoNext files, or None for the default.
        name: train, val, or test for a whole split, or fit or dev for a part of train.
        limit: Use only this many ECGs of the group.
        holdout_fraction: Share of the Training patients held out. Only a part uses it.
        holdout_seed: Seed of the held-out draw. Only a part uses it.
        patient_fraction: Share of the fit patients to keep. Only fit uses it, so dev stays whole.
        patient_seed: Seed of the draw of kept patients. Only fit uses it.

    Returns:
        The dataset whose group attribute equals name.
    """
    if name in PARTS:
        return EchoNextDataset(
            data_dir, "train", limit=limit, part=name, holdout_fraction=holdout_fraction, holdout_seed=holdout_seed,
            patient_fraction=patient_fraction if name == "fit" else 1.0, patient_seed=patient_seed,
        )
    return EchoNextDataset(data_dir, name, limit=limit)
