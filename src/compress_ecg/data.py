"""Load the official EchoNext 1.1.1 preprocessed arrays without refitting transforms.

Source: https://physionet.org/content/echonext/1.1.1/#files-panel
Array rows match metadata filtered by split in its ORIGINAL order, not by patient.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test", "no_split")
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
    official = meta[meta["split"].isin(("train", "val", "test"))]
    if official.groupby("patient_key")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage between official train/val/test splits")
    for split in ("val", "test"):
        if meta.loc[meta["split"].eq(split), "patient_key"].duplicated().any():
            raise ValueError(f"Official {split} split contains multiple ECGs per patient")
    meta.insert(0, "row_id", np.arange(len(meta), dtype=np.int64))
    return meta


class EchoNextDataset(Dataset):
    """Memory-mapped, map-style dataset shared by teacher, student and Mini.

    Items contain float32 waveform [12,2500], tabular [7], labels [12], a
    boolean label_mask [12], original row_id, split-local split_row, patient_key.
    Missing component labels are zero-filled ONLY alongside a False mask. Never
    include them in a loss/metric as negatives. The SHD label must be observed.
    Use the supplied tabular array; metadata echo measurements are never inputs.
    """

    def __init__(
        self, data_dir: str | Path | None = None, split: str = "train", *,
        limit: int | None = None, seed: int = 42,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Choose train, val or test; no_split is excluded from experiments")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        self.root = resolve_data_dir(data_dir)
        self.split = split
        self.seed = seed
        self.metadata = read_metadata(self.root).query("split == @split").reset_index(drop=True)
        self.metadata.insert(1, "split_row", np.arange(len(self.metadata), dtype=np.int64))
        self.waveform_path = self.root / f"EchoNext_{split}_waveforms.npy"
        self.tabular_path = self.root / f"EchoNext_{split}_tabular_features.npy"
        self._waveforms = self._tabular = None
        self._pid = None
        self._open_arrays()
        count = len(self.metadata)
        if not count:
            raise ValueError(f"Official {split} split is empty")
        if self._waveforms.shape != (count, 1, 2500, 12):
            raise ValueError(f"Waveform shape must be {(count, 1, 2500, 12)}, got {self._waveforms.shape}")
        if self._tabular.shape != (count, 7):
            raise ValueError(f"Tabular shape must be {(count, 7)}, got {self._tabular.shape}")
        for array in (self._waveforms, self._tabular):
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError("Expected real numeric arrays")
        labels = self.metadata[list(LABEL_COLUMNS)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        if not (np.isnan(labels) | (labels == 0) | (labels == 1)).all():
            raise ValueError("Labels must be 0, 1 or missing")
        if np.isnan(labels[:, SHD_INDEX]).any():
            raise ValueError("Composite SHD labels must not be missing")
        self.label_mask = np.isfinite(labels)
        self.labels = np.nan_to_num(labels, nan=0.0)
        self.indices = np.arange(count)
        if limit is not None and limit < count:
            self.indices = np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))

    def _open_arrays(self):
        # Reopen after fork/spawn instead of copying multi-GB arrays to workers.
        if self._pid != os.getpid() or self._waveforms is None:
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
        waveform = np.array(self._waveforms[row, 0].T, dtype=np.float32, copy=True)
        tabular = np.array(self._tabular[row], dtype=np.float32, copy=True)
        if not np.isfinite(waveform).all() or not np.isfinite(tabular).all():
            raise ValueError(f"Non-finite input at {self.split} split row {row}")
        record = self.metadata.iloc[row]
        return {
            "waveform": torch.from_numpy(waveform),
            "tabular": torch.from_numpy(tabular),
            "labels": torch.from_numpy(self.labels[row].copy()),
            "label_mask": torch.from_numpy(self.label_mask[row].copy()),
            "row_id": int(record["row_id"]),
            "split_row": row,
            "patient_key": str(record["patient_key"]),
        }

    @property
    def selected_metadata(self) -> pd.DataFrame:
        """Rows in dataset order; filtering never changes the array lookup indices."""
        return self.metadata.iloc[self.indices].copy().reset_index(drop=True)
