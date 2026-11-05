import numpy as np
import pandas as pd
import pytest

from compress_ecg.data import LABEL_COLUMNS


@pytest.fixture
def data_dir(tmp_path):
    """Small official-format dataset with interleaved splits and distinctive ECGs."""
    root = tmp_path / "data"
    root.mkdir()
    records = []
    for index in range(6):
        for split in ("test", "train", "val", "no_split"):
            patient = f"{split}-{index}"
            if split == "no_split":
                patient = f"val-{index}"  # Allowed: unused earlier ECG.
            records.append({
                "patient_key": patient, "split": split,
                "race_ethnicity": "unknown", "location_setting": "outpatient",
                "sex": "female" if index % 2 else "male", "age_at_ecg": 20 + index,
                **{column: index % 2 for column in LABEL_COLUMNS},
            })
    meta = pd.DataFrame(records)
    meta.loc[meta["split"].eq("val").to_numpy().nonzero()[0][2], LABEL_COLUMNS[0]] = np.nan
    meta.to_csv(root / "EchoNext_metadata_100k.csv", index=False)
    for split in ("train", "val", "test", "no_split"):
        rows = meta.index[meta["split"].eq(split)].to_numpy()
        waveforms = np.zeros((len(rows), 1, 2500, 12), dtype=np.float32)
        tabular = np.zeros((len(rows), 7), dtype=np.float32)
        for local, original in enumerate(rows):
            waveforms[local, 0] = (
                original * .01 + np.arange(12)[None, :] * .001
                + np.sin(np.arange(2500)[:, None] / 20) * .1
            )
            tabular[local] = original * .01 + np.arange(7) * .001
        np.save(root / f"EchoNext_{split}_waveforms.npy", waveforms)
        np.save(root / f"EchoNext_{split}_tabular_features.npy", tabular)
    return root
