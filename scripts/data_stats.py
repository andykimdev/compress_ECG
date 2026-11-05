"""
Summarize the EchoNext dataset with aggregate tables.

Purpose:
- Check that the waveform and tabular arrays line up with the metadata labels.
- Report label prevalence, subgroup sizes, split shifts, patient leakage, and input quality.

Special Notes:
- Run from the repository root: python3 scripts/data_stats.py
- Reads data/echonext/1.1.1 unless ECHONEXT_DIR points elsewhere.
- Writes only aggregate counts to results/data_stats, never per-patient rows.
- scripts/data_figures.py draws the figures from these tables.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "data_stats"
SPLITS = ["train", "val", "test", "no_split"]
COMPOSITE = "shd_moderate_or_greater_flag"
COMPONENTS = [
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
]
# Tabular features in the metadata. The .npy arrays hold the same 7 standardized
# features but in a different order, which the alignment tables recover.
TABULAR = [
    "sex",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
]
# Column order of the tabular arrays, recovered from the alignment tables.
ARRAY_FEATURES = [
    "sex",
    "age_at_ecg",
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
]
# The heart ultrasound values behind each label. Wall thickness uses two measurements.
LABEL_VALUES = {
    "lvef_lte_45_flag": ["lvef_value"],
    "lvwt_gte_13_flag": ["ivs_measurement", "lvpw_measurement"],
    "aortic_stenosis_moderate_or_greater_flag": ["aortic_stenosis_value"],
    "aortic_regurgitation_moderate_or_greater_flag": ["aortic_regurgitation_value"],
    "mitral_regurgitation_moderate_or_greater_flag": ["mitral_regurgitation_value"],
    "tricuspid_regurgitation_moderate_or_greater_flag": ["tricuspid_regurgitation_value"],
    "pulmonary_regurgitation_moderate_or_greater_flag": ["pulmonary_regurgitation_value"],
    "rv_systolic_dysfunction_moderate_or_greater_flag": ["rv_systolic_function_value"],
    "pericardial_effusion_moderate_large_flag": ["pericardial_effusion_value"],
    "pasp_gte_45_flag": ["pasp_value"],
    "tr_max_gte_32_flag": ["tr_max_velocity_value"],
}
# Our own judgment of values too extreme to trust, not cutoffs from the dataset.
UNUSUAL_LOW = {"ventricular_rate": 25, "pr_interval": 60, "qrs_duration": 40, "qt_corrected": 250}
UNUSUAL_HIGH = {"ventricular_rate": 220}
SUBGROUPS = ["race_ethnicity", "sex", "age_band", "location_setting"]
AGE_BINS = [18, 45, 65, 80, 91]
AGE_LABELS = ["18-44", "45-64", "65-79", "80-90"]


def resolve_data_dir() -> Path:
    """
    Find the folder that holds the EchoNext files.

    Returns:
        ECHONEXT_DIR when it is set, otherwise data/echonext/1.1.1 in the repository.

    Raises:
        FileNotFoundError: When the folder has no metadata CSV.
    """
    data_dir = Path(os.environ.get("ECHONEXT_DIR", REPO / "data" / "echonext" / "1.1.1"))
    if not (data_dir / "echonext_metadata_100k.csv").is_file():
        raise FileNotFoundError(f"No echonext_metadata_100k.csv in {data_dir}. Set ECHONEXT_DIR.")
    return data_dir


def add_age_band(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the metadata with an age_band column.

    Args:
        meta: Metadata with age_at_ecg in years, 18 to 90.

    Returns:
        The metadata plus age_band, one of AGE_LABELS.
    """
    out = meta.copy()
    out["age_band"] = pd.cut(out["age_at_ecg"], AGE_BINS, right=False, labels=AGE_LABELS)
    return out


def label_prevalence(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count positives and prevalence for every label in every split.

    Args:
        meta: Metadata with a split column and 0 or 1 label flags.

    Returns:
        One row per label and split, with n, positives, and prevalence in percent.
    """
    rows = []
    for label in [COMPOSITE] + COMPONENTS:
        for split in SPLITS:
            flags = meta.loc[meta["split"] == split, label]
            rows.append(
                {
                    "label": label,
                    "split": split,
                    "n": len(flags),
                    "positives": int(flags.sum()),
                    "prevalence_pct": round(100 * flags.mean(), 1),
                }
            )
    return pd.DataFrame(rows)


def component_overlap(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many component labels each ECG has, split by the composite label.

    Args:
        meta: Metadata with the composite and component flags.

    Returns:
        One row per composite value and component count (0, 1, 2, 3 or more), with n.
    """
    count = meta[COMPONENTS].sum(axis=1).clip(upper=3)
    table = (
        pd.DataFrame({"composite": meta[COMPOSITE], "components": count})
        .value_counts()
        .rename("n")
        .reset_index()
        .sort_values(["composite", "components"])
    )
    table["components"] = table["components"].map({0: "0", 1: "1", 2: "2", 3: "3+"})
    return table


def subgroup_table(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count ECGs and composite positives in every subgroup of every split.

    Args:
        meta: Metadata with age_band added and the SUBGROUPS columns.

    Returns:
        One row per split, subgroup column, and level, with n, positives, and prevalence.
    """
    rows = []
    for split in SPLITS:
        part = meta[meta["split"] == split]
        for column in SUBGROUPS:
            grouped = part.groupby(column, observed=True)[COMPOSITE].agg(["size", "sum", "mean"])
            for level, stats in grouped.iterrows():
                rows.append(
                    {
                        "split": split,
                        "subgroup": column,
                        "level": level,
                        "n": int(stats["size"]),
                        "positives": int(stats["sum"]),
                        "prevalence_pct": round(100 * stats["mean"], 1),
                    }
                )
    return pd.DataFrame(rows)


def split_shift(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize acquisition year, age, and care setting mix for each split.

    Args:
        meta: Metadata with split, acquisition_year, age_at_ecg, and location_setting.

    Returns:
        One row per split with year range and median, median age, and setting shares in percent.
    """
    rows = []
    for split in SPLITS:
        part = meta[meta["split"] == split]
        row = {
            "split": split,
            "year_min": int(part["acquisition_year"].min()),
            "year_median": float(part["acquisition_year"].median()),
            "year_max": int(part["acquisition_year"].max()),
            "age_median": float(part["age_at_ecg"].median()),
            "most_recent_ecg_pct": round(100 * part["most_recent_ecg"].mean(), 1),
        }
        shares = part["location_setting"].value_counts(normalize=True)
        for setting, share in shares.items():
            row[f"{setting}_pct"] = round(100 * share, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def patient_leakage(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count patients shared between each pair of splits and ECGs per patient within each split.

    Args:
        meta: Metadata with split and patient_key.

    Returns:
        One row per split pair with the shared patient count, then one row per split
        with patients, ECGs, and the most ECGs any one patient has.
    """
    patients = {split: set(meta.loc[meta["split"] == split, "patient_key"]) for split in SPLITS}
    rows = []
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1 :]:
            rows.append(
                {
                    "check": "shared_patients",
                    "splits": f"{first}+{second}",
                    "value": len(patients[first] & patients[second]),
                }
            )
    for split in SPLITS:
        per_patient = meta.loc[meta["split"] == split, "patient_key"].value_counts()
        rows.append({"check": "patients", "splits": split, "value": len(per_patient)})
        rows.append({"check": "ecgs", "splits": split, "value": int(per_patient.sum())})
        rows.append({"check": "max_ecgs_per_patient", "splits": split, "value": int(per_patient.max())})
    return pd.DataFrame(rows)


def tabular_alignment(meta_split: pd.DataFrame, array: np.ndarray) -> pd.DataFrame:
    """
    Rank-correlate each tabular array column with each metadata feature for one split.

    Args:
        meta_split: Metadata rows of one split, in file order.
        array: Tabular features of the same split, shape (N, 7).

    Returns:
        A 7 by 7 Spearman correlation table, array columns as rows and metadata
        features as columns. Aligned rows give about 1.0 on the diagonal.

    Raises:
        ValueError: When the row counts differ.
    """
    if len(meta_split) != len(array):
        raise ValueError(f"Metadata has {len(meta_split)} rows but the array has {len(array)}.")
    features = meta_split[TABULAR].copy()
    features["sex"] = (features["sex"] == "male").astype(float)
    columns = pd.DataFrame(array, columns=[f"array_{i}" for i in range(array.shape[1])])
    combined = pd.concat([columns, features.reset_index(drop=True)], axis=1)
    corr = combined.corr(method="spearman")
    return corr.loc[columns.columns, TABULAR].round(3)


def tabular_quality(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing values and the value range of each tabular feature.

    Args:
        meta: Metadata with the TABULAR columns.

    Returns:
        One row per numeric feature with the NaN count and the min, 1st, 50th, 99th
        percentiles, and max.
    """
    rows = []
    for column in TABULAR[1:]:
        values = meta[column]
        quantiles = values.quantile([0.01, 0.5, 0.99])
        rows.append(
            {
                "feature": column,
                "nan": int(values.isna().sum()),
                "min": values.min(),
                "p1": quantiles[0.01],
                "p50": quantiles[0.5],
                "p99": quantiles[0.99],
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def waveform_stats(waveforms: np.ndarray, chunk: int = 1000) -> pd.DataFrame:
    """
    Compute per-lead summary statistics over every sample of every ECG.

    Args:
        waveforms: Array of shape (N, 1, 2500, 12), memory-mapped or in memory.
        chunk: ECGs read per step, to bound memory use.

    Returns:
        One row per lead with mean, std, min, max, NaN count, and the number of ECGs
        whose lead is flat (std under 1e-6).
    """
    n_leads = waveforms.shape[-1]
    total = np.zeros(n_leads)
    total_sq = np.zeros(n_leads)
    count = np.zeros(n_leads)
    nan = np.zeros(n_leads, dtype=np.int64)
    flat = np.zeros(n_leads, dtype=np.int64)
    low = np.full(n_leads, np.inf)
    high = np.full(n_leads, -np.inf)
    for start in range(0, len(waveforms), chunk):
        x = np.asarray(waveforms[start : start + chunk, 0], dtype=np.float64)
        missing = np.isnan(x)
        nan += missing.sum(axis=(0, 1))
        count += (~missing).sum(axis=(0, 1))
        total += np.nansum(x, axis=(0, 1))
        total_sq += np.nansum(x**2, axis=(0, 1))
        low = np.minimum(low, np.nanmin(x, axis=(0, 1)))
        high = np.maximum(high, np.nanmax(x, axis=(0, 1)))
        flat += (np.nanstd(x, axis=1) < 1e-6).sum(axis=0)
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return pd.DataFrame(
        {
            "lead": np.arange(n_leads),
            "mean": mean.round(4),
            "std": std.round(4),
            "min": low.round(3),
            "max": high.round(3),
            "nan": nan,
            "flat_ecgs": flat,
        }
    )


def age_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count ECGs at each age in each split.

    Args:
        meta: Metadata with split and age_at_ecg in whole years.

    Returns:
        One row per age, one column of ECG counts per split.
    """
    table = pd.crosstab(meta["age_at_ecg"], meta["split"])[SPLITS]
    return table.reset_index().rename_axis(columns=None)


def age_sex_table(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count ECGs in every age band and sex combination of every split.

    Args:
        meta: Metadata with split, age_band, sex, and most_recent_ecg.

    Returns:
        One row per split, age band, and sex, with counts and within-split shares in
        percent. The latest_ecgs columns count each patient once, which makes Training
        comparable to Validation and Test. Unassigned has no latest ECGs, so its share is NaN.
    """
    table = (
        meta.groupby(["split", "age_band", "sex"], observed=True)
        .agg(ecgs=("sex", "size"), latest_ecgs=("most_recent_ecg", "sum"))
        .reset_index()
    )
    for column in ["ecgs", "latest_ecgs"]:
        totals = table.groupby("split")[column].transform("sum")
        table[f"{column}_pct"] = (100 * table[column] / totals).round(1)
    return table.sort_values("split", key=lambda s: s.map(SPLITS.index), kind="stable")


def count_repeated_waveforms(waveforms: np.ndarray) -> int:
    """
    Count ECGs whose full waveform exactly matches an earlier ECG in the same array.

    Args:
        waveforms: Array of shape (N, 1, 2500, 12), memory-mapped or in memory.

    Returns:
        The number of repeated waveforms.
    """
    seen = {hashlib.md5(np.asarray(ecg).tobytes()).digest() for ecg in waveforms}
    return len(waveforms) - len(seen)


def count_unusual_measurements(meta: pd.DataFrame) -> int:
    """
    Count ECGs with at least one measurement outside the UNUSUAL cutoffs.

    Args:
        meta: Metadata with the raw ECG measurements. NaN does not count as unusual.

    Returns:
        The number of ECGs with an unusual measurement.
    """
    unusual = pd.Series(False, index=meta.index)
    for column, cutoff in UNUSUAL_LOW.items():
        unusual |= meta[column] < cutoff
    for column, cutoff in UNUSUAL_HIGH.items():
        unusual |= meta[column] > cutoff
    return int(unusual.sum())


def clean_checks(
    meta_split: pd.DataFrame, array: np.ndarray, leads: pd.DataFrame, repeats: int
) -> pd.DataFrame:
    """
    Count basic data problems in one split.

    Args:
        meta_split: Metadata rows of one split.
        array: Tabular features of the same split, shape (N, 7).
        leads: Output of waveform_stats for the same split.
        repeats: Output of count_repeated_waveforms for the same split.

    Returns:
        One row per check with its count.
    """
    counts = {
        "ecgs": len(meta_split),
        "missing_waveform_values": int(leads["nan"].sum()),
        "missing_feature_values": int(np.isnan(array).sum()),
        "repeated_ecg_ids": int(meta_split["ecg_key"].duplicated().sum()),
        "repeated_waveforms": repeats,
        "flat_leads": int(leads["flat_ecgs"].sum()),
        "unusual_measurements": count_unusual_measurements(meta_split),
    }
    return pd.DataFrame({"check": list(counts), "value": list(counts.values())})


def feature_averages(array: np.ndarray) -> pd.DataFrame:
    """
    Report the average and spread of each stored feature in one split.

    Args:
        array: Tabular features of one split, shape (N, 7).

    Returns:
        One row per feature with its mean and standard deviation.
    """
    return pd.DataFrame(
        {
            "feature": ARRAY_FEATURES,
            "mean": array.mean(axis=0).round(3),
            "std": array.std(axis=0).round(3),
        }
    )


def missing_features(meta_split: pd.DataFrame, array: np.ndarray) -> pd.DataFrame:
    """
    Count missing raw measurements and show what the stored array holds in their place.

    Args:
        meta_split: Metadata rows of one split, in file order.
        array: Tabular features of the same split, shape (N, 7).

    Returns:
        One row per numeric feature with the missing count, percent, and the lowest and
        highest stored value among those rows. Equal values mean one fixed replacement.
    """
    rows = []
    for index, column in enumerate(ARRAY_FEATURES):
        if column == "sex":
            continue
        missing = meta_split[column].isna().to_numpy()
        filled = array[missing, index]
        rows.append(
            {
                "feature": column,
                "missing": int(missing.sum()),
                "missing_pct": round(100 * missing.mean(), 2),
                "stored_min": round(filled.min(), 3) if missing.any() else np.nan,
                "stored_max": round(filled.max(), 3) if missing.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def label_sources(meta_split: pd.DataFrame) -> pd.DataFrame:
    """
    Split each finding's labels by whether a heart ultrasound value stands behind them.

    Args:
        meta_split: Metadata rows of one split.

    Returns:
        One row per finding with counts of yes and of no, each with and without a value.
    """
    rows = []
    for label, columns in LABEL_VALUES.items():
        measured = meta_split[columns].notna().any(axis=1)
        positive = meta_split[label] == 1
        rows.append(
            {
                "label": label,
                "yes_measured": int((positive & measured).sum()),
                "yes_not_measured": int((positive & ~measured).sum()),
                "no_measured": int((~positive & measured).sum()),
                "no_not_measured": int((~positive & ~measured).sum()),
            }
        )
    return pd.DataFrame(rows)


def no_ultrasound_values(meta_split: pd.DataFrame) -> dict[str, float]:
    """
    Count ECGs with no heart ultrasound value at all, and how many are labeled as disease.

    Args:
        meta_split: Metadata rows of one split.

    Returns:
        The count, its percent of the split, and the count labeled with heart disease.
    """
    columns = [column for names in LABEL_VALUES.values() for column in names]
    empty = meta_split[columns].isna().all(axis=1)
    return {
        "ecgs": int(empty.sum()),
        "pct": round(100 * empty.mean(), 1),
        "with_disease": int(meta_split.loc[empty, COMPOSITE].sum()),
    }


def save(table: pd.DataFrame, name: str) -> None:
    """Write a table to the output folder and print it."""
    table.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"\n== {name} ==")
    print(table.to_string(index=False))


def main() -> None:
    data_dir = resolve_data_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = add_age_band(pd.read_csv(data_dir / "echonext_metadata_100k.csv", index_col=0))
    print(f"Metadata: {len(meta)} rows from {data_dir}")

    arrays = {}
    for split in SPLITS:
        arrays[split] = np.load(data_dir / f"EchoNext_{split}_tabular_features.npy")
        alignment = tabular_alignment(meta[meta["split"] == split], arrays[split])
        save(alignment.reset_index(names="array_column"), f"alignment_{split}")
        print(f"array NaN count: {int(np.isnan(arrays[split]).sum())}")

    save(label_prevalence(meta), "label_prevalence")
    save(component_overlap(meta), "component_overlap")
    subgroups = subgroup_table(meta)
    save(subgroups, "subgroups")
    save(split_shift(meta), "split_shift")
    save(patient_leakage(meta), "patient_leakage")
    save(tabular_quality(meta), "tabular_quality")

    lead_tables = []
    check_tables = []
    for split in SPLITS:
        part = meta[meta["split"] == split]
        waveforms = np.load(data_dir / f"EchoNext_{split}_waveforms.npy", mmap_mode="r")
        leads = waveform_stats(waveforms)
        checks = clean_checks(part, arrays[split], leads, count_repeated_waveforms(waveforms))
        check_tables.append(checks.assign(split=split))
        leads.insert(0, "split", split)
        lead_tables.append(leads)
    leads = pd.concat(lead_tables)
    save(leads, "waveform_leads")
    save(pd.concat(check_tables), "clean_checks")

    parts = {split: meta[meta["split"] == split] for split in SPLITS}
    save(pd.concat(feature_averages(arrays[s]).assign(split=s) for s in SPLITS), "feature_averages")
    save(
        pd.concat(missing_features(parts[s], arrays[s]).assign(split=s) for s in SPLITS),
        "missing_features",
    )
    save(pd.concat(label_sources(parts[s]).assign(split=s) for s in SPLITS), "label_sources")
    save(
        pd.DataFrame([{"split": s, **no_ultrasound_values(parts[s])} for s in SPLITS]),
        "no_ultrasound_values",
    )

    save(age_counts(meta), "age_counts")
    save(age_sex_table(meta), "age_sex")
    print(f"\nWrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
