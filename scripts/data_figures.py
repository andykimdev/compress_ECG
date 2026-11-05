"""
Draw the data check figures from the aggregate tables in results/data_stats.

Purpose:
- Give every figure the same names, colors, fonts, and size, so they read as one set.
- Work without the patient data, since it reads only the saved aggregate tables.

Special Notes:
- Run from the repository root after scripts/data_stats.py: python3 scripts/data_figures.py
- Writes one PNG per figure to results/data_stats/figures, sized for a 16:9 slide.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "data_stats"
OUT_DIR = TABLES / "figures"

SPLITS = ["train", "val", "test", "no_split"]
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test", "no_split": "Unassigned"}
# Each group keeps one color in every figure. The two blues differ in lightness, not hue alone.
SPLIT_COLORS = {"train": "#1F4E79", "val": "#6FA8DC", "test": "#D9711E", "no_split": "#9AA0A6"}
WITH_DISEASE = "#D9711E"
WITHOUT_DISEASE = "#F4D3B5"
NEUTRAL = "#4B5563"
INK = "#1F2933"
PAPER = "#FAFAF7"
GRID = "#E3E3DE"

COMPOSITE = "shd_moderate_or_greater_flag"
LABEL_NAMES = {
    "lvef_lte_45_flag": "Weak pumping, left side",
    "lvwt_gte_13_flag": "Thick heart wall",
    "aortic_stenosis_moderate_or_greater_flag": "Narrow aortic valve",
    "aortic_regurgitation_moderate_or_greater_flag": "Leaky aortic valve",
    "mitral_regurgitation_moderate_or_greater_flag": "Leaky mitral valve",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Leaky tricuspid valve",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Leaky pulmonary valve",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "Weak pumping, right side",
    "pericardial_effusion_moderate_large_flag": "Fluid around the heart",
    "pasp_gte_45_flag": "High lung pressure (estimate)",
    "tr_max_gte_32_flag": "High lung pressure (jet speed)",
}
CHECK_NAMES = {
    "missing_waveform_values": "Missing waveform values",
    "missing_feature_values": "Missing feature values",
    "repeated_ecg_ids": "Repeated ECG IDs",
    "repeated_waveforms": "Repeated waveforms",
    "flat_leads": "Flat leads",
    "unusual_measurements": "ECGs with an unusual measurement",
}
FEATURE_NAMES = {
    "age_at_ecg": "Age",
    "ventricular_rate": "Heart rate",
    "atrial_rate": "Atrial rate",
    "pr_interval": "PR interval",
    "qrs_duration": "QRS duration",
    "qt_corrected": "QT interval",
}
SUBGROUP_NAMES = {
    "race_ethnicity": "Race and ethnicity",
    "sex": "Sex",
    "age_band": "Age",
    "location_setting": "Care setting",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.facecolor": PAPER,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "legend.frameon": False,
    }
)


def new_figure(ncols: int = 1) -> tuple[plt.Figure, list[plt.Axes]]:
    """Make a slide-sized figure with one or more side-by-side panels."""
    fig, axes = plt.subplots(1, ncols, figsize=(16, 6.4), squeeze=False)
    return fig, list(axes[0])


def comma_axis(axis: matplotlib.axis.Axis) -> None:
    """Show whole-number ticks with thousands separators."""
    axis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))


def tilt_labels(ax: plt.Axes) -> None:
    """Rotate x labels so each one ends under its bar."""
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor", fontsize=16)


def label_bars(ax: plt.Axes, bars, texts: list[str]) -> None:
    """Write one label just above each bar."""
    for bar, text in zip(bars, texts):
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to the output folder and close it."""
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=100)
    plt.close(fig)
    print(f"wrote {name}.png")


def plot_group_sizes(leakage: pd.DataFrame) -> None:
    """Draw ECG counts per group, with the patient count in each label."""
    count = leakage.set_index(["check", "splits"])["value"]
    ecgs = [count[("ecgs", s)] for s in SPLITS]
    patients = [count[("patients", s)] for s in SPLITS]
    fig, (ax,) = new_figure()
    bars = ax.bar([SPLIT_NAMES[s] for s in SPLITS], ecgs, color=[SPLIT_COLORS[s] for s in SPLITS])
    label_bars(ax, bars, [f"{e:,} ECGs\n{p:,} patients" for e, p in zip(ecgs, patients)])
    ax.set_ylabel("Number of ECGs")
    ax.set_ylim(0, max(ecgs) * 1.25)
    comma_axis(ax.yaxis)
    save(fig, "01_group_sizes")


def plot_disease_rate_by_group(prevalence: pd.DataFrame) -> None:
    """Draw the share of ECGs with heart disease in each group."""
    rows = prevalence[prevalence["label"] == COMPOSITE].set_index("split").loc[SPLITS]
    fig, (ax,) = new_figure()
    bars = ax.bar(
        [SPLIT_NAMES[s] for s in SPLITS],
        rows["prevalence_pct"],
        color=[SPLIT_COLORS[s] for s in SPLITS],
    )
    label_bars(ax, bars, [f"{p:.0f}%" for p in rows["prevalence_pct"]])
    ax.set_ylabel("ECGs with heart disease (%)")
    ax.set_ylim(0, 70)
    save(fig, "02_disease_rate_by_group")


def plot_shared_patients(leakage: pd.DataFrame) -> None:
    """Draw how many patients each pair of groups has in common."""
    pairs = leakage[leakage["check"] == "shared_patients"]
    names = [
        " and ".join(SPLIT_NAMES[s] for s in pair.split("+")) for pair in pairs["splits"]
    ]
    fig, (ax,) = new_figure()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    bars = ax.barh(names, pairs["value"], color=NEUTRAL)
    for bar, value in zip(bars, pairs["value"]):
        ax.annotate(
            f"{value:,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    ax.invert_yaxis()
    ax.set_xlabel("Patients in both groups")
    comma_axis(ax.xaxis)
    ax.set_xlim(0, pairs["value"].max() * 1.15)
    save(fig, "03_shared_patients")


def plot_signal_scale(leads: pd.DataFrame) -> None:
    """Draw the typical spread of the stored ECG signal in each group."""
    spread = leads.groupby("split")["std"].mean().loc[SPLITS]
    fig, (ax,) = new_figure()
    bars = ax.bar([SPLIT_NAMES[s] for s in SPLITS], spread, color=[SPLIT_COLORS[s] for s in SPLITS])
    label_bars(ax, bars, [f"{v:.2f}" for v in spread])
    ax.set_ylabel("Typical signal spread")
    ax.set_ylim(0, 1.15)
    save(fig, "04_signal_scale")


def plot_disease_types(prevalence: pd.DataFrame) -> None:
    """Draw the number of test ECGs with each type of heart disease, fewest first."""
    rows = prevalence[(prevalence["split"] == "test") & (prevalence["label"].isin(LABEL_NAMES))]
    rows = rows.sort_values("positives")
    fig, (ax,) = new_figure()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    bars = ax.barh([LABEL_NAMES[l] for l in rows["label"]], rows["positives"], color=WITH_DISEASE)
    for bar, value in zip(bars, rows["positives"]):
        ax.annotate(
            f"{value:,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=16,
        )
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Test ECGs with this type")
    comma_axis(ax.xaxis)
    ax.set_xlim(0, rows["positives"].max() * 1.12)
    save(fig, "05_disease_types_test")


def subgroup_panels(subgroups: pd.DataFrame):
    """Yield each test-set patient group column with its rows, in a fixed order."""
    test = subgroups[subgroups["split"] == "test"]
    for column, title in SUBGROUP_NAMES.items():
        part = test[test["subgroup"] == column]
        levels = [str(level).capitalize().replace("Procedural", "Procedure").replace("80-90", "80+") for level in part["level"]]
        yield title, levels, part


def plot_subgroup_sizes(subgroups: pd.DataFrame) -> None:
    """Draw test ECG counts per patient group, split into with and without heart disease."""
    fig, axes = new_figure(ncols=4)
    for ax, (title, levels, part) in zip(axes, subgroup_panels(subgroups)):
        without = part["n"] - part["positives"]
        ax.bar(levels, part["positives"], color=WITH_DISEASE, label="With heart disease")
        bars = ax.bar(levels, without, bottom=part["positives"], color=WITHOUT_DISEASE, label="Without heart disease")
        for bar, n in zip(bars, part["n"]):
            ax.annotate(
                f"{n:,}",
                (bar.get_x() + bar.get_width() / 2, n),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=15,
            )
        ax.set_title(title)
        ax.set_ylim(0, 3000)
        tilt_labels(ax)
    axes[0].set_ylabel("Test ECGs")
    comma_axis(axes[0].yaxis)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "06_patient_group_sizes_test.png", dpi=100)
    plt.close(fig)
    print("wrote 06_patient_group_sizes_test.png")


def plot_subgroup_rates(subgroups: pd.DataFrame) -> None:
    """Draw the share of test ECGs with heart disease in each patient group."""
    fig, axes = new_figure(ncols=4)
    for ax, (title, levels, part) in zip(axes, subgroup_panels(subgroups)):
        bars = ax.bar(levels, part["prevalence_pct"], color=WITH_DISEASE)
        label_bars(ax, bars, [f"{p:.0f}%" for p in part["prevalence_pct"]])
        for text in ax.texts:
            text.set_fontsize(15)
        ax.set_title(title)
        ax.set_ylim(0, 75)
        tilt_labels(ax)
    axes[0].set_ylabel("Test ECGs with heart disease (%)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    save(fig, "07_disease_rate_by_patient_group_test")


def plot_ages(ages: pd.DataFrame) -> None:
    """Draw the age distribution of each group as the share of its ECGs in 5-year bands."""
    edges = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 91]
    names = ["18-24"] + [f"{lo}-{lo + 4}" for lo in edges[1:-2]] + ["90+"]
    bands = pd.cut(ages["age_at_ecg"], edges, right=False, labels=names)
    counts = ages[SPLITS].groupby(bands, observed=True).sum()
    fig, (ax,) = new_figure()
    for split in SPLITS:
        share = 100 * counts[split] / counts[split].sum()
        ax.plot(names, share, color=SPLIT_COLORS[split], linewidth=3, marker="o", label=SPLIT_NAMES[split])
    ax.set_xlabel("Age at ECG (years)")
    ax.set_ylabel("Share of the group's ECGs (%)")
    ax.set_ylim(0, None)
    tilt_labels(ax)
    ax.legend(loc="upper left")
    save(fig, "08_ages_by_group")


def plot_age_and_sex(table: pd.DataFrame) -> None:
    """Draw each age and sex combination as a share of its group's ECGs."""
    width = 0.27
    fig, axes = new_figure(ncols=2)
    for ax, sex in zip(axes, ["female", "male"]):
        part = table[table["sex"] == sex]
        for offset, split in enumerate(SPLITS[:3]):
            rows = part[part["split"] == split]
            positions = [i + (offset - 1) * width for i in range(len(rows))]
            bars = ax.bar(positions, rows["ecgs_pct"], width, color=SPLIT_COLORS[split], label=SPLIT_NAMES[split])
            label_bars(ax, bars, [f"{p:.0f}%" for p in rows["ecgs_pct"]])
        for text in ax.texts:
            text.set_fontsize(15)
        ax.set_xticks(range(len(rows)), [band.replace("80-90", "80+") for band in rows["age_band"]])
        ax.set_title(sex.capitalize())
        ax.set_xlabel("Age at ECG (years)")
        ax.set_ylim(0, table["ecgs_pct"].max() * 1.2)
    axes[0].set_ylabel("Share of the group's ECGs (%)")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="upper left")
    save(fig, "09_age_and_sex_by_group")


def grouped_bars(ax: plt.Axes, table: pd.DataFrame, value: str, names: dict[str, str], text: str) -> None:
    """Draw one bar per used group for each named feature, with a label above each bar."""
    width = 0.27
    for offset, split in enumerate(SPLITS[:3]):
        rows = table[table["split"] == split].set_index("feature").loc[list(names)]
        positions = [i + (offset - 1) * width for i in range(len(names))]
        bars = ax.bar(positions, rows[value], width, color=SPLIT_COLORS[split], label=SPLIT_NAMES[split])
        for bar, number in zip(bars, rows[value]):
            ax.annotate(
                # Adding 0.0 turns a rounded -0.0 into 0.0.
                text.format(round(number, 2) + 0.0),
                (bar.get_x() + bar.get_width() / 2, number),
                xytext=(0, 5 if number >= 0 else -5),
                textcoords="offset points",
                ha="center",
                va="bottom" if number >= 0 else "top",
                fontsize=13,
            )
    ax.set_xticks(range(len(names)), list(names.values()))


def plot_data_problems(checks: pd.DataFrame) -> None:
    """Draw how many of each basic data problem the used groups contain."""
    totals = checks[checks["split"].isin(SPLITS[:3])].groupby("check")["value"].sum()
    fig, (ax,) = new_figure()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    bars = ax.barh(list(CHECK_NAMES.values()), totals.loc[list(CHECK_NAMES)], color=NEUTRAL)
    for bar, value in zip(bars, totals.loc[list(CHECK_NAMES)]):
        ax.annotate(
            f"{value:,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    ax.invert_yaxis()
    ax.set_title(f"Data problems found in {totals['ecgs']:,} ECGs (Training, Validation, Test)")
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(totals.loc[list(CHECK_NAMES)].max() * 1.15, 1))
    save(fig, "10_data_problems")


def plot_waveform_range(leads: pd.DataFrame) -> None:
    """Draw the lowest and highest stored waveform value in each group."""
    low = leads.groupby("split")["min"].min().loc[SPLITS]
    high = leads.groupby("split")["max"].max().loc[SPLITS]
    fig, (ax,) = new_figure()
    ax.bar(
        [SPLIT_NAMES[s] for s in SPLITS],
        high - low,
        bottom=low,
        color=[SPLIT_COLORS[s] for s in SPLITS],
    )
    for index, split in enumerate(SPLITS):
        ax.annotate(f"{high[split]:.1f}", (index, high[split]), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.annotate(f"{low[split]:.1f}", (index, low[split]), xytext=(0, -5), textcoords="offset points", ha="center", va="top")
    ax.set_title("Lowest to highest waveform value in each group")
    ax.set_ylabel("Stored waveform value")
    ax.set_ylim(-13, 13)
    save(fig, "11_waveform_range")


def plot_feature_averages(averages: pd.DataFrame) -> None:
    """Draw each rescaled feature's average, by group, on a fixed scale."""
    fig, (ax,) = new_figure()
    grouped_bars(ax, averages, "mean", FEATURE_NAMES, "{:.2f}")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_title("Average of each feature compared with Training")
    ax.set_ylabel("Difference from Training\n(standard deviations)")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc="upper right", ncols=3)
    save(fig, "12_feature_averages")


def plot_missing_measurements(missing: pd.DataFrame) -> None:
    """Draw how often each raw ECG measurement is missing, by group."""
    names = {key: name for key, name in FEATURE_NAMES.items() if key != "age_at_ecg"}
    fig, (ax,) = new_figure()
    grouped_bars(ax, missing, "missing_pct", names, "{:.1f}%")
    ax.set_title("How often each ECG measurement is missing")
    ax.set_ylabel("ECGs missing this value (%)")
    ax.set_ylim(0, 15)
    ax.legend(loc="upper right")
    save(fig, "13_missing_measurements")


def plot_label_sources(sources: pd.DataFrame) -> None:
    """Draw how many labels rest on a heart ultrasound value and how many default to no."""
    used = sources[sources["split"].isin(SPLITS[:3])].groupby("label").sum(numeric_only=True)
    shares = 100 * used.div(used.sum(axis=1), axis=0)
    shares = shares.sort_values("no_not_measured")
    names = [LABEL_NAMES[label] for label in shares.index]
    yes = shares["yes_measured"] + shares["yes_not_measured"]
    fig, (ax,) = new_figure()
    ax.grid(axis="y", visible=False)
    ax.barh(names, yes, color=WITH_DISEASE, label="Yes")
    ax.barh(names, shares["no_measured"], left=yes, color=WITHOUT_DISEASE, label="No, measured")
    ax.barh(names, shares["no_not_measured"], left=yes + shares["no_measured"], color=SPLIT_COLORS["no_split"], label="No, not measured (share at right)")
    for name, share in zip(names, shares["no_not_measured"]):
        ax.annotate(f"{share:.0f}%", (100, name), xytext=(6, 0), textcoords="offset points", va="center", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlabel("Share of ECGs (%)")
    ax.set_xlim(0, 106)
    ax.set_title("Where each finding's yes or no label comes from", pad=44)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    save(fig, "14_label_sources")


def plot_no_ultrasound(empty: pd.DataFrame) -> None:
    """Draw the share of each used group's ECGs with no heart ultrasound value."""
    rows = empty.set_index("split").loc[SPLITS[:3]]
    fig, (ax,) = new_figure()
    bars = ax.bar(
        [SPLIT_NAMES[s] for s in rows.index],
        rows["pct"],
        color=[SPLIT_COLORS[s] for s in rows.index],
    )
    label_bars(ax, bars, [f"{p:.1f}%\n{e:,} ECGs" for p, e in zip(rows["pct"], rows["ecgs"])])
    ax.set_title("ECGs with no heart ultrasound values")
    ax.set_ylabel("Share of the group's ECGs (%)")
    ax.set_ylim(0, 16)
    save(fig, "15_no_ultrasound_values")


def plot_used_group_sizes(leakage: pd.DataFrame) -> None:
    """Draw ECG counts for the groups we use, with the patient count in each label."""
    count = leakage.set_index(["check", "splits"])["value"]
    used = SPLITS[:3]
    ecgs = [count[("ecgs", s)] for s in used]
    patients = [count[("patients", s)] for s in used]
    fig, (ax,) = new_figure()
    bars = ax.bar([SPLIT_NAMES[s] for s in used], ecgs, color=[SPLIT_COLORS[s] for s in used])
    label_bars(ax, bars, [f"{e:,} ECGs\n{p:,} patients" for e, p in zip(ecgs, patients)])
    ax.set_title("Training, Validation, and Test sizes")
    ax.set_ylabel("Number of ECGs")
    ax.set_ylim(0, max(ecgs) * 1.25)
    comma_axis(ax.yaxis)
    save(fig, "16_train_val_test_sizes")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    leakage = pd.read_csv(TABLES / "patient_leakage.csv")
    prevalence = pd.read_csv(TABLES / "label_prevalence.csv")
    subgroups = pd.read_csv(TABLES / "subgroups.csv")
    plot_group_sizes(leakage)
    plot_disease_rate_by_group(prevalence)
    plot_shared_patients(leakage)
    plot_signal_scale(pd.read_csv(TABLES / "waveform_leads.csv"))
    plot_disease_types(prevalence)
    plot_subgroup_sizes(subgroups)
    plot_subgroup_rates(subgroups)
    plot_ages(pd.read_csv(TABLES / "age_counts.csv"))
    plot_age_and_sex(pd.read_csv(TABLES / "age_sex.csv"))
    plot_data_problems(pd.read_csv(TABLES / "clean_checks.csv"))
    plot_waveform_range(pd.read_csv(TABLES / "waveform_leads.csv"))
    plot_feature_averages(pd.read_csv(TABLES / "feature_averages.csv"))
    plot_missing_measurements(pd.read_csv(TABLES / "missing_features.csv"))
    plot_label_sources(pd.read_csv(TABLES / "label_sources.csv"))
    plot_no_ultrasound(pd.read_csv(TABLES / "no_ultrasound_values.csv"))
    plot_used_group_sizes(leakage)


if __name__ == "__main__":
    main()
