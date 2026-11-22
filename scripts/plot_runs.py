#!/usr/bin/env python3
"""
Draw every Teacher training run from the run table as one dot plot.

Purpose:
- Show each run's best Validation score, so settings can be compared at a glance.
- Mark the released EchoNext-Mini on the same ECGs as a dashed line in each panel.
- Read only results/teacher/runs.csv and probe.csv, which hold aggregate scores and no patient data.

Special Notes:
- Run from the repository root after scripts/collect_runs.py: python3 scripts/plot_runs.py
- Writes results/teacher/figures/runs.png. The baseline appears in the first two groups on purpose.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, PAPER, GRID = "#1F2933", "#FAFAF7", "#E3E3DE"
EXTRAS_GROUP = "Extra inputs and weight averaging"
GROUP_COLORS = {"Learning rates": "#1F4E79", "Overfitting controls": "#D9711E", EXTRAS_GROUP: "#2E7D5B"}
CONTROLS = ["head_dropout", "lead_mask_prob", "balance_patients"]
# The off value of every setting that tells two settings apart. A blank in the table means off.
OFF_VALUES = {
    **{name: 0.0 for name in CONTROLS}, "use_tabular": False, "ema_decay": 0.0, "evals_per_epoch": 1,
    "select_by": "mean",
}
SETTING = ["lr", "head_lr", *OFF_VALUES]
SHD_COLUMN = "shd_moderate_or_greater_flag"
# Room left on each side of the outermost dot, in AUROC points.
AXIS_MARGIN = 0.5

plt.rcParams.update({
    "font.family": "Arial", "font.size": 15, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.facecolor": PAPER, "figure.facecolor": PAPER, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "x", "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
})


def fill_blank_settings(table: pd.DataFrame) -> pd.DataFrame:
    """Return the run table with every blank setting set to its off value, as older runs left them blank."""
    filled = table.copy()
    for name, off in OFF_VALUES.items():
        # A column with blanks holds mixed types, so it is filled value by value.
        filled[name] = [off if pd.isna(value) else value for value in filled[name]]
    return filled


def setting_label(row: pd.Series) -> str:
    """
    Name a setting in plain words.

    Args:
        row: One run of the table from fill_blank_settings.

    Returns:
        The overfitting controls that are on, then the 7 extra numbers and weight averaging
        when on. A run with none of these is named by its learning rates.
    """
    on = {name: row[name] for name in CONTROLS if row[name] > 0}
    names = {"head_dropout": "Dropout", "lead_mask_prob": "Lead masking", "balance_patients": "Patient balancing"}
    parts = ["All three controls"] if len(on) == 3 else [f"{names[name]} {value:g}" for name, value in on.items()]
    parts += ["7 extra numbers"] if row["use_tabular"] else []
    parts += ["weight averaging"] if row["ema_decay"] > 0 else []
    if parts:
        return ", ".join(parts)
    if (row["lr"], row["head_lr"]) == (1e-5, 1e-4):
        return "No control (baseline)"
    return f"ECG-FM rate {row['lr']:g}, output rate {row['head_lr']:g}"


def build_rows(table: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """
    Split the runs into two groups and list the settings in each, best first.

    Args:
        table: The run table with settings and Validation scores.

    Returns:
        One entry per setting as (group, label, its runs). The baseline is in the first two
        groups. Runs with the 7 extra numbers or weight averaging form the third group.

    Raises:
        ValueError: When a run falls in no group, so the figure would leave it out.
    """
    table = fill_blank_settings(table)
    no_control = (table[CONTROLS] == 0).all(axis=1)
    at_baseline_rates = (table["lr"] == 1e-5) & (table["head_lr"] == 1e-4)
    extras = table["use_tabular"].astype(bool) | (table["ema_decay"] > 0)
    groups = {
        "Learning rates": table[no_control & ~extras], "Overfitting controls": table[at_baseline_rates & ~extras],
        EXTRAS_GROUP: table[at_baseline_rates & extras],
    }
    left_out = table[~table.index.isin(pd.concat(groups.values()).index)]
    if not left_out.empty:
        raise ValueError(f"No group draws these runs: {sorted(left_out['run_id'])}")
    rows = []
    for group, part in groups.items():
        settings = [rows_ for _, rows_ in part.groupby(SETTING)]
        settings.sort(key=lambda runs: -runs["val_mean_auroc"].mean())
        rows += [(group, setting_label(runs.iloc[0]), runs) for runs in settings]
    return rows


def axis_limits(values: list[float]) -> tuple[float, float]:
    """
    Return axis limits that hold every value with room on each side.

    Args:
        values: The scores to show, in AUROC points.

    Returns:
        The lowest value minus AXIS_MARGIN rounded down to a half point, and the highest
        value plus AXIS_MARGIN rounded up to a half point.

    Raises:
        ValueError: When there are no values.
    """
    if len(values) == 0:
        raise ValueError("Axis limits need scores, but there are no values")
    return math.floor(2 * (min(values) - AXIS_MARGIN)) / 2, math.ceil(2 * (max(values) + AXIS_MARGIN)) / 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results/teacher/runs.csv")
    parser.add_argument("--mini", default="results/teacher/probe.csv", help="Has the echonext_mini row")
    parser.add_argument("--output", default="results/teacher/figures/runs.png")
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    rows = build_rows(table)
    mini = pd.read_csv(args.mini).query("model == 'echonext_mini'").iloc[0]
    flags = [column for column in mini.index if column.endswith("_flag")]
    mini_scores = {"val_mean_auroc": 100 * mini[flags].astype(float).mean(), "val_shd_auroc": 100 * mini[SHD_COLUMN]}

    # One line per setting, plus a heading and a gap for each group.
    lines = len(rows) + 2 * len({group for group, _, _ in rows})
    fig, axes = plt.subplots(1, 2, figsize=(16, max(9, 0.45 * lines + 1.5)), sharey=True)
    for ax, column, title in (
        (axes[0], "val_mean_auroc", "Average over all 12 outputs"),
        (axes[1], "val_shd_auroc", "Overall heart disease (SHD)"),
    ):
        limits = axis_limits([*(100 * table[column]), mini_scores[column]])
        position, last_group, ticks = 0, None, []
        for group, label, runs in rows:
            if group != last_group:
                position += 1 if last_group else 0
                ax.text(limits[0], position - 0.15, group, fontweight="bold", color=GROUP_COLORS[group], va="bottom",
                        zorder=4, bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 2})
                position += 1
                last_group = group
            for _, run in runs.iterrows():
                filled = run["seed"] == 42
                ax.scatter(100 * run[column], position, s=150, zorder=3, color=GROUP_COLORS[group],
                           facecolors=GROUP_COLORS[group] if filled else "none", linewidths=2)
            ticks.append((position, label))
            position += 1
        mini_value = mini_scores[column]
        ax.axvline(mini_value, color=INK, linestyle="--", linewidth=2, zorder=2)
        ax.text(mini_value + 0.05, position - 0.1, f"EchoNext-Mini {mini_value:.1f}", va="bottom", fontsize=14)
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.set_xlabel("Validation AUROC, best epoch of each run")
        ax.set_ylim(position, -0.6)
    axes[0].set_yticks([p for p, _ in ticks], [label for _, label in ticks])
    axes[0].scatter([], [], s=150, color=INK, label="Seed 42")
    axes[0].scatter([], [], s=150, facecolors="none", edgecolors=INK, linewidths=2, label="Other seeds")
    # Below the panels, so the key never covers a dot however many runs there are.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.suptitle(f"Every Teacher training run ({len(table)} runs)", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
