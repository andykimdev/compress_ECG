"""
Turn saved evaluation, timing, and profiling results into W&B summary runs.

Purpose:
- Summarize per-label metrics, seed spreads, and paired seed gaps as flat score names.
- Build one run spec per model or measurement, so W&B holds every number the slides report.

Special Notes:
- Only aggregate numbers are built here. No patient-level value is ever included.
- Every builder is pure. scripts/log_results_wandb.py reads the files and sends the runs.
"""

from __future__ import annotations

import math
import re
import statistics

from scipy import stats

from .data import LABEL_COLUMNS, SHD_INDEX

SHD_LABEL = LABEL_COLUMNS[SHD_INDEX]
# The W&B group that holds every run built here.
GROUP = "evaluation"
# Student arms by the path prefix of their seed folders, relative to results/student.
STUDENT_ARMS = {
    "distill": "student_distill", "labels_only": "student_labels_only",
    "mini_copy": "stand_in/mini_teacher_phoenix",
}


def label_scores(metrics: dict) -> dict:
    """
    Return SHD AUROC, the mean AUROC over all labels, and SHD Brier score.

    Args:
        metrics: Label name to a dict with auroc and brier, for every label.

    Returns:
        shd_auroc, mean_auroc, and shd_brier.

    Raises:
        KeyError: When SHD or a label's auroc is missing.
    """
    return {
        "shd_auroc": metrics[SHD_LABEL]["auroc"],
        "mean_auroc": statistics.fmean(scores["auroc"] for scores in metrics.values()),
        "shd_brier": metrics[SHD_LABEL]["brier"],
    }


def prefixed(prefix: str, scores: dict) -> dict:
    """Return scores with each name starting with prefix and an underscore."""
    return {f"{prefix}_{name}": value for name, value in scores.items()}


def seed_spread(values: list[float]) -> dict:
    """
    Return the mean and sample standard deviation of one score over seeds.

    Raises:
        ValueError: When fewer than 2 seeds are given, since a spread needs 2.
    """
    if len(values) < 2:
        raise ValueError(f"A seed spread needs at least 2 values, got {len(values)}")
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values), "n_seeds": len(values)}


def paired_gap(values: list[float], baseline: list[float], confidence: float = 0.95) -> dict:
    """
    Return the mean seed-by-seed gap between two arms and its t interval.

    Args:
        values: One score per seed for the arm of interest.
        baseline: The same score for the compared arm, seed for seed.
        confidence: Coverage of the interval, from 0 to 1.

    Returns:
        gap, gap_low, gap_high, seeds_ahead (seeds where values beat baseline), and n_seeds.

    Raises:
        ValueError: When the lists differ in length or hold fewer than 2 seeds.
    """
    if len(values) != len(baseline) or len(values) < 2:
        raise ValueError(f"Paired gaps need 2 or more matching seeds, got {len(values)} and {len(baseline)}")
    gaps = [value - base for value, base in zip(values, baseline)]
    half = stats.t.ppf((1 + confidence) / 2, len(gaps) - 1) * statistics.stdev(gaps) / math.sqrt(len(gaps))
    mean = statistics.fmean(gaps)
    return {"gap": mean, "gap_low": mean - half, "gap_high": mean + half,
            "seeds_ahead": sum(gap > 0 for gap in gaps), "n_seeds": len(gaps)}


def seed_number(folder: str) -> int:
    """
    Return the seed of a folder named <prefix>_s<seed>, such as 42 for student_distill_s42.

    Raises:
        ValueError: When the name does not end in _s and a number.
    """
    match = re.fullmatch(r".+_s(\d+)", folder)
    if match is None:
        raise ValueError(f"Folder {folder} has no seed, so it must end in _s<seed>")
    return int(match.group(1))


def paired_seed_gap(values: dict[int, float], baseline: dict[int, float]) -> dict:
    """
    Return paired_gap for two arms, matched by seed number instead of list position.

    Args:
        values: Seed to score for the arm of interest.
        baseline: Seed to score for the compared arm.

    Raises:
        ValueError: When the two arms do not hold the same seeds.
    """
    if values.keys() != baseline.keys():
        raise ValueError(f"Paired gaps need the same seeds, got {sorted(values)} and {sorted(baseline)}")
    seeds = sorted(values)
    return paired_gap([values[seed] for seed in seeds], [baseline[seed] for seed in seeds])


def speed_scores(timing: dict, *, rows: tuple[tuple[str, int], ...]) -> dict:
    """
    Return ms per ECG and peak GPU memory for chosen precisions and batch sizes.

    Args:
        timing: A speed file from time_teacher.py or time_student.py.
        rows: (precision, batch size) pairs to keep.

    Returns:
        <precision>_b<batch>_ms_per_ecg and <precision>_b<batch>_peak_gpu_bytes for each pair.

    Raises:
        KeyError: When a pair has no row in the file.
    """
    found = {(row["precision"], row["batch_size"]): row for row in timing["rows"]}
    scores = {}
    for precision, batch in rows:
        row = found[(precision, batch)]
        scores[f"{precision}_b{batch}_ms_per_ecg"] = row["ms_per_ecg"]
        scores[f"{precision}_b{batch}_peak_gpu_bytes"] = row["peak_gpu_memory_bytes"]
    return scores


def arm_scores(seed_metrics: list[dict]) -> dict:
    """Return the seed mean and spread of each label score for one Student arm."""
    per_seed = [label_scores(metrics) for metrics in seed_metrics]
    scores = {}
    for name in per_seed[0]:
        spread = seed_spread([seed[name] for seed in per_seed])
        scores.update({f"val_{name}_mean": spread["mean"], f"val_{name}_sd": spread["sd"]})
    scores["n_seeds"] = len(per_seed)
    return scores


def wait_range(profiles: list[dict], num_workers: int) -> dict:
    """Return the lowest and highest median data wait, in ms, over profiling rounds for one worker count."""
    waits = [row["median_wait_ms"] for profile in profiles for row in profile["rows"]
             if row["num_workers"] == num_workers]
    if not waits:
        raise ValueError(f"No profiling row has num_workers={num_workers}")
    return {"wait_ms_low": min(waits), "wait_ms_high": max(waits), "rounds": len(waits)}


def run_spec(name: str, summary: dict, config: dict) -> dict:
    """Return one W&B run spec in the evaluation group."""
    return {"name": name, "group": GROUP, "job_type": "evaluation", "config": config, "summary": summary}


def build_runs(results: dict) -> list[dict]:
    """
    Build every evaluation, timing, and profiling run from loaded result files.

    Args:
        results: The output of scripts/log_results_wandb.py load_results, with metric dicts per
            model and split, Student seed metrics per arm, FP16 rows, speed files, and profiles.

    Returns:
        Run specs with name, group, job_type, config, and summary.
    """
    arms = {arm: [seed for _, seed in sorted(results["students"][arm].items())] for arm in STUDENT_ARMS}
    distill_shd, labels_shd = (
        {seed_number(folder): label_scores(metrics)["shd_auroc"] for folder, metrics in results["students"][arm].items()}
        for arm in ("distill", "labels_only")
    )
    fp16 = results["fp16"]
    teacher_speed = results["teacher_speed"]
    student_speed = results["student_speed"]
    runs = [
        run_spec("eval_echonext_mini", {
            **prefixed("val", label_scores(results["mini_val"])),
            **prefixed("test", label_scores(results["mini_test"])),
        }, {"model": "EchoNext-Mini", "weights": "released"}),
        run_spec("eval_logistic_regression", prefixed("val", label_scores(results["logistic_val"])),
                 {"model": "logistic regression", "inputs": "45 ECG summaries and the 7 routine numbers"}),
        run_spec("eval_gradient_boosting", {
            **prefixed("val", label_scores(results["boosting_val"])),
            **prefixed("test", label_scores(results["boosting_test"])),
        }, {"model": "gradient boosting", "inputs": "the 7 routine numbers"}),
        run_spec("eval_teacher", {
            **prefixed("val", label_scores(results["teacher_val"])),
            **prefixed("val_gap_vs_mini", {key: results["teacher_gap"][key] for key in ("gap", "gap_low", "gap_high")}),
            "gap_resamples": results["teacher_gap"]["n_resamples"],
            **speed_scores(teacher_speed, rows=(("fp32", 1), ("fp32", 64), ("bf16", 64))),
        }, {"model": "Teacher, weight average of 6 runs", "parameters": teacher_speed["parameters"],
            "gpu": teacher_speed["gpu"], "warmup": teacher_speed["warmup"], "repeats": teacher_speed["repeats"]}),
        run_spec("eval_student_distill", {
            **arm_scores(arms["distill"]),
            **prefixed("val_shd_auroc", {key: value for key, value in paired_seed_gap(distill_shd, labels_shd).items()
                                         if key != "n_seeds"}),
            "fp16_val_shd_auroc_mean": statistics.fmean(row["shd16"] for row in fp16["rows"]),
            "fp16_max_shd_auroc_change": max(abs(row["shd16"] - row["shd32"]) for row in fp16["rows"]),
            "fp32_weight_bytes": fp16["weight_and_buffer_bytes"]["fp32"],
            "fp16_weight_bytes": fp16["weight_and_buffer_bytes"]["fp16"],
            **speed_scores(student_speed, rows=(("fp32", 1), ("fp32", 64), ("fp16", 64))),
        }, {"model": "Student, distilled from the Teacher", "parameters": student_speed["parameters"],
            "gap_compared_with": "Student, labels only", "gpu": student_speed["gpu"],
            "warmup": student_speed["warmup"], "repeats": student_speed["repeats"]}),
        run_spec("eval_student_labels_only", arm_scores(arms["labels_only"]),
                 {"model": "Student, true labels only"}),
        run_spec("eval_student_mini_copy", arm_scores(arms["mini_copy"]),
                 {"model": "Student, distilled from EchoNext-Mini"}),
    ]
    before, after = results["profiles_before"], results["profiles_after"]
    runs.append(run_spec("profile_student_data_loading", {
        **prefixed("before_w0", wait_range(before, 0)),
        **prefixed("after_w4", wait_range(after, 4)),
    }, {"workload": before[0]["workload"], "batch_size": before[0]["batch_size"],
        "warmup_steps": before[0]["warmup_steps"], "timed_steps": before[0]["timed_steps"], "gpu": before[0]["gpu"]}))
    return runs
