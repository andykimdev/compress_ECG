"""
Time a model's forward pass on batches that are already in memory.

Purpose:
- Measure how long each forward pass takes, after warm-up calls.
- Summarize the timings as time per ECG and ECGs per second.

Special Notes:
- Reading from disk is not timed. The caller supplies the batch and the device wait.
- The summary holds no patient data and is safe to commit.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn

# Percentiles that bound the middle 90 percent of the timings.
SPREAD = (5, 95)


def forward_seconds(
    model: nn.Module, waveform: torch.Tensor, tabular: torch.Tensor, *, warmup: int = 10, repeats: int = 50,
    synchronize: Callable[[], None] | None = None, clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """
    Time repeated forward passes of a model on one batch.

    Args:
        model: Takes waveform [batch, 12, 2500] and tabular [batch, 7].
        waveform: The batch of waveforms, already on the model's device.
        tabular: The batch of the 7 numbers, on the same device.
        warmup: Calls made first and not timed, so one-time setup is left out.
        repeats: Timed calls.
        synchronize: Waits for the device to finish its queued work, such as
            torch.cuda.synchronize. None means the device runs each call to the end itself.
        clock: Returns the current time in seconds.

    Returns:
        The seconds each timed call took, in order.

    Raises:
        ValueError: When warmup is negative or repeats is below 1.

    Notes:
        A GPU returns from a call before the work is done. The wait runs before the clock
        starts, so earlier work is not counted, and again before the clock stops.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be 0 or more, got {warmup}")
    if repeats < 1:
        raise ValueError(f"repeats must be 1 or more, got {repeats}")
    wait = synchronize if synchronize is not None else lambda: None
    for _ in range(warmup):
        model(waveform, tabular)
    seconds = []
    for _ in range(repeats):
        wait()
        start = clock()
        model(waveform, tabular)
        wait()
        seconds.append(clock() - start)
    return seconds


def timing_summary(seconds: Sequence[float], batch_size: int) -> dict:
    """
    Summarize the timings of one batch size.

    Args:
        seconds: Seconds per forward pass, from forward_seconds.
        batch_size: ECGs in the batch that was timed.

    Returns:
        batch_size, repeats, the median ms_per_batch and ms_per_ecg, the 5th and 95th
        percentiles of the time per ECG as ms_per_ecg_p5 and ms_per_ecg_p95, and
        ecgs_per_second at the median.

    Raises:
        ValueError: When there are no timings, a timing is not above 0, or batch_size is below 1.
    """
    if len(seconds) == 0:
        raise ValueError("A summary needs at least one timing")
    if batch_size < 1:
        raise ValueError(f"batch_size must be 1 or more, got {batch_size}")
    if min(seconds) <= 0:
        raise ValueError(f"Every timing must be above 0 seconds, got {min(seconds)}")
    median = float(np.median(seconds))
    low, high = (float(value) for value in np.percentile(seconds, SPREAD))
    return {
        "batch_size": batch_size, "repeats": len(seconds), "ms_per_batch": median * 1000,
        "ms_per_ecg": median * 1000 / batch_size, "ms_per_ecg_p5": low * 1000 / batch_size,
        "ms_per_ecg_p95": high * 1000 / batch_size, "ecgs_per_second": batch_size / median,
    }


def step_summary(wait_seconds: Sequence[float], compute_seconds: Sequence[float], batch_size: int) -> dict:
    """
    Summarize training steps split into waiting for data and computing.

    Args:
        wait_seconds: Seconds each step waited for its batch from the loader.
        compute_seconds: Seconds each step spent on the device work, from copying the
            batch in to the optimizer update finishing.
        batch_size: ECGs per step.

    Returns:
        batch_size, steps, the median wait and compute times in ms, wait_share as the
        share of all step time spent waiting, and ecgs_per_second over all steps.

    Raises:
        ValueError: When there are no steps or the two lists differ in length.
    """
    if not wait_seconds or len(wait_seconds) != len(compute_seconds):
        raise ValueError("Give one wait and one compute timing for every step, and at least one step")
    waited, computed = float(np.sum(wait_seconds)), float(np.sum(compute_seconds))
    return {
        "batch_size": batch_size, "steps": len(wait_seconds),
        "median_wait_ms": float(np.median(wait_seconds)) * 1000,
        "median_compute_ms": float(np.median(compute_seconds)) * 1000,
        "wait_share": waited / (waited + computed),
        "ecgs_per_second": batch_size * len(wait_seconds) / (waited + computed),
    }
