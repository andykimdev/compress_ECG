"""
Measure how fast and how large each model is.

Purpose:
- Create synthetic inputs, the only inputs allowed on hosted devices.
- Measure speed and memory for every model.

Special Notes:
- Owner: Role A, after the baselines. profile_models is still to be written.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn


def synthetic_inputs(batch_size: int = 1, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create random inputs in the shared call format, containing no patient data.

    Args:
        batch_size: Number of synthetic ECGs.
        seed: Random seed.

    Returns:
        Waveform float32 [batch, 12, 2500] and tabular float32 [batch, 7].
    """
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch_size, 12, 2500, generator=generator),
        torch.randn(batch_size, 7, generator=generator),
    )


def profile_models(
    models: dict[str, nn.Module], *, batch_sizes: tuple[int, ...] = (1, 8, 32, 64), device: str = "cpu",
) -> pd.DataFrame:
    """
    Measure each model across batch sizes.

    Args:
        models: Model name to model, each in the shared call format.
        batch_sizes: Batch sizes to sweep for throughput.
        device: Where the models run.

    Returns:
        One row per model and batch size, with latency, throughput, memory, and file size. The
        table is aggregate, so it can be saved under results/.
    """
    raise NotImplementedError("Role A: use PyTorch Profiler, torch.compile, and ONNX Runtime")
