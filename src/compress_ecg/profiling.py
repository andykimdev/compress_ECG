"""Measure inference speed, memory, and model size using synthetic ECG inputs."""

from __future__ import annotations

import io
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from typing import Callable
from urllib.parse import quote

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn

BACKENDS = ("eager", "compile", "onnx")
MEMORY_SAMPLE_SECONDS = 0.005


def synthetic_inputs(batch_size: int = 1, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create random inputs in the shared call format, containing no patient data.

    Args:
        batch_size: Number of synthetic ECGs.
        seed: Random seed.

    Returns:
        Waveform float32 [batch, 12, 2500] and tabular float32 [batch, 7].
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch_size, 12, 2500, generator=generator),
        torch.randn(batch_size, 7, generator=generator),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _measure_memory(forward: Callable[[], torch.Tensor], device: torch.device, repeats: int) -> dict:
    """Sample process RSS during forwards and read the CUDA allocator's peak when available."""
    process = psutil.Process()
    initial = process.memory_info().rss
    samples = [initial]
    done = Event()

    def sample_rss():
        while not done.wait(MEMORY_SAMPLE_SECONDS):
            samples.append(process.memory_info().rss)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        for _ in range(repeats):
            forward()
            _synchronize(device)
            samples.append(process.memory_info().rss)
    finally:
        done.set()
        sampler.join()
    return {
        "rss_start_bytes": initial,
        "rss_peak_sampled_bytes": max(samples),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def _onnx_forward(model: nn.Module, inputs: tuple[torch.Tensor, ...], path: Path):
    try:
        import onnx  # noqa: F401 - required by the PyTorch exporter
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install ONNX profiling dependencies with pip install -e '.[onnx]'") from error

    input_names = ("waveform", "tabular")
    # The released model squeezes axis 1 of its 12-output tensor. PyTorch treats
    # that as a no-op. The TorchScript exporter preserves that behavior in ONNX.
    torch.onnx.export(
        model, inputs, path, input_names=input_names, output_names=["logits"],
        dynamo=False, opset_version=17, verbose=False,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    arrays = dict(zip(input_names, (tensor.detach().numpy() for tensor in inputs)))
    feed = {argument.name: arrays[argument.name] for argument in session.get_inputs()}

    def forward():
        return torch.from_numpy(session.run(["logits"], feed)[0])

    return forward, ort.__version__


def _torch_trace(forward: Callable[[], torch.Tensor], device: torch.device, path: Path) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = torch.profiler.ProfilerActivity.CUDA
    if device.type == "cuda" and cuda in torch.profiler.supported_activities():
        activities.append(cuda)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as trace:
        forward()
        _synchronize(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.export_chrome_trace(str(path))


def profile_models(
    models: dict[str, nn.Module], *, batch_sizes: tuple[int, ...] = (1, 8, 32, 64), device: str = "cpu",
    backends: tuple[str, ...] = ("eager",), warmup: int = 5, repeats: int = 30,
    checkpoint_paths: dict[str, str | Path] | None = None, trace_dir: str | Path | None = None,
    compile_backend: str = "inductor",
) -> pd.DataFrame:
    """Benchmark warm inference for models that accept waveform and tabular tensors.

    Models move to the chosen device and enter eval mode. Inputs stay on that
    device. Timings exclude compilation, export, memory sampling, and traces.
    RSS is a sampled process metric and includes the runtime and its caches.
    """
    if not models or not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("Provide at least one model and positive batch sizes")
    if not backends or any(backend not in BACKENDS for backend in backends):
        raise ValueError(f"Choose profiling backends from {BACKENDS}")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    target = torch.device(device)
    if "onnx" in backends and target.type != "cpu":
        raise ValueError("The ONNX Runtime benchmark uses the CPU execution provider; use --device cpu")

    rows = []
    with TemporaryDirectory(prefix="echonext-profile-") as scratch:
        onnx_path = Path(scratch) / "model.onnx"
        for name, model in models.items():
            model = model.to(target).eval()
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
            if checkpoint_paths and name in checkpoint_paths:
                pytorch_size = Path(checkpoint_paths[name]).stat().st_size
                pytorch_format = "checkpoint"
            else:
                with io.BytesIO() as buffer:
                    torch.save(model.state_dict(), buffer)
                    pytorch_size = buffer.tell()
                pytorch_format = "state_dict"
            compiled = torch.compile(model, backend=compile_backend, dynamic=False) if "compile" in backends else None
            for batch_size in batch_sizes:
                inputs = tuple(tensor.to(device=target, dtype=dtype) for tensor in synthetic_inputs(batch_size))
                with torch.inference_mode():
                    expected = model(*inputs)
                if expected.shape != (batch_size, 12) or not torch.isfinite(expected).all():
                    raise ValueError(f"{name} must return finite logits with shape [{batch_size}, 12]")

                for backend in backends:
                    _synchronize(target)
                    setup_start = perf_counter()
                    artifact_format, artifact_bytes = pytorch_format, pytorch_size
                    runtime_version = torch.__version__
                    if backend == "onnx":
                        forward, runtime_version = _onnx_forward(model, inputs, onnx_path)
                        artifact_format, artifact_bytes = "onnx", onnx_path.stat().st_size
                    else:
                        runner = compiled if backend == "compile" else model

                        def forward():
                            return runner(*inputs)

                    with torch.inference_mode():
                        actual = forward()
                        _synchronize(target)
                        setup_seconds = perf_counter() - setup_start
                        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
                        del actual
                        for _ in range(warmup):
                            forward()
                        _synchronize(target)
                        timings = []
                        for _ in range(repeats):
                            start = perf_counter()
                            forward()
                            _synchronize(target)
                            timings.append(perf_counter() - start)
                        memory = _measure_memory(forward, target, min(repeats, 5))
                        if trace_dir is not None and backend != "onnx":
                            filename = f"{quote(name, safe='')}_{backend}_batch{batch_size}.json"
                            _torch_trace(forward, target, Path(trace_dir) / filename)

                    mean_seconds = float(np.mean(timings))
                    rows.append({
                        "model": name, "backend": backend, "device": str(target), "batch_size": batch_size,
                        "latency_mean_ms": mean_seconds * 1000,
                        "latency_p50_ms": float(np.median(timings)) * 1000,
                        "latency_p95_ms": float(np.percentile(timings, 95)) * 1000,
                        "ecgs_per_second": batch_size / mean_seconds,
                        **memory,
                        "parameter_count": parameter_count, "artifact_format": artifact_format,
                        "artifact_bytes": artifact_bytes, "setup_seconds": setup_seconds,
                        "warmup": warmup, "repeats": repeats, "cpu_threads": torch.get_num_threads(),
                        "compile_backend": compile_backend if backend == "compile" else None,
                        "runtime_version": runtime_version, "torch_version": torch.__version__,
                        "platform": platform.platform(),
                        "hardware": torch.cuda.get_device_name(target) if target.type == "cuda" else platform.machine(),
                    })
                    del forward
    return pd.DataFrame(rows)
