"""
Check the timing of a model's forward pass and the summary of the timings.

Purpose:
- Pin the timed region with a clock the test controls, so no real time is read.
- Pin the summary numbers to values worked out by hand.
"""

import json

import pytest
import torch
from torch import nn

from compress_ecg.data import LABEL_COLUMNS
from compress_ecg.timing import forward_seconds, timing_summary

WAVEFORM = torch.zeros(4, 12, 2500)
TABULAR = torch.zeros(4, 7)


class FakeClock:
    """A clock that moves only when the test moves it."""

    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class SlowModel(nn.Module):
    """Advance the fake clock by the next duration on every call, then return zero logits."""

    def __init__(self, clock: FakeClock, durations: list[float]):
        super().__init__()
        self.clock = clock
        self.durations = iter(durations)

    def forward(self, waveform, tabular):
        self.clock.now += next(self.durations)
        return torch.zeros(len(waveform), len(LABEL_COLUMNS))


def test_every_timed_call_is_measured_on_its_own():
    clock = FakeClock()
    model = SlowModel(clock, [2.0, 3.0, 4.0])

    seconds = forward_seconds(model, WAVEFORM, TABULAR, warmup=0, repeats=3, clock=clock)

    assert seconds == [2.0, 3.0, 4.0]


def test_warmup_calls_are_left_out_of_the_timings():
    clock = FakeClock()
    model = SlowModel(clock, [10.0, 10.0, 1.0, 1.0, 1.0])

    seconds = forward_seconds(model, WAVEFORM, TABULAR, warmup=2, repeats=3, clock=clock)

    assert seconds == [1.0, 1.0, 1.0]


def test_waiting_for_the_device_after_a_call_counts_and_waiting_before_it_does_not():
    clock = FakeClock()
    model = SlowModel(clock, [1.0, 1.0])
    waits = iter([50.0, 5.0, 50.0, 5.0])  # Before call 1, after call 1, before call 2, after call 2.

    def synchronize():
        clock.now += next(waits)

    seconds = forward_seconds(model, WAVEFORM, TABULAR, warmup=0, repeats=2, synchronize=synchronize, clock=clock)

    assert seconds == [6.0, 6.0]


@pytest.mark.parametrize("warmup, repeats, message", [(0, 0, "repeats"), (-1, 1, "warmup")])
def test_bad_call_counts_are_refused(warmup, repeats, message):
    with pytest.raises(ValueError, match=message):
        forward_seconds(SlowModel(FakeClock(), []), WAVEFORM, TABULAR, warmup=warmup, repeats=repeats)


def test_summary_matches_the_hand_worked_values():
    # Sorted, the timings are 0.1, 0.2, 0.3, 0.4 and 0.9 s. The median is 0.3 s for a batch of 4, so
    # 300 ms per batch, 75 ms per ECG and 4 / 0.3 ECGs per second. The mean would be 0.38 s.
    # The 5th percentile sits 0.2 of the way from 0.1 to 0.2 s, which is 120 ms per batch and 30 ms
    # per ECG. The 95th sits 0.8 of the way from 0.4 to 0.9 s, which is 800 ms and 200 ms per ECG.
    summary = timing_summary([0.9, 0.1, 0.3, 0.2, 0.4], batch_size=4)

    assert summary == {
        "batch_size": 4, "repeats": 5, "ms_per_batch": pytest.approx(300.0), "ms_per_ecg": pytest.approx(75.0),
        "ms_per_ecg_p5": pytest.approx(30.0), "ms_per_ecg_p95": pytest.approx(200.0),
        "ecgs_per_second": pytest.approx(4 / 0.3),
    }


def test_summary_of_one_timing_has_no_spread():
    summary = timing_summary([0.2], batch_size=1)

    assert (summary["ms_per_ecg"], summary["ms_per_ecg_p5"], summary["ms_per_ecg_p95"]) == (
        pytest.approx(200.0), pytest.approx(200.0), pytest.approx(200.0),
    )
    assert summary["ecgs_per_second"] == pytest.approx(5.0)


@pytest.mark.parametrize("seconds, batch_size, message", [
    ([], 4, "at least one"),
    ([0.1], 0, "batch_size"),
    ([0.1, 0.0], 4, "above 0"),
])
def test_summary_refuses_bad_inputs(seconds, batch_size, message):
    with pytest.raises(ValueError, match=message):
        timing_summary(seconds, batch_size=batch_size)


class ZeroModel(nn.Module):
    """Return zero logits in the shared call format."""

    def forward(self, waveform, tabular):
        return torch.zeros(len(waveform), len(LABEL_COLUMNS))


def run_command(load_script, monkeypatch, data_dir, tmp_path, *extra: str) -> dict:
    """Run time_teacher.py on the small dataset with a stand-in model and return the saved summary."""
    script = load_script("time_teacher")
    monkeypatch.setattr(script, "load_frozen_teacher", lambda checkpoint, device: ZeroModel())
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"stand-in weights")
    output = tmp_path / "results" / "speed.json"
    script.main([
        "--checkpoint", str(checkpoint), "--data-dir", str(data_dir), "--warmup", "0", "--repeats", "2",
        "--output", str(output), *extra,
    ])
    return json.loads(output.read_text())


def test_command_writes_one_row_per_precision_and_batch_size(load_script, monkeypatch, data_dir, tmp_path):
    saved = run_command(load_script, monkeypatch, data_dir, tmp_path, "--batch-sizes", "1", "4", "--code-commit", "abc123")

    assert (saved["split"], saved["device"], saved["gpu"], saved["warmup"]) == ("val", "cpu", None, 0)
    assert saved["code"]["commit"] == "abc123"
    # SHA-256 of the 16 bytes written above, from shasum on the command line.
    assert saved["checkpoint_sha256"] == "becdb1ed3bbc3a63808c20e1fc2695a73b1c61ceee7a4b219680d6e32a6a4db5"
    assert [(row["precision"], row["batch_size"], row["repeats"]) for row in saved["rows"]] == [
        ("fp32", 1, 2), ("fp32", 4, 2),
    ]
    assert all(row["peak_gpu_memory_bytes"] is None and row["ecgs_per_second"] > 0 for row in saved["rows"])


def test_command_refuses_a_batch_larger_than_the_split(load_script, monkeypatch, data_dir, tmp_path):
    # The small dataset has 6 Validation ECGs.
    with pytest.raises(SystemExit, match="6 ECGs"):
        run_command(load_script, monkeypatch, data_dir, tmp_path, "--batch-sizes", "7")
