"""Tests for Role D's model quantization."""

import torch
from torch import nn

from compress_ecg.quantize import to_fp16


class TinyECGModel(nn.Module):
    """Small stand-in using the team's waveform/tabular interface."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(7, 12)

    def forward(self, waveform, tabular):
        return self.fc(tabular) + waveform.mean(dim=(1, 2)).unsqueeze(1)


def test_fp16_conversion():
    torch.manual_seed(42)

    original = TinyECGModel().eval()
    fp16 = to_fp16(original)

    waveform = torch.randn(2, 12, 2500)
    tabular = torch.randn(2, 7)

    with torch.inference_mode():
        expected = original(waveform, tabular)
        actual = fp16(waveform, tabular)

    # The original model must remain FP32.
    assert next(original.parameters()).dtype == torch.float32

    # The converted model must have FP16 parameters.
    assert next(fp16.parameters()).dtype == torch.float16

    # Outputs must remain compatible with the shared pipeline.
    assert actual.shape == (2, 12)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()

    # Quantization should not substantially change predictions.
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
