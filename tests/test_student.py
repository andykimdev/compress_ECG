import pytest
import torch

from compress_ecg.baseline import DEFAULT_MINI_DIR
from compress_ecg.profiling import synthetic_inputs
from compress_ecg.student import build_student, load_student, save_student


@pytest.fixture
def released_code():
    if not (DEFAULT_MINI_DIR / "cradlenet/models/resnet1d_tabular.py").is_file():
        pytest.skip("Run bash scripts/download_weights.sh --mini-only for Student checks")


@pytest.mark.parametrize("width,parameters", [(5, 104_436), (16, 1_046_080)])
def test_width_gives_the_planned_parameter_count(released_code, width, parameters):
    model = build_student(width)
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_student_follows_the_call_format(released_code):
    logits = build_student().eval()(*synthetic_inputs(2))
    assert logits.shape == (2, 12)


def test_saved_student_gives_the_same_outputs(released_code, tmp_path):
    torch.manual_seed(0)
    student = build_student().eval()
    loaded = load_student(save_student(student, tmp_path / "nested" / "student.pt"))
    inputs = synthetic_inputs(2)
    torch.testing.assert_close(student(*inputs), loaded(*inputs))
