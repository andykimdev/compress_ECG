"""Check the plan that scripts/sweep_teacher.py prints in a dry run. No run is started."""

import json
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
HOLDOUT_HEAD = "checkpoints/teacher/probe_ho0.2_hs0/rescaled_lr0.001.pt"
WHOLE_SPLIT_HEAD = "checkpoints/teacher/probe/rescaled_lr0.001.pt"


@pytest.fixture
def dry_run(tmp_path, monkeypatch, load_script):
    """Return a function that dry runs a sweep config from inside an empty working folder."""
    monkeypatch.chdir(tmp_path)
    script = load_script("sweep_teacher")
    return lambda config: script.main(["--config", str(config), "--dry-run"])


def make_file(path: str) -> None:
    """Create an empty file at a path under the working folder."""
    Path(path).parent.mkdir(parents=True)
    Path(path).touch()


def test_dry_run_of_the_holdout_config_fails_while_its_output_layer_is_missing(dry_run, capsys):
    # The whole-split layer exists, as on the server, but the holdout one has not been built.
    make_file(WHOLE_SPLIT_HEAD)
    with pytest.raises(FileNotFoundError, match="run_teacher_probe.py --holdout-fraction 0.2 --holdout-seed 0"):
        dry_run(CONFIGS / "teacher_holdout.json")
    assert "run_teacher.py" not in capsys.readouterr().out


def test_dry_run_of_the_holdout_config_plans_three_runs_once_its_output_layer_exists(dry_run, capsys):
    make_file(HOLDOUT_HEAD)
    dry_run(CONFIGS / "teacher_holdout.json")
    commands = [line for line in capsys.readouterr().out.splitlines() if "run_teacher.py" in line]
    assert len(commands) == 3
    assert all(f"--score-splits dev" in command and f"--probe-head {HOLDOUT_HEAD}" in command for command in commands)


def test_dry_run_refuses_a_holdout_sweep_that_starts_from_the_whole_split_output_layer(dry_run, tmp_path):
    make_file(WHOLE_SPLIT_HEAD)
    config = json.loads((CONFIGS / "teacher_holdout.json").read_text())
    config["base"]["probe_head"] = WHOLE_SPLIT_HEAD
    (tmp_path / "leaking.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fitted without them"):
        dry_run(tmp_path / "leaking.json")


def test_dry_run_of_a_sweep_without_a_holdout_accepts_the_whole_split_output_layer(dry_run, capsys):
    make_file(WHOLE_SPLIT_HEAD)
    dry_run(CONFIGS / "teacher_ema.json")
    commands = [line for line in capsys.readouterr().out.splitlines() if "run_teacher.py" in line]
    assert len(commands) == 6
