from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def _write_design(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "exp_mock_01",
                        "title": "Mock Experiment",
                        "goal_id": "G1",
                        "addresses_questions": ["Q1"],
                        "hypothesis": "Mock hypothesis",
                        "model": "tiny-net",
                        "dataset": "mockset",
                        "baselines": ["baseline-a"],
                        "metrics": ["accuracy"],
                        "ablations": ["remove-x"],
                        "success_criteria": "accuracy > 0.7",
                        "estimated_runtime_hours": 0.1,
                        "end_stage": 4,
                        "end_stage_rationale": "full validation",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_experiment_cli_mock_creates_canonical_outputs(tmp_path: Path) -> None:
    design_path = tmp_path / "experiment_workspace" / "experiment_design.json"
    _write_design(design_path)

    cmd = [
        sys.executable,
        "-m",
        "consortium.toolkits.experimentation.run_experiment_cli",
        "--workspace",
        str(tmp_path),
        "--spec-file",
        str(design_path),
        "--experiment-id",
        "exp_mock_01",
        "--mock",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["experiment_id"] == "exp_mock_01"
    assert payload["status"] == "success"
    assert payload["end_stage_executed"] == 4

    run_root = tmp_path / payload["run_dir"]
    assert (run_root / "execution_result.json").exists()
    experiment_dirs = list((run_root / "experiments").iterdir())
    assert experiment_dirs
    log_dir = experiment_dirs[0] / "logs" / "0-run"
    assert (log_dir / "baseline_summary.json").exists()
    assert (log_dir / "research_summary.json").exists()
    assert (log_dir / "ablation_summary.json").exists()

    execution_log = json.loads((tmp_path / "experiment_workspace" / "execution_log.json").read_text(encoding="utf-8"))
    assert execution_log["runs"][0]["experiment_id"] == "exp_mock_01"


def test_run_experiment_cli_selects_single_spec_without_experiment_id(tmp_path: Path) -> None:
    spec_path = tmp_path / "experiment_workspace" / "single_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp_single",
                "title": "Single Spec",
                "hypothesis": "Single hypothesis",
                "metrics": ["loss"],
                "end_stage": 2,
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "consortium.toolkits.experimentation.run_experiment_cli",
        "--workspace",
        str(tmp_path),
        "--spec-file",
        str(spec_path),
        "--mock",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["end_stage_executed"] == 2


def test_run_experiment_cli_stage_engine_writes_canonical_summaries(tmp_path: Path) -> None:
    from consortium.toolkits.experimentation.run_experiment_cli import _run_cli_stages

    run_root = tmp_path / "experiment_runs" / "run1"
    run_root.mkdir(parents=True)
    spec = {
        "experiment_id": "exp_stage_01",
        "title": "Stage Experiment",
        "hypothesis": "Stage hypothesis",
        "metrics": ["accuracy"],
        "end_stage": 4,
    }

    fake_summary = json.dumps(
        {
            "baseline": {"accuracy": 0.61},
            "tuning": {"accuracy": 0.68},
            "creative": {"accuracy": 0.72},
            "ablation": {"accuracy": 0.70},
        }
    )

    with patch("consortium.experiment_runner.run_experiment_stages", return_value=fake_summary) as mocked:
        rc, exp_dir, failure_reason = _run_cli_stages(
            run_root=run_root,
            spec=spec,
            timeout_seconds=5,
            backend="claude",
            model="test-model",
        )

    assert rc == 0
    assert failure_reason is None
    assert exp_dir is not None
    mocked.assert_called_once()
    log_dir = exp_dir / "logs" / "0-run"
    assert (log_dir / "baseline_summary.json").exists()
    assert (log_dir / "research_summary.json").exists()
    assert (log_dir / "ablation_summary.json").exists()
