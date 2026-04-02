"""
Tests for consortium/runner.py in CLI-agent mode.
"""

import json
from datetime import datetime
from unittest.mock import patch


class TestValidateCliTools:
    def test_no_error_when_binary_exists(self):
        from consortium.runner import _validate_cli_tools

        with patch("consortium.runner.subprocess.run") as run:
            run.return_value = None
            assert _validate_cli_tools("claude") == []

    def test_error_when_binary_missing(self):
        from consortium.runner import _validate_cli_tools

        with patch("consortium.runner.subprocess.run", side_effect=FileNotFoundError):
            errors = _validate_cli_tools("codex")
        assert len(errors) == 1
        assert "codex" in errors[0]


class TestCollectRequiredCliBackends:
    def test_collects_default_and_overrides(self):
        from consortium.runner import _collect_required_cli_backends
        from consortium.utils import CLIBackendRegistry, CLIBackendSpec

        registry = CLIBackendRegistry(
            default=CLIBackendSpec(backend="claude", model="claude-opus-4-6"),
            agent_overrides={
                "math_prover_agent": CLIBackendSpec(backend="codex", model="gpt-5.4"),
                "reviewer_agent": CLIBackendSpec(backend="gemini", model="gemini-2.5-pro"),
            },
        )

        assert _collect_required_cli_backends(registry) == ["claude", "codex", "gemini"]


class TestListRuns:
    def test_no_results_dir(self, tmp_path, monkeypatch, capsys):
        from consortium.runner import _list_runs
        monkeypatch.chdir(tmp_path)
        _list_runs(str(tmp_path / "nonexistent"))
        out = capsys.readouterr().out
        assert "No results" in out

    def test_empty_results_dir(self, tmp_path, capsys):
        from consortium.runner import _list_runs
        (tmp_path / "results").mkdir()
        _list_runs(str(tmp_path / "results"))
        out = capsys.readouterr().out
        assert "No past runs" in out

    def test_lists_workspace_with_cli_budget_state(self, tmp_path, capsys):
        from consortium.runner import _list_runs

        ws = tmp_path / "consortium_20260101_120000"
        ws.mkdir()
        (ws / "cli_budget_state.json").write_text(json.dumps({"total_seconds": 91}))
        (ws / "STATUS.txt").write_text("COMPLETE")
        (ws / "run_summary.json").write_text(json.dumps({"task": "Test research task"}))

        _list_runs(str(tmp_path))
        out = capsys.readouterr().out
        assert "consortium_20260101_120000" in out
        assert "91s" in out
        assert "COMPLETE" in out


class TestWriteExperimentMetadata:
    def test_writes_metadata_json(self, tmp_path):
        from consortium.runner import _write_experiment_metadata

        class FakeArgs:
            enable_math_agents = False
            output_format = "latex"
            enforce_paper_artifacts = False
            min_review_score = 8

        _write_experiment_metadata(
            str(tmp_path),
            FakeArgs(),
            "Test task",
            "claude",
            "claude-opus-4-6",
        )
        meta_path = tmp_path / "experiment_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["mode"] == "cli_agent"
        assert meta["cli_backend"] == "claude"
        assert meta["cli_model"] == "claude-opus-4-6"
        assert meta["cli_args"]["enable_math_agents"] is False

    def test_task_preview_truncated(self, tmp_path):
        from consortium.runner import _write_experiment_metadata

        class FakeArgs:
            enable_math_agents = False
            output_format = "markdown"
            enforce_paper_artifacts = False
            min_review_score = 8

        _write_experiment_metadata(
            str(tmp_path),
            FakeArgs(),
            "x" * 500,
            "codex",
            "gpt-5.4",
        )
        meta = json.loads((tmp_path / "experiment_metadata.json").read_text())
        assert len(meta["task_preview"]) <= 200


class TestWriteRunSummary:
    def test_writes_summary_json(self, tmp_path):
        from consortium.runner import _write_run_summary

        start = datetime(2026, 1, 1, 12, 0, 0)
        _write_run_summary(
            workspace_dir=str(tmp_path),
            task="Test task",
            cli_backend="claude",
            cli_model="claude-opus-4-6",
            start_time=start,
            stages_completed=["literature_review_agent", "brainstorm_agent"],
        )
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        assert summary["task"] == "Test task"
        assert summary["mode"] == "cli_agent"
        assert summary["cli_backend"] == "claude"
        assert "brainstorm_agent" in summary["stages_completed"]
        assert summary["duration_seconds"] >= 0

    def test_reads_cli_budget_summary(self, tmp_path):
        from consortium.runner import _write_run_summary

        (tmp_path / "cli_budget_state.json").write_text(
            json.dumps({"total_seconds": 120, "max_invocations": 100})
        )
        _write_run_summary(
            str(tmp_path),
            "task",
            "gemini",
            "gemini-2.5-pro",
            datetime.now(),
            [],
        )
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        assert summary["cli_budget"]["total_seconds"] == 120
