import importlib.util
import sys
from pathlib import Path


_WATCHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "idea_watcher.py"
_SPEC = importlib.util.spec_from_file_location("idea_watcher_test_module", _WATCHER_PATH)
idea_watcher = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.path.insert(0, str(_WATCHER_PATH.parent))
sys.modules.setdefault("idea_watcher_test_module", idea_watcher)
_SPEC.loader.exec_module(idea_watcher)


def test_load_seen_backfills_workspaces(tmp_path):
    state_path = tmp_path / "watcher_state.json"
    state_path.write_text('{"issues": [1], "comments": {"1": 2}}')

    seen = idea_watcher._load_seen(str(state_path))

    assert seen["issues"] == [1]
    assert seen["comments"] == {"1": 2}
    assert seen["workspaces"] == {}


def test_find_prior_plan_artifact_prefers_issue_workspace_outputs(tmp_path):
    workspace = tmp_path / "run_1"
    workspace.mkdir()
    (workspace / "final_paper.tex").write_text("\\documentclass{article}")
    paper_ws = workspace / "paper_workspace"
    paper_ws.mkdir()
    (paper_ws / "research_plan.md").write_text("fallback plan")

    prior_plan = idea_watcher._find_prior_plan_artifact(str(workspace))

    assert prior_plan == str(workspace / "final_paper.tex")


def test_handle_command_plan_uses_issue_local_workspace(tmp_path, monkeypatch):
    state_path = tmp_path / "watcher_state.json"
    prior_workspace = tmp_path / "issue_7_run"
    prior_workspace.mkdir()
    prior_plan = prior_workspace / "final_paper.tex"
    prior_plan.write_text("prior plan")
    idea_watcher._save_seen(
        str(state_path),
        {"issues": [7], "comments": {}, "workspaces": {"7": str(prior_workspace)}},
    )

    captured = {}

    def fake_build_task_with_feedback(idea, feedback, prior_plan_path=None):
        captured["prior_plan_path"] = prior_plan_path
        return "task"

    monkeypatch.setattr(idea_watcher, "build_task_with_feedback", fake_build_task_with_feedback)
    monkeypatch.setattr(idea_watcher, "_pipeline_args_for_command", lambda command, modifiers: ["--quick-pass"])
    monkeypatch.setattr(idea_watcher, "launch_pipeline", lambda task, args: (0, str(tmp_path / "new_run")))
    monkeypatch.setattr(idea_watcher, "add_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "add_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "remove_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "_publish_and_comment", lambda *args, **kwargs: None)

    issue = {"number": 7, "title": "Idea title", "body": "Body"}
    idea_watcher.handle_command("owner/repo", issue, "plan", [], "feedback", str(state_path))

    assert captured["prior_plan_path"] == str(prior_plan)


def test_handle_new_issue_records_workspace_on_success(tmp_path, monkeypatch):
    state_path = tmp_path / "watcher_state.json"
    idea_watcher._save_seen(str(state_path), {"issues": [9], "comments": {}, "workspaces": {}})
    workspace = tmp_path / "workspace_9"

    monkeypatch.setattr(idea_watcher, "launch_pipeline", lambda task, args: (0, str(workspace)))
    monkeypatch.setattr(idea_watcher, "add_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "remove_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "add_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(idea_watcher, "_publish_and_comment", lambda *args, **kwargs: None)

    issue = {"number": 9, "title": "New idea", "body": ""}
    idea_watcher.handle_new_issue("owner/repo", issue, str(state_path))

    seen = idea_watcher._load_seen(str(state_path))
    assert seen["workspaces"]["9"] == str(workspace)


def test_build_launch_args_resume_full_run_from_issue_workspace():
    args = idea_watcher._build_launch_args("run", ["counsel"], "/tmp/existing_workspace")

    assert args == [
        "--enable-math-agents",
        "--enable-counsel",
        "--resume",
        "/tmp/existing_workspace",
        "--start-from-stage",
        "math_literature_agent",
    ]


def test_build_launch_args_resume_theory_run_enables_math():
    args = idea_watcher._build_launch_args("theory", [], "/tmp/existing_workspace")

    assert args == [
        "--enable-math-agents",
        "--execution-scope",
        "theory",
        "--resume",
        "/tmp/existing_workspace",
        "--start-from-stage",
        "math_literature_agent",
    ]


def test_build_launch_args_falls_back_to_fresh_run_without_workspace():
    args = idea_watcher._build_launch_args("run", [], None)

    assert args == ["--enable-math-agents"]


def test_build_launch_args_resume_experiment_run_stays_non_math():
    args = idea_watcher._build_launch_args("experiment", [], "/tmp/existing_workspace")

    assert args == [
        "--execution-scope",
        "experiment",
        "--resume",
        "/tmp/existing_workspace",
        "--start-from-stage",
        "experiment_literature_agent",
    ]
