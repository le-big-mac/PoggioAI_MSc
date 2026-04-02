import json
import subprocess
from unittest.mock import patch


def test_claim_graph_cli_uses_math_workspace(tmp_path):
    result = subprocess.run(
        [
            "python3",
            "-m",
            "consortium.toolkits.math.claim_graph_cli",
            "--workspace",
            str(tmp_path),
            "--action",
            "init",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert (tmp_path / "math_workspace" / "claim_graph.json").exists()
    assert not (tmp_path / "claim_graph.json").exists()


def test_proof_rigor_cli_reads_proof_from_math_workspace(tmp_path):
    math_ws = tmp_path / "math_workspace"
    proofs_dir = math_ws / "proofs"
    proofs_dir.mkdir(parents=True)

    (math_ws / "claim_graph.json").write_text(
        json.dumps({"claims": [{"id": "T1", "depends_on": [], "assumptions": []}]})
    )
    (proofs_dir / "T1.md").write_text(
        "## Claim\nX\n## Detailed Steps\n1. test\n## Conclusion\nDone"
    )

    result = subprocess.run(
        [
            "python3",
            "-m",
            "consortium.toolkits.math.proof_rigor_cli",
            "--workspace",
            str(tmp_path),
            "--claim-id",
            "T1",
            "--check-level",
            "strict",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["claim_id"] == "T1"
    assert payload["critical_issues"] == ["proof has fewer than four explicit steps"]


def test_adversarial_verifier_nodes_build_with_cli_backend_spec():
    from consortium.agents.experiment_verification_agent import build_node as build_experiment_node
    from consortium.agents.cli_agent import create_cli_agent
    from consortium.agents.math_rigorous_verifier_agent import build_node as build_math_node
    from consortium.utils import CLIBackendSpec

    spec = CLIBackendSpec(backend="claude", model=None, timeout_seconds=1)

    math_node = build_math_node(model=spec, workspace_dir=".", adversarial=True)
    experiment_node = build_experiment_node(model=spec, workspace_dir=".", adversarial=True)

    assert callable(math_node)
    assert callable(experiment_node)


def test_cli_agent_persistent_session_resumes_on_reentry(tmp_path):
    from consortium.agents.cli_agent import create_cli_agent

    calls = []

    def fake_runner(prompt, workspace_dir, model, timeout, allowed_tools=None, session_id=None, resume=False, metadata=None, env=None):
        calls.append({"session_id": session_id, "resume": resume, "cwd": workspace_dir})
        return subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="ok", stderr="")

    with patch("consortium.agents.cli_agent._RUNNERS", {"claude": fake_runner}):
        node = create_cli_agent(
            cli_backend="claude",
            system_prompt="system",
            agent_name="brainstorm_agent",
            workspace_dir=str(tmp_path),
            persist_session=True,
        )

        first = node({"task": "first pass"})
        second = node({"task": "second pass", "_cli_agent_sessions": first["_cli_agent_sessions"]})

    assert calls[0]["resume"] is False
    assert calls[0]["session_id"]
    assert first["_cli_agent_sessions"]["brainstorm_agent"] == calls[0]["session_id"]
    assert calls[1]["resume"] is True
    assert calls[1]["session_id"] == calls[0]["session_id"]
    assert second["_cli_agent_sessions"]["brainstorm_agent"] == calls[0]["session_id"]


def test_cli_agent_without_persistence_does_not_store_session(tmp_path):
    from consortium.agents.cli_agent import create_cli_agent

    def fake_runner(prompt, workspace_dir, model, timeout, allowed_tools=None, session_id=None, resume=False, metadata=None, env=None):
        return subprocess.CompletedProcess(args=["claude"], returncode=0, stdout="ok", stderr="")

    with patch("consortium.agents.cli_agent._RUNNERS", {"claude": fake_runner}):
        node = create_cli_agent(
            cli_backend="claude",
            system_prompt="system",
            agent_name="reviewer_agent",
            workspace_dir=str(tmp_path),
            persist_session=False,
        )
        result = node({"task": "review"})

    assert "_cli_agent_sessions" not in result


def test_run_cli_agent_subprocess_records_budget_and_codex_session(tmp_path):
    from consortium.agents.cli_agent import run_cli_agent_subprocess
    from consortium.cli_budget import CLIBudgetTracker, set_global_cli_tracker

    tracker = CLIBudgetTracker(state_dir=str(tmp_path))
    set_global_cli_tracker(tracker)

    def fake_runner(prompt, workspace_dir, model, timeout, session_id=None, resume=False, metadata=None, env=None):
        return subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout="<FINAL_OUTPUT>done</FINAL_OUTPUT>",
            stderr="Session ID: abc-123\n",
        )

    with patch("consortium.agents.cli_agent._RUNNERS", {"codex": fake_runner}):
        run = run_cli_agent_subprocess(
            cli_backend="codex",
            prompt="test prompt",
            workspace_dir=str(tmp_path),
            agent_name="quick_verdict",
            model="gpt-5.4",
        )

    assert run["output"] == "done"
    assert run["session_id"] == "abc-123"
    assert tracker.summary["invocation_count"] == 1
    assert tracker.summary["by_agent"]["quick_verdict"] >= 0
    set_global_cli_tracker(None)
