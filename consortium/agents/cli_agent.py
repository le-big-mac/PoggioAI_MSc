"""
CLI Agent wrapper — runs local CLI agents (Claude Code, Codex, Gemini CLI)
as subprocess replacements for the LangChain ReAct agent loop.

Each CLI agent gets the full system prompt + task, runs with its own tool use
in the workspace directory, and returns its stdout as the agent output.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Delimiter used to extract final output from CLI agent stdout.
# Agents are instructed to wrap their final answer in these markers
# so we can separate it from progress/status noise.
_OUTPUT_START = "<FINAL_OUTPUT>"
_OUTPUT_END = "</FINAL_OUTPUT>"


def _base_env(cli_backend: str, model: Optional[str], agent_name: str) -> dict:
    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_root if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    )
    env["CONSORTIUM_REPO_ROOT"] = repo_root
    env["CONSORTIUM_ACTIVE_CLI_BACKEND"] = cli_backend
    if model:
        env["CONSORTIUM_ACTIVE_CLI_MODEL"] = model
    env["CONSORTIUM_ACTIVE_AGENT_NAME"] = agent_name
    return env


def _extract_final_output(raw: str) -> str:
    """Extract content between FINAL_OUTPUT markers, or return full text."""
    start = raw.rfind(_OUTPUT_START)
    end = raw.rfind(_OUTPUT_END)
    if start != -1 and end != -1 and end > start:
        return raw[start + len(_OUTPUT_START):end].strip()
    return raw.strip()


def _strip_react_boilerplate(system_prompt: str) -> str:
    """Extract domain instructions from a ReAct-formatted system prompt.

    The ReAct prompt template wraps the actual instructions inside an
    ``## Agent-Specific Instructions`` section. We extract that section,
    falling back to the full prompt if the markers aren't found.
    """
    # Try to extract the instructions section
    marker = "## Agent-Specific Instructions"
    idx = system_prompt.find(marker)
    if idx != -1:
        instructions = system_prompt[idx + len(marker):]
        # Strip any trailing managed-agents boilerplate
        for end_marker in ["## Managed Agents", "Now Begin!", "You have access to a team"]:
            end_idx = instructions.find(end_marker)
            if end_idx != -1:
                instructions = instructions[:end_idx]
        return instructions.strip()

    # Fallback: try to find content after the tools section
    tools_end = system_prompt.find("Here are the rules you should always follow")
    if tools_end != -1:
        # Skip past the rules to the workspace/instructions sections
        ws_marker = "## Workspace Management"
        ws_idx = system_prompt.find(ws_marker, tools_end)
        if ws_idx != -1:
            return system_prompt[ws_idx:].strip()

    # Last resort: return the full prompt
    return system_prompt


def _build_prompt(system_prompt: str, task: str, workspace_dir: str, agent_name: str) -> str:
    """Compose the full prompt sent to the CLI agent.

    Strips ReAct boilerplate from the system prompt and wraps it in
    CLI-agent-friendly context.
    """
    from ..prompts.system_prompt_template import adapt_prompt_for_cli

    domain_instructions = _strip_react_boilerplate(system_prompt)
    adapted = adapt_prompt_for_cli(domain_instructions, workspace_dir, agent_name)

    return f"""{adapted}

<task>
{task}
</task>

When you have completed your work, output your final summary/analysis.
If you produced files, list the key files you created or modified."""


def _run_claude(prompt: str, workspace_dir: str, model: Optional[str],
                timeout: int, allowed_tools: Optional[List[str]] = None,
                session_id: Optional[str] = None, resume: bool = False,
                metadata: Optional[dict] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run Claude Code CLI in print mode."""
    cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "100"]
    if model:
        cmd.extend(["--model", model])
    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])
    else:
        cmd.extend(["--allowedTools",
                     "Edit,Read,Write,WebFetch,WebSearch,Bash(python*),Bash(curl*),Bash(ls*),Bash(cat*),Bash(grep*),Bash(find*),Bash(cd*),Bash(mkdir*),Bash(cp*),Bash(mv*),Bash(pip*),Bash(tectonic*),Glob,Grep"])
    if resume and session_id:
        cmd.extend(["--resume", session_id])
    elif session_id:
        cmd.extend(["--session-id", session_id])

    return subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        cwd=workspace_dir, timeout=timeout,
        env=env or os.environ.copy(),
    )


def _run_codex(prompt: str, workspace_dir: str, model: Optional[str],
               timeout: int, session_id: Optional[str] = None,
               resume: bool = False, metadata: Optional[dict] = None,
               env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run OpenAI Codex CLI in full-auto mode."""
    if resume and session_id:
        cmd = ["codex", "exec", "resume", session_id, "--dangerously-bypass-approvals-and-sandbox"]
    else:
        cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
    if model:
        cmd.extend(["--model", model])

    return subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        cwd=workspace_dir, timeout=timeout,
        env=env or os.environ.copy(),
    )


def _run_gemini(prompt: str, workspace_dir: str, model: Optional[str],
                timeout: int, session_id: Optional[str] = None,
                resume: bool = False, metadata: Optional[dict] = None,
                env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run Gemini CLI."""
    cmd = ["gemini", "--approval-mode", "yolo"]
    if model:
        cmd.extend(["--model", model])
    if resume:
        cmd.extend(["--resume", "latest"])

    return subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        cwd=workspace_dir, timeout=timeout,
        env=env or os.environ.copy(),
    )


_RUNNERS = {
    "claude": _run_claude,
    "codex": _run_codex,
    "gemini": _run_gemini,
}


def run_cli_agent_subprocess(
    *,
    cli_backend: str,
    prompt: str,
    workspace_dir: str,
    agent_name: str,
    model: Optional[str] = None,
    timeout_seconds: int = 3600,
    allowed_tools: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    resume: bool = False,
    track_budget: bool = True,
) -> dict:
    """Run a CLI backend using the shared agent subprocess plumbing.

    Returns a dict with:
      - ``result``: ``subprocess.CompletedProcess``
      - ``output``: extracted final stdout content
      - ``elapsed``: wall-clock duration in seconds
      - ``session_id``: stable/resolved session ID after the call
    """
    runner = _RUNNERS.get(cli_backend)
    if runner is None:
        raise ValueError(
            f"Unknown CLI backend: {cli_backend!r}. "
            f"Supported: {list(_RUNNERS.keys())}"
        )

    meta: dict = {}
    env = _base_env(cli_backend, model, agent_name)
    t0 = time.time()

    if cli_backend == "claude":
        result = runner(
            prompt, workspace_dir, model, timeout_seconds,
            allowed_tools, session_id=session_id, resume=resume, metadata=meta, env=env,
        )
    else:
        result = runner(
            prompt, workspace_dir, model, timeout_seconds,
            session_id=session_id, resume=resume, metadata=meta, env=env,
        )

    elapsed = time.time() - t0
    resolved_session_id = session_id

    if cli_backend == "codex" and result.stderr:
        from ..cli_completion import extract_session_id
        real_session = extract_session_id(result.stderr)
        if real_session:
            resolved_session_id = real_session

    output = _extract_final_output(result.stdout or "")

    if track_budget:
        try:
            from ..cli_budget import get_global_cli_tracker
            tracker = get_global_cli_tracker()
            if tracker is not None:
                tracker.record_invocation(
                    agent_name=agent_name,
                    backend=cli_backend,
                    model=model or "default",
                    resumed=resume,
                    duration_seconds=elapsed,
                    prompt_chars=len(prompt),
                    output_chars=len(output),
                )
        except Exception:
            pass  # Never break the pipeline for tracking errors

    return {
        "result": result,
        "output": output,
        "elapsed": elapsed,
        "session_id": resolved_session_id,
    }


def create_cli_agent(
    cli_backend: str,
    system_prompt: str,
    agent_name: str,
    workspace_dir: str,
    model: Optional[str] = None,
    timeout_seconds: int = 3600,
    allowed_tools: Optional[List[str]] = None,
    mandatory_artifacts: Optional[List[str]] = None,
    persist_session: bool = False,
) -> Callable:
    """
    Build a LangGraph node that runs a CLI agent as a subprocess.

    Drop-in replacement for ``create_specialist_agent`` — same return contract.
    The CLI agent runs with full tool use in the workspace directory.

    Args:
        cli_backend: One of "claude", "codex", "gemini".
        system_prompt: The full system prompt for this specialist.
        agent_name: Name of this agent node (used in state updates).
        workspace_dir: Path to the research workspace.
        model: Optional model override (e.g. "claude-sonnet-4-6").
        timeout_seconds: Max wall-clock time for the subprocess.
        allowed_tools: Optional list of allowed tools (Claude Code only).

    Returns:
        A callable ``node_fn(state) -> state_update`` for LangGraph.
    """
    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")
        prompt = _build_prompt(system_prompt, task, workspace_dir, agent_name)
        sessions = dict(state.get("_cli_agent_sessions") or {})
        supports_resume = cli_backend in {"claude", "codex"}
        use_session = persist_session and supports_resume
        session_id = sessions.get(agent_name)
        is_resume = bool(session_id)
        if use_session and cli_backend == "claude" and session_id is None:
            session_id = str(uuid.uuid4())
            sessions[agent_name] = session_id

        logger.info(
            "[CLI Agent] %s starting — backend=%s model=%s cwd=%s resume=%s",
            agent_name, cli_backend, model, workspace_dir, is_resume,
        )

        try:
            run = run_cli_agent_subprocess(
                cli_backend=cli_backend,
                prompt=prompt,
                workspace_dir=workspace_dir,
                agent_name=agent_name,
                model=model,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools,
                session_id=session_id,
                resume=is_resume,
            )
        except subprocess.TimeoutExpired:
            output = (
                f"[{agent_name}] CLI agent timed out after {timeout_seconds}s. "
                f"Backend: {cli_backend}, model: {model}"
            )
            logger.error(output)
            result_state = {
                "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
                "agent_task": None,
            }
            if use_session:
                result_state["_cli_agent_sessions"] = sessions
            return result_state

        result = run["result"]
        elapsed = run["elapsed"]
        output = run["output"]

        if use_session and run["session_id"]:
            sessions[agent_name] = run["session_id"]

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:2000]
            stdout_snippet = (result.stdout or "")[:2000]
            output = (
                f"[{agent_name}] CLI agent exited with code {result.returncode}.\n"
                f"stderr: {stderr_snippet}\n"
                f"stdout: {stdout_snippet}"
            )
            logger.error(output)
        else:
            output = _extract_final_output(result.stdout or "")
            logger.info(
                "[CLI Agent] %s completed in %.1fs — output length: %d chars",
                agent_name, elapsed, len(output),
            )

        # Check mandatory artifacts after agent completes
        if mandatory_artifacts:
            missing = [
                a for a in mandatory_artifacts
                if not os.path.isfile(os.path.join(workspace_dir, a))
            ]
            if missing:
                raise RuntimeError(
                    f"[{agent_name}] Missing mandatory artifacts: {missing}"
                )

        result_state = {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }
        if use_session:
            result_state["_cli_agent_sessions"] = sessions
        return result_state

    node_fn.__name__ = agent_name
    return node_fn
