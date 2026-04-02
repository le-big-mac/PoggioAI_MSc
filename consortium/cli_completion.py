"""
CLI completion helper — drop-in replacement for litellm.completion().

Used for single-turn reasoning tasks (debate, synthesis, evaluation, scoring)
where a full agentic loop isn't needed — just prompt in, text out.

Supports session resume for multi-turn conversations (persona council debate).
For multi-turn agentic tasks with tool use, use cli_agent.py instead.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


def cli_completion(
    prompt: str,
    system_prompt: str = "",
    backend: str = "claude",
    model: Optional[str] = None,
    timeout: int = 300,
    session_id: Optional[str] = None,
    resume: bool = False,
    allow_web: bool = False,
) -> str:
    """LLM completion via CLI tool subprocess.

    For pure-reasoning tasks — prompt in, text out. Optionally resumes a
    prior session for multi-turn conversations.

    Args:
        prompt: The user/task prompt.
        system_prompt: Optional system context prepended to the prompt.
        backend: CLI backend — "claude", "codex", or "gemini".
        model: Optional model override.
        timeout: Max seconds for the subprocess.
        session_id: Session UUID for multi-turn. Required if resume=True.
        resume: If True, resume the session instead of starting a new one.
        allow_web: If True, enable WebFetch/WebSearch tools.

    Returns:
        Response text. On error/timeout, returns a descriptive error string.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    t0 = time.time()
    try:
        if backend == "claude":
            cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "5"]
            if model:
                cmd.extend(["--model", model])
            tools = "WebFetch,WebSearch" if allow_web else ""
            cmd.extend(["--allowedTools", tools])
            if resume and session_id:
                cmd.extend(["--resume", session_id])
            elif session_id:
                cmd.extend(["--session-id", session_id])

        elif backend == "codex":
            if resume and session_id:
                cmd = ["codex", "exec", "resume", session_id,
                       "--dangerously-bypass-approvals-and-sandbox"]
            else:
                cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
            if model:
                cmd.extend(["-m", model])

        elif backend == "gemini":
            cmd = ["gemini", "--approval-mode", "yolo"]
            if model:
                cmd.extend(["--model", model])
            if resume and session_id:
                cmd.extend(["--resume", session_id])

        else:
            return f"[cli_completion error: unknown backend {backend!r}]"

        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True,
            timeout=timeout, env=os.environ.copy(),
        )

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        msg = f"[cli_completion timed out after {elapsed:.0f}s — backend={backend}, model={model}]"
        logger.warning(msg)
        return msg
    except FileNotFoundError:
        msg = f"[cli_completion error: '{backend}' CLI tool not found on PATH]"
        logger.error(msg)
        return msg

    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[:500]
        msg = f"[cli_completion error (rc={result.returncode}): {stderr_snippet}]"
        logger.warning(msg)
        return msg

    output = result.stdout.strip()

    # Record invocation for budget tracking
    try:
        from .cli_budget import get_global_cli_tracker
        tracker = get_global_cli_tracker()
        if tracker is not None:
            tracker.record_invocation(
                agent_name="cli_completion",
                backend=backend,
                model=model or "default",
                duration_seconds=elapsed,
                prompt_chars=len(full_prompt),
                output_chars=len(output),
            )
    except Exception:
        pass

    logger.debug(
        "[cli_completion] backend=%s model=%s elapsed=%.1fs output_len=%d",
        backend, model, elapsed, len(output),
    )
    return output


def extract_session_id(stdout: str) -> Optional[str]:
    """Extract session ID from codex CLI output."""
    for line in stdout.splitlines():
        if "session id:" in line.lower():
            return line.split(":")[-1].strip()
    return None
