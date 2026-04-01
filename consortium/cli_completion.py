"""
CLI completion helper — drop-in replacement for litellm.completion().

Used for single-turn reasoning tasks (debate, synthesis, evaluation, scoring)
where a full agentic loop isn't needed — just prompt in, text out, no tool use.

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
) -> str:
    """Single-turn LLM completion via CLI tool subprocess.

    Replaces ``litellm.completion()`` for pure-reasoning tasks. Sends a prompt
    to the CLI tool and returns the response text. No tool use is enabled —
    the CLI agent just thinks and responds.

    Args:
        prompt: The user/task prompt.
        system_prompt: Optional system context prepended to the prompt.
        backend: CLI backend — "claude", "codex", or "gemini".
        model: Optional model override (e.g. "claude-sonnet-4-6").
        timeout: Max seconds for the subprocess.

    Returns:
        Response text. On error/timeout, returns a descriptive error string
        (never raises — matches existing fallback patterns in counsel/persona).
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    t0 = time.time()
    try:
        if backend == "claude":
            cmd = ["claude", "-p", "--output-format", "text"]
            if model:
                cmd.extend(["--model", model])
            cmd.extend(["--allowedTools", "WebFetch,WebSearch,Read"])
            result = subprocess.run(
                cmd, input=full_prompt, capture_output=True, text=True,
                timeout=timeout, env=os.environ.copy(),
            )
        elif backend == "codex":
            cmd = ["codex", "--approval-mode", "full-auto", "--quiet"]
            if model:
                cmd.extend(["--model", model])
            result = subprocess.run(
                cmd, input=full_prompt, capture_output=True, text=True,
                timeout=timeout, env=os.environ.copy(),
            )
        elif backend == "gemini":
            cmd = ["gemini"]
            if model:
                cmd.extend(["--model", model])
            result = subprocess.run(
                cmd, input=full_prompt, capture_output=True, text=True,
                timeout=timeout, env=os.environ.copy(),
            )
        else:
            return f"[cli_completion error: unknown backend {backend!r}]"

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
        pass  # Never break callers for tracking errors

    logger.debug(
        "[cli_completion] backend=%s model=%s elapsed=%.1fs output_len=%d",
        backend, model, elapsed, len(output),
    )
    return output
