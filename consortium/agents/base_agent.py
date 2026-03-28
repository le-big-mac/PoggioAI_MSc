"""
Base agent factory — CLI agent mode.

Routes all specialist agent nodes to local CLI agents (Claude Code, Codex,
Gemini CLI) which run as subprocesses with their own tool-use loops.

Each specialist agent module exposes:
  - get_tools(workspace_dir, model_id)  -> list[BaseTool]
  - build_node(model, workspace_dir, authorized_imports, **cfg) -> Callable
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..models import get_context_limit  # noqa: F401 — re-exported for backward compat


def create_specialist_agent(
    model: Any,
    tools: List[Any],
    system_prompt: str,
    agent_name: str,
    workspace_dir: Optional[str] = None,
) -> Callable:
    """
    Build a CLI agent node for a specialist.

    The agent runs as a local CLI subprocess (Claude Code / Codex / Gemini CLI)
    with its own tool-use loop.  The *tools* parameter is accepted for
    signature compatibility with agent modules but is not used — CLI agents
    discover and manage their own tools.

    The returned callable accepts a ResearchState dict and returns a state
    update dict suitable for use as a LangGraph node.
    """
    from ..utils import CLIBackendSpec
    from .cli_agent import create_cli_agent

    if not isinstance(model, CLIBackendSpec):
        raise TypeError(
            f"CLI-only branch: expected CLIBackendSpec, got {type(model).__name__}. "
            f"Ensure cli_mode is configured in .llm_config.yaml."
        )

    return create_cli_agent(
        cli_backend=model.backend,
        system_prompt=system_prompt,
        agent_name=agent_name,
        workspace_dir=workspace_dir or "",
        model=model.model,
        timeout_seconds=model.timeout_seconds,
    )
