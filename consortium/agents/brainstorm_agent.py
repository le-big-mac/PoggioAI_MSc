"""
BrainstormAgent — LangGraph node module.

Completion-based: reads proposal + lit review, outputs structured brainstorm.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.brainstorm_instructions import get_brainstorm_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_brainstorm_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "brainstorm_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="brainstorm_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "paper_workspace/research_proposal.md",
            "paper_workspace/literature_review.tex",
            "paper_workspace/novelty_flags.json",
        ],
        output_files=[
            "paper_workspace/brainstorm.json",
            "paper_workspace/brainstorm.md",
        ],
        mandatory_artifacts=[
            "paper_workspace/brainstorm.json",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
