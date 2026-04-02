"""
FormalizeGoalsAgent — LangGraph node module.

Completion-based: reads brainstorm + proposal, outputs research goals and track decomposition.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.formalize_goals_instructions import get_formalize_goals_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_formalize_goals_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "formalize_goals_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="formalize_goals_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "paper_workspace/brainstorm.json",
            "paper_workspace/brainstorm.md",
            "paper_workspace/research_proposal.md",
            "paper_workspace/novelty_flags.json",
            "paper_workspace/literature_review.tex",
        ],
        output_files=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
        ],
        mandatory_artifacts=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
