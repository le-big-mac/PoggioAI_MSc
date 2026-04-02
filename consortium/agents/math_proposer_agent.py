"""
MathProposerAgent — LangGraph node module.

Completion-based: reads research goals, outputs structured claim graph.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.math_proposer_instructions import get_math_proposer_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_math_proposer_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "math_proposer_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="math_proposer_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
            "paper_workspace/research_proposal.md",
        ],
        output_files=[
            "math_workspace/claim_graph.json",
            "math_workspace/claim_design_notes.md",
        ],
        mandatory_artifacts=[
            "math_workspace/claim_graph.json",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
