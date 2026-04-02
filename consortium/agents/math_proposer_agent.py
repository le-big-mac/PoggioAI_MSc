"""
MathProposerAgent — LangGraph node module.

Full CLI agent: builds structured claim graphs iteratively using
MathClaimGraphTool (add_claim, add_dependency, set_status, validate_graph).
Needs tool access to construct the graph incrementally.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.math_proposer_instructions import get_math_proposer_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = []
    system_prompt = get_math_proposer_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "math_proposer_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="math_proposer_agent",
        workspace_dir=workspace_dir,
        persist_session=True,
        mandatory_artifacts=[
            "math_workspace/claim_graph.json",
        ],
    )
