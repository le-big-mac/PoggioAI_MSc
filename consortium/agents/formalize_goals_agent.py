"""
FormalizeGoalsAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.formalize_goals_instructions import get_formalize_goals_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = []
    system_prompt = get_formalize_goals_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "formalize_goals_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="formalize_goals_agent",
        workspace_dir=workspace_dir,
        mandatory_artifacts=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
        ],
    )
