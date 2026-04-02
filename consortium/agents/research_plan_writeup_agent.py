"""
ResearchPlanWriteupAgent — LangGraph node module.

Reads research_goals.json and track_decomposition.json produced by
formalize_goals_agent and renders them into research_plan.tex + .pdf.
Separated from formalize_goals_agent so LaTeX failures cannot block
goal formalization.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.research_plan_writeup_instructions import get_research_plan_writeup_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = []
    system_prompt = get_research_plan_writeup_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "research_plan_writeup_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="research_plan_writeup_agent",
        workspace_dir=workspace_dir,
        persist_session=True,
        mandatory_artifacts=[
            "paper_workspace/research_plan.tex",
        ],
    )
