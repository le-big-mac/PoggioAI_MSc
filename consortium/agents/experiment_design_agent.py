"""
ExperimentDesignAgent — LangGraph node module.

Transforms empirical questions into executable experiment specs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_design_instructions import get_experiment_design_system_prompt
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool


def get_tools(
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
) -> list:
    tools = [
        PaperSearchTool(),
        CitationSearchTool(),
    ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = get_tools(workspace_dir, authorized_imports=authorized_imports)
    system_prompt = get_experiment_design_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experiment_design_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experiment_design_agent",
        workspace_dir=workspace_dir,
    )
