"""
BrainstormAgent — LangGraph node module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.brainstorm_instructions import get_brainstorm_system_prompt
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        PaperSearchTool(),
        FetchArxivPapersTool(working_dir=workspace_dir),
        LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
    ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_brainstorm_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "brainstorm_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="brainstorm_agent",
        workspace_dir=workspace_dir,
    )
