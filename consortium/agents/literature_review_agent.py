"""
LiteratureReviewAgent — LangGraph node module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.literature_review_instructions import get_literature_review_system_prompt
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool

try:
    from ..toolkits.search.open_deep_search.ods_tool import OpenDeepSearchTool
except (ImportError, ModuleNotFoundError):
    OpenDeepSearchTool = None


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        PaperSearchTool(),
        FetchArxivPapersTool(working_dir=workspace_dir),
        CitationSearchTool(),
        LaTeXCompilerTool(model=model_id, working_dir=workspace_dir),
        LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
    ]
    if OpenDeepSearchTool is not None:
        tools.insert(2, OpenDeepSearchTool(model_name=model_id))
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
    system_prompt = get_literature_review_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "literature_review_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="literature_review_agent",
        workspace_dir=workspace_dir,
    )
