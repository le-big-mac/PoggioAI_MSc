"""
FormalizeResultsAgent — LangGraph node module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.formalize_results_instructions import get_formalize_results_system_prompt
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool


def get_tools(workspace_dir: Optional[str], model_id: str, authorized_imports: Optional[List[str]] = None) -> list:
    tools = [
        PaperSearchTool(),
        FetchArxivPapersTool(working_dir=workspace_dir),
        CitationSearchTool(),
        LaTeXCompilerTool(model=model_id, working_dir=workspace_dir),
        LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
        LaTeXContentVerificationTool(working_dir=workspace_dir),
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
    tools = get_tools(workspace_dir, model_id, authorized_imports=authorized_imports)
    system_prompt = get_formalize_results_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "formalize_results_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="formalize_results_agent",
        workspace_dir=workspace_dir,
    )
