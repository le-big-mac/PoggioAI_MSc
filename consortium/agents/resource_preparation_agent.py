"""
ResourcePreparationAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.resource_preparation_instructions import get_resource_preparation_system_prompt
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool


def get_tools(workspace_dir: Optional[str], model_id: str, authorized_imports: Optional[List[str]] = None) -> list:
    tools = [
        CitationSearchTool(),
        LaTeXCompilerTool(working_dir=workspace_dir, model=model_id),
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
    system_prompt = get_resource_preparation_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "resource_preparation_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="resource_preparation_agent",
        workspace_dir=workspace_dir,
    )
