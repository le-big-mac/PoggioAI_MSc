"""
MathProverAgent — LangGraph node module.

Creates proof drafts for claim-graph items.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.math_prover_instructions import get_math_prover_system_prompt
from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
from ..toolkits.math.proof_workspace_tool import MathProofWorkspaceTool
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool


def get_tools(workspace_dir: Optional[str]) -> list:
    tools = [
        MathClaimGraphTool(working_dir=workspace_dir, allow_accepted_transition=False),
        MathProofWorkspaceTool(working_dir=workspace_dir),
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
    tools = get_tools(workspace_dir)
    system_prompt = get_math_prover_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "math_prover_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="math_prover_agent",
        workspace_dir=workspace_dir,
    )
