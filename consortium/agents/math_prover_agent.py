"""
MathProverAgent — LangGraph node module.

Full CLI agent: drafts proofs iteratively, reading the claim graph to
determine what needs proving, writing proof files, and updating claim
status via MathClaimGraphTool. Needs tool access for multi-turn
proof refinement.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.math_prover_instructions import get_math_prover_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = []
    system_prompt = get_math_prover_system_prompt(tools=[], managed_agents=None)
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
        mandatory_artifacts=[
            "math_workspace/claim_graph.json",
        ],
    )
