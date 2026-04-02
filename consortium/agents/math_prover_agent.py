"""
MathProverAgent — LangGraph node module.

Completion-based: reads claim graph + lemma library, outputs proof drafts.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.math_prover_instructions import get_math_prover_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_math_prover_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "math_prover_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="math_prover_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "math_workspace/claim_graph.json",
            "math_workspace/lemma_library.md",
            "math_workspace/lemma_library_index.json",
            "paper_workspace/research_goals.json",
        ],
        output_files=[
            "math_workspace/proofs/proofs.md",
        ],
        mandatory_artifacts=[
            "math_workspace/proofs/proofs.md",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
