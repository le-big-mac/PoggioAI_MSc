"""
ExperimentDesignAgent — LangGraph node module.

Completion-based: reads goals + baselines, outputs experiment design specs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.experiment_design_instructions import get_experiment_design_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_experiment_design_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "experiment_design_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="experiment_design_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
            "paper_workspace/research_proposal.md",
            "experiment_workspace/experiment_baselines.json",
            "experiment_workspace/experiment_literature.md",
        ],
        output_files=[
            "experiment_workspace/experiment_design.json",
            "experiment_workspace/experiment_rationale.md",
        ],
        mandatory_artifacts=[
            "experiment_workspace/experiment_design.json",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
