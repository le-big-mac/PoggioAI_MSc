"""
FormalizeResultsAgent — LangGraph node module.

Completion-based: reads all track outputs, maps results back to research goals.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.formalize_results_instructions import get_formalize_results_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    system_prompt = get_formalize_results_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], "formalize_results_agent", workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name="formalize_results_agent",
        workspace_dir=workspace_dir or "",
        input_files=[
            "paper_workspace/research_goals.json",
            "paper_workspace/track_decomposition.json",
            "paper_workspace/theory_sections.tex",
            "paper_workspace/experiment_track_summary.json",
            "paper_workspace/theory_track_summary.json",
            "math_workspace/claim_graph.json",
            "experiment_workspace/experiment_design.json",
            "paper_workspace/brainstorm.json",
        ],
        output_files=[
            "paper_workspace/formalized_results.md",
            "paper_workspace/formalized_results.json",
        ],
        mandatory_artifacts=[
            "paper_workspace/formalized_results.json",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
