"""
ExperimentTranscriptionAgent — LangGraph node module.

Converts verified experiment artifacts to publication-quality LaTeX.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_transcription_instructions import get_experiment_transcription_system_prompt


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = []
    system_prompt = get_experiment_transcription_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experiment_transcription_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experiment_transcription_agent",
        workspace_dir=workspace_dir,
        mandatory_artifacts=[
            "paper_workspace/experiment_report.tex",
        ],
    )
