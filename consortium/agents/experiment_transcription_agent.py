"""
ExperimentTranscriptionAgent — LangGraph node module.

Converts verified experiment artifacts to publication-quality LaTeX.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_transcription_instructions import get_experiment_transcription_system_prompt
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
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
    system_prompt = get_experiment_transcription_system_prompt(tools=tools, managed_agents=None)
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
    )
