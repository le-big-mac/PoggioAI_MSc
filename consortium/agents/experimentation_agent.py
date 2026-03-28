"""
ExperimentationAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experimentation_instructions import get_experimentation_system_prompt
from ..toolkits.experimentation.idea_standardization_tool import IdeaStandardizationTool
from ..toolkits.experimentation.run_experiment_tool import RunExperimentTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        IdeaStandardizationTool(model=model_id),
        RunExperimentTool(workspace_dir=workspace_dir),
        LaTeXCompilerTool(working_dir=workspace_dir, model=model_id),
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
    system_prompt = get_experimentation_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experimentation_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experimentation_agent",
        workspace_dir=workspace_dir,
    )
