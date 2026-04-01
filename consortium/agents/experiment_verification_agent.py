"""
ExperimentVerificationAgent — LangGraph node module.

Audits experimental outputs for statistical and reporting sanity.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_verification_instructions import get_experiment_verification_system_prompt


ADVERSARIAL_EXPERIMENT_PROMPT_PREFIX = """Your agent_name is "experiment_verification_agent" (ADVERSARIAL MODE).

ROLE
You are a SKEPTICAL experimental reviewer. Your goal is to find every possible flaw
in this experimental work. Challenge the statistical significance, identify confounders,
and question the experimental design.

ADVERSARIAL RULES
- Could the results be explained by a simpler hypothesis or a bug in the code?
- Are the controls adequate? Is there selection bias?
- Would a hostile Nature/NeurIPS reviewer accept this?
- Rate each issue: CRITICAL (results unreliable), MAJOR (significant concern), MINOR (cosmetic).
- Check for: p-hacking, multiple comparisons without correction, cherry-picked results,
  inadequate baselines, leaky train/test splits, unreported hyperparameter tuning.
- Verify that error bars, confidence intervals, and significance tests are appropriate.
- Do NOT suggest fixes — only identify problems.
- The experiment passes your review ONLY if you genuinely cannot find a way to
  invalidate the conclusions.

TOOLS: Use the same verification tools to inspect experimental outputs, run
statistical checks, and examine figures.

"""


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    adversarial: bool = False,
    **cfg: Any,
) -> Callable:
    tools = []
    if adversarial:
        from ..prompts.system_prompt_template import build_system_prompt
        system_prompt = build_system_prompt(
            tools=[],
            instructions=ADVERSARIAL_EXPERIMENT_PROMPT_PREFIX,
            managed_agents=None,
        )
    else:
        system_prompt = get_experiment_verification_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    agent_name = "experiment_verification_agent"
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, agent_name, workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name=agent_name,
        workspace_dir=workspace_dir,
    )
