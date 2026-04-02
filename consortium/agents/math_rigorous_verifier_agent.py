"""
MathRigorousVerifierAgent — LangGraph node module.

Completion-based: reads proofs + claim graph, outputs rigor audit.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..prompts.math_rigorous_verifier_instructions import get_math_rigorous_verifier_system_prompt


ADVERSARIAL_SYSTEM_PROMPT_PREFIX = """Your agent_name is "math_rigorous_verifier_agent" (ADVERSARIAL MODE).

ROLE
You are a HOSTILE mathematical reviewer whose sole objective is to BREAK this proof.
Find every gap, every unjustified step, every implicit assumption. Assume NOTHING is
correct until proven rigorously.

ADVERSARIAL RULES
- Your goal is to INVALIDATE the proof. You succeed when you find a genuine error.
- Rate each issue: CRITICAL (proof is invalid), MAJOR (significant gap), MINOR (cosmetic).
- The proof passes your review ONLY if you genuinely cannot find a way to invalidate it.
- Do NOT suggest fixes — only identify problems.
- Do NOT give the benefit of the doubt. If a step is not fully justified, flag it.
- Look specifically for: missing quantifier scoping, unjustified limit exchanges,
  hidden regularity assumptions, circular reasoning, appeal to intuition.
- Check boundary cases and degenerate inputs.
- Verify that all cited lemmas/theorems are applicable (correct hypotheses satisfied).

OUTPUT STANDARD (ADVERSARIAL MODE)
For each claim audited, produce a structured verdict with:
- verdict: "invalidated" or "survived"
- critical_issues: list of {location, description, why_invalid}
- major_issues: list
- recommendation: "return_to_prover" | "proceed_with_caution" | "reject_claim"

STATUS RULES (ADVERSARIAL MODE):
- verdict=invalidated: set status to proved_draft (block verified_symbolic).
- verdict=survived: adversarial audit satisfied — standard verification may proceed.
"""


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    adversarial: bool = False,
    **cfg: Any,
) -> Callable:
    if adversarial:
        from ..prompts.system_prompt_template import build_system_prompt
        system_prompt = build_system_prompt(
            tools=[],
            instructions=ADVERSARIAL_SYSTEM_PROMPT_PREFIX,
            managed_agents=None,
        )
    else:
        system_prompt = get_math_rigorous_verifier_system_prompt(tools=[], managed_agents=None)

    agent_name = "math_rigorous_verifier_agent"
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, [], agent_name, workspace_dir, counsel_models)

    from .completion_node import create_completion_node
    return create_completion_node(
        system_prompt=system_prompt,
        agent_name=agent_name,
        workspace_dir=workspace_dir or "",
        input_files=[
            "math_workspace/claim_graph.json",
            "math_workspace/proofs/proofs.md",
            "paper_workspace/research_goals.json",
        ],
        output_files=[
            "math_workspace/verification_audit.md",
            "math_workspace/claim_graph.json",
        ],
        mandatory_artifacts=[
            "math_workspace/verification_audit.md",
        ],
        backend=model.backend,
        model=model.model,
        timeout=model.timeout_seconds,
    )
