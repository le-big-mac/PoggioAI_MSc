"""
MathRigorousVerifierAgent — LangGraph node module.

Audits proof rigor and symbolic completeness.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
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

TOOLS: Use the same claim graph and proof workspace tools to read proofs and record
your adversarial audit findings.

OUTPUT STANDARD (ADVERSARIAL MODE)
For each claim audited, append to checks/<claim_id>.jsonl:
{
  "agent": "math_rigorous_verifier_agent_adversarial",
  "check_kind": "adversarial_audit",
  "verdict": "invalidated" | "survived",
  "critical_issues": [
    {"location": "Step N", "description": "...", "why_invalid": "..."}
  ],
  "major_issues": [...],
  "recommendation": "return_to_prover" | "proceed_with_caution" | "reject_claim"
}

STATUS RULES (ADVERSARIAL MODE):
- verdict=invalidated: set status to proved_draft (block verified_symbolic).
  The standard verifier must NOT promote this claim until the prover addresses
  the adversarial findings.
- verdict=survived: adversarial audit satisfied — standard verification may proceed.
  The standard verifier will then run its own checklist independently.

After auditing all claims, write math_workspace/adversarial_audit_summary.md:
## Adversarial Audit Summary
- Claims invalidated: [list]
- Claims survived: [list]
- Most critical issues found: [top 3 with claim_id and one-line description]

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
            instructions=ADVERSARIAL_SYSTEM_PROMPT_PREFIX,
            managed_agents=None,
        )
    else:
        system_prompt = get_math_rigorous_verifier_system_prompt(tools=[], managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    agent_name = "math_rigorous_verifier_agent"
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
