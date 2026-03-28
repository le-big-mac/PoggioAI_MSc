"""LLM-based proof strategy generation for tree search branching.

Given a claim from the claim graph and the current proof/verification state,
generates N candidate proof strategies that the tree controller can branch on.
Each strategy becomes a PROOF_STRATEGY tree node executed by the math_prover
agent with a specific directive.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from ..cli_completion import cli_completion


@dataclass
class ProofStrategy:
    """A candidate proof approach for a claim."""

    name: str
    description: str
    prompt_directive: str  # injected into math_prover_agent's task
    estimated_difficulty: str  # easy | medium | hard
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt_directive": self.prompt_directive,
            "estimated_difficulty": self.estimated_difficulty,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Proof strategy generation
# ---------------------------------------------------------------------------

_PROOF_STRATEGY_SYSTEM = """\
You are a mathematical research strategist.  Given a claim from a formal claim
graph (with its statement, assumptions, dependencies, and any prior proof
attempt / verification gaps), produce exactly {n} distinct proof strategies.

Each strategy must be genuinely different in approach — not just cosmetic
rephrasing.  Good dimensions of variation include:

- Direct proof vs proof by contradiction vs proof by contrapositive
- Different decompositions into sub-lemmas
- Different key inequalities or theorems applied (e.g. Lyapunov vs LaSalle vs Gronwall)
- Bypassing a problematic dependency entirely via an alternative route
- Weakening or strengthening assumptions to make the proof tractable
- Reformulating the claim in an equivalent but more tractable form

Respond with a JSON array of exactly {n} objects, each with fields:
  name            — short slug (e.g. "direct_via_lyapunov")
  description     — 2-3 sentence summary of the approach
  prompt_directive — a precise instruction paragraph that will be given to a
                     math_prover agent so it knows exactly what strategy to
                     follow.  Include which lemmas/theorems to apply, which
                     intermediate steps to target, and what to avoid.
  estimated_difficulty — one of "easy", "medium", "hard"
  rationale       — why this approach might succeed where others fail

Return ONLY the JSON array, no markdown fences or commentary.
"""


def generate_proof_strategies(
    claim: dict[str, Any],
    *,
    n: int = 3,
    prior_proof: Optional[str] = None,
    verification_gaps: Optional[list[dict]] = None,
    claim_graph_context: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    failure_memory: Optional[Any] = None,
) -> list[ProofStrategy]:
    """Ask an LLM to propose *n* distinct proof strategies for *claim*.

    Parameters
    ----------
    claim : dict
        A claim dict from claim_graph.json (id, statement, assumptions, etc.).
    n : int
        Number of strategies to generate.
    prior_proof : str, optional
        Text of a previous proof attempt (e.g. from proofs/<claim_id>.md).
    verification_gaps : list[dict], optional
        Gap descriptions from the rigorous verifier's check logs.
    claim_graph_context : str, optional
        Summary of the broader claim graph for context.
    model : str
        LLM model ID for the strategy generation call.
    failure_memory : FailureMemory, optional
        Cross-branch failure records.  When provided, failed strategies and
        common failure patterns are included in the prompt so the LLM avoids
        repeating them.
    """
    user_parts = [f"## Claim\n```json\n{json.dumps(claim, indent=2)}\n```"]

    if prior_proof:
        user_parts.append(f"## Prior Proof Attempt\n{prior_proof}")
    if verification_gaps:
        user_parts.append(
            f"## Verification Gaps\n```json\n{json.dumps(verification_gaps, indent=2)}\n```"
        )
    if claim_graph_context:
        user_parts.append(f"## Claim Graph Context\n{claim_graph_context}")

    # Inject failure memory so the LLM avoids repeating failed strategies
    if failure_memory is not None:
        failure_text = failure_memory.format_for_strategy_prompt(claim.get("id", ""))
        if failure_text:
            user_parts.append(f"## FAILED STRATEGIES — DO NOT REPEAT\n{failure_text}")

    user_msg = "\n\n".join(user_parts)

    raw = cli_completion(
        user_msg,
        system_prompt=_PROOF_STRATEGY_SYSTEM.format(n=n),
        backend="claude",
    ).strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    strategies_data = json.loads(raw)
    return [
        ProofStrategy(
            name=s["name"],
            description=s["description"],
            prompt_directive=s["prompt_directive"],
            estimated_difficulty=s.get("estimated_difficulty", "medium"),
            rationale=s.get("rationale", ""),
        )
        for s in strategies_data[:n]
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prior_proof(workspace_dir: str, claim_id: str) -> Optional[str]:
    """Read the existing proof draft for *claim_id*, if any."""
    path = os.path.join(workspace_dir, "math_workspace", "proofs", f"{claim_id}.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def load_verification_gaps(workspace_dir: str, claim_id: str) -> list[dict]:
    """Read verifier check logs for *claim_id* and extract gap entries."""
    path = os.path.join(workspace_dir, "math_workspace", "checks", f"{claim_id}.jsonl")
    if not os.path.exists(path):
        return []
    gaps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("severity") in ("high", "critical") or entry.get("has_gap"):
                    gaps.append(entry)
            except json.JSONDecodeError:
                continue
    return gaps
