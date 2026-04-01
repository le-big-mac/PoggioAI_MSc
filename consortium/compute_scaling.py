"""Adaptive test-time compute scaling for pipeline agents.

Different tasks require different amounts of reasoning compute. Easy lemmas
can use low thinking budgets; hard proofs that fail on first attempt should
retry with escalated compute.

This module provides:
- ``ComputeProfile``: per-provider reasoning parameters
- ``COMPUTE_TIERS``: pre-defined profiles from minimal to extreme
- ``AdaptiveComputeScheduler``: selects tier based on task difficulty signals
- ``apply_compute_profile``: applies a profile to a ChatLiteLLM model instance
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Compute profiles
# ---------------------------------------------------------------------------

@dataclass
class ComputeProfile:
    """Reasoning parameters for a single LLM call, per-provider."""

    claude_effort: str = "high"           # "low" | "medium" | "high" | "max"
    claude_budget_tokens: int = 16384     # 1024 - 131072
    gpt_reasoning_effort: str = "medium"  # "minimal" | "low" | "medium" | "high" | "xhigh"
    gemini_thinking_budget: int = 16384   # 1024 - 131072


COMPUTE_TIERS: dict[str, ComputeProfile] = {
    "minimal":  ComputeProfile("low",    1024,   "minimal",  1024),
    "low":      ComputeProfile("medium", 4096,   "low",      4096),
    "standard": ComputeProfile("high",   16384,  "medium",   16384),
    "high":     ComputeProfile("high",   32768,  "high",     32768),
    "max":      ComputeProfile("max",    65536,  "xhigh",    65536),
    "extreme":  ComputeProfile("max",    131072, "xhigh",    131072),
}

_TIER_ORDER = ["minimal", "low", "standard", "high", "max", "extreme"]


# ---------------------------------------------------------------------------
# Adaptive scheduler
# ---------------------------------------------------------------------------

class AdaptiveComputeScheduler:
    """Selects compute tier based on task difficulty signals.

    Signals that increase compute:
    - Claim has failed before (from failure_memory)
    - Claim has high downstream impact (>2 dependents)
    - Adversarial verifier found issues on prior attempt
    - Tree depth > 1 (debugging branch = prior approach failed)
    - Per-agent override specified in config
    """

    def __init__(
        self,
        default_tier: str = "standard",
        per_agent_overrides: Optional[dict[str, str]] = None,
        escalate_on_failure: bool = True,
    ):
        self.default_tier = default_tier
        self.per_agent_overrides = per_agent_overrides or {}
        self.escalate_on_failure = escalate_on_failure

    def select_tier(
        self,
        agent_name: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        """Return a tier name based on agent identity and difficulty signals.

        Parameters
        ----------
        agent_name : str, optional
            Pipeline agent name (e.g. "math_prover_agent").
        context : dict, optional
            Difficulty signals:
            - ``num_prior_failures`` (int): how many times this claim has failed
            - ``downstream_impact`` (int): number of downstream dependent claims
            - ``depth`` (int): tree search depth (>0 = debugging branch)
            - ``adversarial_issues`` (bool): prior adversarial verifier flagged issues
        """
        ctx = context or {}

        # Start with per-agent override or default
        if agent_name and agent_name in self.per_agent_overrides:
            tier = self.per_agent_overrides[agent_name]
        else:
            tier = self.default_tier

        tier_idx = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 2

        # Escalation signals
        failures = ctx.get("num_prior_failures", 0)
        if failures >= 2:
            tier_idx = max(tier_idx, _TIER_ORDER.index("max"))
        elif failures >= 1:
            tier_idx = max(tier_idx, _TIER_ORDER.index("high"))

        if ctx.get("downstream_impact", 0) > 2:
            tier_idx = max(tier_idx, _TIER_ORDER.index("high"))

        if ctx.get("depth", 0) > 1:
            tier_idx = max(tier_idx, _TIER_ORDER.index("high"))

        if ctx.get("adversarial_issues", False):
            tier_idx = max(tier_idx, _TIER_ORDER.index("max"))

        return _TIER_ORDER[min(tier_idx, len(_TIER_ORDER) - 1)]

    @staticmethod
    def escalate(current_tier: str) -> str:
        """Move to the next higher tier. Returns the same tier if already at max."""
        if current_tier not in _TIER_ORDER:
            return "high"
        idx = _TIER_ORDER.index(current_tier)
        return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


# ---------------------------------------------------------------------------
# Applying profiles to models
# ---------------------------------------------------------------------------

def apply_compute_profile(model: Any, profile: ComputeProfile) -> Any:
    """Apply a ComputeProfile to a ChatLiteLLM model instance.

    Modifies the model's parameters in-place based on the provider detected
    from the model name. Returns the same model instance for chaining.
    """
    model_name = getattr(model, "model", getattr(model, "model_name", ""))
    if not isinstance(model_name, str):
        model_name = str(model_name)

    model_lower = model_name.lower()

    if "claude" in model_lower or "anthropic" in model_lower:
        # Claude: set effort and budget_tokens via model_kwargs
        kwargs = getattr(model, "model_kwargs", {}) or {}
        kwargs["reasoning_effort"] = profile.claude_effort
        kwargs["budget_tokens"] = profile.claude_budget_tokens
        model.model_kwargs = kwargs

    elif "gpt" in model_lower or "openai" in model_lower or "o3" in model_lower or "o4" in model_lower:
        # GPT: set reasoning_effort
        kwargs = getattr(model, "model_kwargs", {}) or {}
        kwargs["reasoning_effort"] = profile.gpt_reasoning_effort
        model.model_kwargs = kwargs

    elif "gemini" in model_lower or "google" in model_lower:
        # Gemini: set thinking_budget via model_kwargs
        kwargs = getattr(model, "model_kwargs", {}) or {}
        kwargs["thinking_budget"] = profile.gemini_thinking_budget
        model.model_kwargs = kwargs

    return model


def build_counsel_specs(profile: ComputeProfile) -> list[dict]:
    """Build counsel model specs from a ComputeProfile.

    Returns a list of model spec dicts compatible with
    ``consortium.counsel.DEFAULT_COUNSEL_MODEL_SPECS``.
    """
    return [
        {
            "model": "claude-opus-4-6",
            "reasoning_effort": profile.claude_effort,
            "budget_tokens": profile.claude_budget_tokens,
        },
        {
            "model": "claude-sonnet-4-6",
            "reasoning_effort": profile.claude_effort,
            "budget_tokens": profile.claude_budget_tokens,
        },
        {
            "model": "gpt-5.2",
            "reasoning_effort": profile.gpt_reasoning_effort,
        },
        {
            "model": "gemini-3.1-pro-preview",
            "thinking_budget": profile.gemini_thinking_budget,
        },
    ]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_compute_config(llm_config: Optional[dict] = None) -> AdaptiveComputeScheduler:
    """Load compute scaling config from .llm_config.yaml.

    Expected config structure::

        compute_scaling:
          default_tier: "standard"
          escalate_on_failure: true
          per_agent_overrides:
            math_prover_agent: "high"
            math_rigorous_verifier_agent: "max"
            adversarial_verifier: "max"
            ideation_agent: "high"
            experiment_design_agent: "high"
            writeup_agent: "standard"
            proofreading_agent: "low"
    """
    cfg = (llm_config or {}).get("compute_scaling", {})
    return AdaptiveComputeScheduler(
        default_tier=cfg.get("default_tier", "standard"),
        per_agent_overrides=cfg.get("per_agent_overrides", {}),
        escalate_on_failure=cfg.get("escalate_on_failure", True),
    )
