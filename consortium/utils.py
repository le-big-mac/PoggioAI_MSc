"""
Utilities for CLI agent mode.

Provides CLIBackendSpec, CLIBackendRegistry, and helper functions for
routing agent nodes to local CLI tools (Claude Code, Codex, Gemini CLI).
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from .models import AVAILABLE_MODELS  # noqa: F401 — re-exported for backward compat

logger = logging.getLogger(__name__)


def extract_content_between_markers(response: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Extract content between specified start and end markers from a response string."""
    try:
        start_escaped = re.escape(start_marker)
        end_escaped = re.escape(end_marker)
        pattern = f"{start_escaped}(.*?){end_escaped}"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None
    except Exception as e:
        logger.warning("extract_content_between_markers failed: %s", e)
        return None


def save_agent_memory(manager):
    """No-op after LangGraph migration — SqliteSaver checkpointer handles persistence."""
    pass


# ---------------------------------------------------------------------------
# CLI agent backend spec and registry
# ---------------------------------------------------------------------------


@dataclass
class CLIBackendSpec:
    """Specifies which CLI agent backend to use for a given agent node.

    Flows through the ``model`` parameter slot of agent build functions.
    ``create_specialist_agent`` expects this type and dispatches to the
    CLI agent subprocess wrapper.
    """

    backend: str          # "claude" | "codex" | "gemini"
    model: Optional[str] = None  # e.g. "claude-opus-4-6", "gpt-5.4"
    timeout_seconds: int = 3600


class CLIBackendRegistry:
    """Maps agent names to CLI backend specs with a default fallback."""

    def __init__(
        self,
        default: CLIBackendSpec,
        agent_overrides: Optional[dict[str, CLIBackendSpec]] = None,
    ):
        self._default = default
        self._overrides = agent_overrides or {}

    def get(self, agent_name: str) -> CLIBackendSpec:
        """Return the CLI backend spec for *agent_name*, or the default."""
        return self._overrides.get(agent_name, self._default)

    @property
    def default_spec(self) -> CLIBackendSpec:
        return self._default


def infer_cli_backend(model_id: Optional[str], default_backend: str = "claude") -> str:
    """Infer CLI backend name from a model identifier."""
    if not model_id:
        return default_backend
    mid = str(model_id).lower()
    if "claude" in mid or "anthropic" in mid:
        return "claude"
    if "gpt" in mid or "codex" in mid or mid.startswith(("o1-", "o3-", "o4-")):
        return "codex"
    if "gemini" in mid or "google" in mid:
        return "gemini"
    return default_backend


def create_cli_backend_registry(llm_config: dict) -> CLIBackendRegistry:
    """Build a :class:`CLIBackendRegistry` from the LLM config.

    Expected top-level keys in the config::

        default_backend: claude
        default_model: claude-opus-4-6
        timeout_seconds: 3600
        agent_backends:
          math_prover_agent:
            backend: claude
            model: claude-opus-4-6
            timeout_seconds: 7200
    """
    default = CLIBackendSpec(
        backend=llm_config.get("default_backend", "claude"),
        model=llm_config.get("default_model"),
        timeout_seconds=int(llm_config.get("timeout_seconds", 3600)),
    )

    overrides: dict[str, CLIBackendSpec] = {}
    for agent_name, spec in llm_config.get("agent_backends", {}).items():
        overrides[agent_name] = CLIBackendSpec(
            backend=spec.get("backend", default.backend),
            model=spec.get("model", default.model),
            timeout_seconds=int(spec.get("timeout_seconds", default.timeout_seconds)),
        )

    return CLIBackendRegistry(default, overrides)
