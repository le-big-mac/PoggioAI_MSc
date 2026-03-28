"""
CLI budget tracker — invocation and wall-clock based budget enforcement
for subscription-based CLI agents (Claude Code, Codex, Gemini CLI).

Replaces the per-token USD budget tracking in budget.py when running
in CLI mode, since subscription-based CLI tools have no per-token cost.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class CLIBudgetExceededError(Exception):
    """Raised when the CLI invocation or wall-clock budget is exceeded."""


@dataclass
class CLIInvocationRecord:
    agent_name: str
    backend: str
    model: str
    duration_seconds: float
    prompt_chars: int
    output_chars: int
    timestamp: float = field(default_factory=time.time)


class CLIBudgetTracker:
    """Tracks CLI agent invocations and enforces limits.

    Limits:
        max_invocations: Total CLI calls allowed (default 100).
        max_wall_clock_seconds: Total subprocess time allowed (default 4h).
    """

    def __init__(
        self,
        max_invocations: int = 100,
        max_wall_clock_seconds: float = 14400,
        state_dir: Optional[str] = None,
    ):
        self.max_invocations = max_invocations
        self.max_wall_clock = max_wall_clock_seconds
        self._lock = threading.Lock()
        self._invocations: list[CLIInvocationRecord] = []
        self._total_seconds = 0.0
        self._state_dir = state_dir

        # Resume from saved state if available
        if state_dir:
            self._state_path = os.path.join(state_dir, "cli_budget_state.json")
            self._ledger_path = os.path.join(state_dir, "cli_budget_ledger.jsonl")
            self._load_state()
        else:
            self._state_path = None
            self._ledger_path = None

    def _load_state(self) -> None:
        if self._state_path and os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                self._total_seconds = state.get("total_seconds", 0.0)
                count = state.get("invocation_count", 0)
                logger.info(
                    "Resumed CLI budget: %d invocations, %.1fs elapsed",
                    count, self._total_seconds,
                )
            except (json.JSONDecodeError, OSError):
                pass

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            state = {
                "invocation_count": len(self._invocations),
                "total_seconds": self._total_seconds,
                "max_invocations": self.max_invocations,
                "max_wall_clock_seconds": self.max_wall_clock,
            }
            tmp = self._state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self._state_path)
        except OSError:
            pass

    def _append_ledger(self, record: CLIInvocationRecord) -> None:
        if not self._ledger_path:
            return
        try:
            with open(self._ledger_path, "a") as f:
                f.write(json.dumps({
                    "agent_name": record.agent_name,
                    "backend": record.backend,
                    "model": record.model,
                    "duration_seconds": record.duration_seconds,
                    "prompt_chars": record.prompt_chars,
                    "output_chars": record.output_chars,
                    "timestamp": record.timestamp,
                }) + "\n")
        except OSError:
            pass

    def check_budget(self) -> None:
        """Raise CLIBudgetExceededError if limits are hit."""
        with self._lock:
            if len(self._invocations) >= self.max_invocations:
                raise CLIBudgetExceededError(
                    f"CLI invocation limit reached: {len(self._invocations)}/{self.max_invocations}"
                )
            if self._total_seconds >= self.max_wall_clock:
                raise CLIBudgetExceededError(
                    f"CLI wall-clock limit reached: {self._total_seconds:.0f}s/{self.max_wall_clock:.0f}s"
                )

    def record_invocation(
        self,
        agent_name: str,
        backend: str,
        model: str,
        duration_seconds: float,
        prompt_chars: int,
        output_chars: int,
    ) -> None:
        """Record a CLI agent invocation and persist state."""
        record = CLIInvocationRecord(
            agent_name=agent_name,
            backend=backend,
            model=model,
            duration_seconds=duration_seconds,
            prompt_chars=prompt_chars,
            output_chars=output_chars,
        )
        with self._lock:
            self._invocations.append(record)
            self._total_seconds += duration_seconds
            self._save_state()
            self._append_ledger(record)

        logger.info(
            "[CLI Budget] %s: %.1fs (total: %d/%d invocations, %.0f/%.0fs wall-clock)",
            agent_name, duration_seconds,
            len(self._invocations), self.max_invocations,
            self._total_seconds, self.max_wall_clock,
        )

    @property
    def summary(self) -> dict:
        with self._lock:
            by_backend: dict[str, int] = {}
            by_agent: dict[str, float] = {}
            for r in self._invocations:
                by_backend[r.backend] = by_backend.get(r.backend, 0) + 1
                by_agent[r.agent_name] = by_agent.get(r.agent_name, 0.0) + r.duration_seconds
            return {
                "invocation_count": len(self._invocations),
                "total_seconds": self._total_seconds,
                "by_backend": by_backend,
                "by_agent": by_agent,
            }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_cli_tracker: Optional[CLIBudgetTracker] = None


def set_global_cli_tracker(tracker: CLIBudgetTracker) -> None:
    global _global_cli_tracker
    _global_cli_tracker = tracker


def get_global_cli_tracker() -> Optional[CLIBudgetTracker]:
    return _global_cli_tracker
