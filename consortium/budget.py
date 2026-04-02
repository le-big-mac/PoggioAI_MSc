"""
Legacy USD budget tracking compatibility module.

CLI mode uses ``cli_budget.py`` for real enforcement, but keeping this module
available preserves older utility/tests and makes the branch less surprising to
downstream imports.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional


class BudgetExceededError(RuntimeError):
    """Raised when the configured USD budget has been exhausted."""


class BudgetManager:
    def __init__(
        self,
        usd_limit: float,
        pricing: Dict[str, Dict[str, float]],
        state_path: str,
        ledger_path: str,
        lock_path: str,
        hard_stop: bool = True,
        fail_closed: bool = True,
    ) -> None:
        self.usd_limit = float(usd_limit)
        self.pricing = pricing or {}
        self.state_path = state_path
        self.ledger_path = ledger_path
        self.lock_path = lock_path
        self.hard_stop = hard_stop
        self.fail_closed = fail_closed
        self.total_usd = 0.0
        self.by_model: Dict[str, float] = {}
        self._record_lock = threading.Lock()
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.total_usd = float(data.get("total_usd", 0.0))
        self.by_model = data.get("by_model", {}) or {}

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        data = {
            "usd_limit": self.usd_limit,
            "total_usd": round(self.total_usd, 6),
            "by_model": self.by_model,
            "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def _write_ledger(self, entry):
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _get_pricing(self, model_id: str):
        if model_id in self.pricing:
            return self.pricing[model_id]
        if "/" in model_id:
            return self.pricing.get(model_id.split("/")[-1])
        return self.pricing.get(model_id)

    def _compute_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._get_pricing(model_id)
        if not pricing:
            if self.fail_closed:
                raise BudgetExceededError(
                    f"Budget enforcement: no pricing configured for model '{model_id}'."
                )
            return 0.0
        input_per_1k = float(pricing.get("input_per_1k", 0.0))
        output_per_1k = float(pricing.get("output_per_1k", 0.0))
        return (prompt_tokens / 1000.0) * input_per_1k + (completion_tokens / 1000.0) * output_per_1k

    def check_budget(self) -> None:
        if not self.hard_stop:
            return
        if os.path.exists(self.lock_path):
            raise BudgetExceededError(
                f"Budget lock file present: {self.lock_path}. USD limit {self.usd_limit} reached."
            )
        if self.total_usd >= self.usd_limit:
            raise BudgetExceededError(
                f"USD budget limit reached: {self.total_usd:.4f} / {self.usd_limit:.2f}."
            )

    def record_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        call_id: Optional[str] = None,
    ) -> float:
        cost = self._compute_cost(model_id, prompt_tokens, completion_tokens)

        with self._record_lock:
            self.total_usd += cost
            self.by_model[model_id] = round(self.by_model.get(model_id, 0.0) + cost, 6)
            snapshot_total = round(self.total_usd, 6)
            self._save_state()

        entry = {
            "call_id": call_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_id": model_id,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": round(cost, 6),
            "total_usd": snapshot_total,
            "usd_limit": self.usd_limit,
        }
        self._write_ledger(entry)

        if self.hard_stop and self.total_usd >= self.usd_limit:
            os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
            with open(self.lock_path, "w", encoding="utf-8") as f:
                f.write(f"Budget exceeded: {self.total_usd:.4f} / {self.usd_limit:.2f} USD\n")
        return cost

