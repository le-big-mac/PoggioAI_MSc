"""
LLM configuration loader for CLI agent mode.
"""

import logging
import os
import yaml
import functools

logger = logging.getLogger(__name__)


def _validate_config(config):
    """Warn about missing or malformed config sections. Does not block execution."""
    if not isinstance(config, dict):
        logger.warning("LLM config is not a dict — ignoring")
        return
    backend = config.get("default_backend", "claude")
    if backend not in ("claude", "codex", "gemini"):
        logger.warning(
            "default_backend must be 'claude', 'codex', or 'gemini', got %r",
            backend,
        )
    if "default_model" not in config:
        logger.warning("default_model is not set in .llm_config.yaml")


def load_llm_config():
    """Load LLM configuration from .llm_config.yaml if it exists."""
    config_path = ".llm_config.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded LLM config from %s", config_path)
            _validate_config(config)
            return config
        except yaml.YAMLError as e:
            logger.warning("Error loading %s: %s", config_path, e)
            return None
    else:
        logger.info("No %s found, using defaults", config_path)
        return None


def filter_model_params(original_func):
    """Compatibility shim for legacy tests and docs.

    CLI mode does not route model calls through litellm, but some helpers and
    tests still depend on the old parameter normalization behavior.
    """

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else "")

        if isinstance(model, str) and (model.startswith("gpt-5") or "codex" in model):
            unsupported_params = {
                "stop",
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "logprobs",
                "top_logprobs",
                "logit_bias",
                "max_tokens",
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
            if "max_tokens" in kwargs:
                filtered_kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            final_kwargs = filtered_kwargs

        elif isinstance(model, str) and ("claude" in model or "anthropic" in model):
            fk = kwargs.copy()

            if "effort" in fk and "reasoning_effort" not in fk:
                fk["reasoning_effort"] = fk.pop("effort")
            elif "effort" in fk:
                fk.pop("effort")

            if "temperature" in fk and "top_p" in fk:
                fk.pop("top_p")

            budget = fk.pop("budget_tokens", None)
            thinking = fk.get("thinking")
            if budget is not None:
                if budget <= 0:
                    fk["thinking"] = {"type": "disabled"}
                else:
                    fk["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
            elif thinking is None:
                pass

            if isinstance(fk.get("thinking"), dict) and fk["thinking"].get("type") == "enabled":
                budget_tokens = int(fk["thinking"].get("budget_tokens", 0))
                margin = 2048
                mt = fk.get("max_tokens")
                if mt is None or int(mt) <= budget_tokens:
                    fk["max_tokens"] = int(budget_tokens + margin)

            final_kwargs = fk
        else:
            final_kwargs = kwargs

        return original_func(*args, **final_kwargs)

    return wrapper
