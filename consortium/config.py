"""
LLM configuration loader for CLI agent mode.
"""

import logging
import os
import yaml

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
