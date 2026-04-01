"""
Canonical model registry — single source of truth for supported models,
context limits, and provider mappings.
"""

from __future__ import annotations

MODEL_REGISTRY: dict[str, dict] = {
    # OpenAI GPT-5 models
    "gpt-5": {"context_limit": 256_000, "provider": "openai"},
    "gpt-5-mini": {"context_limit": 256_000, "provider": "openai"},
    "gpt-5-nano": {"context_limit": 256_000, "provider": "openai"},
    "gpt-5.4": {"context_limit": 1_050_000, "provider": "openai"},
    "gpt-5.3-codex": {"context_limit": 200_000, "provider": "openai"},
    "gpt-5.2": {"context_limit": 256_000, "provider": "openai"},
    # OpenAI GPT-4 models
    "gpt-4o": {"context_limit": 128_000, "provider": "openai"},
    "gpt-4.1-mini-2025-04-14": {"context_limit": 128_000, "provider": "openai"},
    # OpenAI reasoning models
    "o3-2025-04-16": {"context_limit": 200_000, "provider": "openai"},
    "o3-pro-2025-06-10": {"context_limit": 200_000, "provider": "openai"},
    "o4-mini-2025-04-16": {"context_limit": 128_000, "provider": "openai"},
    # Anthropic Claude models
    "claude-opus-4-6": {"context_limit": 200_000, "provider": "anthropic"},
    "claude-sonnet-4-6": {"context_limit": 200_000, "provider": "anthropic"},
    "claude-opus-4-20250514": {"context_limit": 200_000, "provider": "anthropic"},
    "claude-sonnet-4-20250514": {"context_limit": 200_000, "provider": "anthropic"},
    "claude-sonnet-4-5": {"context_limit": 200_000, "provider": "anthropic"},
    "claude-sonnet-4-5-20250929": {"context_limit": 200_000, "provider": "anthropic"},
    # DeepSeek models
    "deepseek-chat": {"context_limit": 64_000, "provider": "deepseek"},
    "deepseek-coder": {"context_limit": 64_000, "provider": "deepseek"},
    # xAI models
    "grok-4-0709": {"context_limit": 128_000, "provider": "xai"},
    # Google Gemini models
    "gemini-2.5-pro": {"context_limit": 1_000_000, "provider": "google"},
    "gemini-2.5-flash": {"context_limit": 1_000_000, "provider": "google"},
    "gemini-3.1-pro-preview": {"context_limit": 2_000_000, "provider": "google"},
}

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def get_context_limit(model_id: str) -> int:
    """Return the context window size for a model, with fallback for unknown models."""
    entry = MODEL_REGISTRY.get(model_id)
    if entry:
        return entry["context_limit"]
    # Fallback for provider-prefixed variants like "anthropic/claude-opus-4-6"
    if "/" in model_id:
        bare = model_id.split("/")[-1]
        entry = MODEL_REGISTRY.get(bare)
        if entry:
            return entry["context_limit"]
    return 128_000


def get_provider(model_id: str) -> str:
    """Return the provider name for a model (e.g. 'openai', 'anthropic')."""
    entry = MODEL_REGISTRY.get(model_id)
    if entry:
        return entry["provider"]
    if "/" in model_id:
        bare = model_id.split("/")[-1]
        entry = MODEL_REGISTRY.get(bare)
        if entry:
            return entry["provider"]
    # Infer from string patterns as fallback
    if "claude" in model_id or "anthropic" in model_id:
        return "anthropic"
    if "gpt" in model_id or model_id.startswith(("o1-", "o3-", "o4-")):
        return "openai"
    if "gemini" in model_id:
        return "google"
    if "deepseek" in model_id:
        return "deepseek"
    if "grok" in model_id:
        return "xai"
    if "llama" in model_id:
        return "openrouter"
    return "unknown"


# Provider → CLI backend mapping
_PROVIDER_TO_CLI_BACKEND = {
    "anthropic": "claude",
    "openai": "codex",
    "google": "gemini",
}


def get_cli_backend(model_id: str) -> str:
    """Return the CLI backend name for a model (e.g. 'claude', 'codex', 'gemini').

    Falls back to 'claude' for unknown providers.
    """
    provider = get_provider(model_id)
    return _PROVIDER_TO_CLI_BACKEND.get(provider, "claude")
