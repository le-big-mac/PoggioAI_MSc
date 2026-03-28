"""
Model utilities for tools.

Extracts a model identifier string from whatever model object is passed in.
Tools that previously used litellm.completion() now delegate to
cli_completion() for LLM calls; this helper is still used to unwrap
model wrappers into a plain string identifier.
"""


def get_raw_model(model):
    """
    Extract a model identifier string from whatever model object is passed.

    Unwraps ChatLiteLLM and BudgetedLiteLLMModel wrappers to return a plain
    model_id string that tools can use for identification and logging.

    Args:
        model: ChatLiteLLM instance, BudgetedLiteLLMModel, a string model_id, or None.

    Returns:
        model_id string, or None if no model provided.
    """
    if model is None:
        return None

    if isinstance(model, str):
        return model

    # langchain_community ChatLiteLLM stores model name in .model attribute
    if hasattr(model, "model") and isinstance(model.model, str):
        return model.model

    # BudgetedLiteLLMModel: self.model is the wrapped ChatLiteLLM
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "model") and isinstance(inner.model, str):
        return inner.model

    # Fallback: return as-is (may be a legacy LiteLLMModel; tools handle it)
    return model
