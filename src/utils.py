"""Utility functions for the Gemini Pipeline."""

import re
from typing import Any


def sanitize_header_value(value: Any, max_length: int = 255) -> str:
    """Sanitize header values for safe transmission."""
    if value is None:
        return ""
    # Convert to string and remove all control characters
    sanitized = re.sub(r"[\x00-\x1F\x7F]", "", str(value))
    sanitized = sanitized.strip()
    return (
        sanitized[:max_length]
        if len(sanitized) > max_length
        else sanitized
    )


def strip_prefix(model_name: str) -> str:
    """
    Extract the model identifier using regex, handling various naming conventions.
    e.g., "google_gemini_pipeline.gemini-2.5-flash-preview-04-17" → "gemini-2.5-flash-preview-04-17"
    e.g., "models/gemini-1.5-flash-001" → "gemini-1.5-flash-001"
    e.g., "publishers/google/models/gemini-1.5-pro" → "gemini-1.5-pro"
    """
    # Use regex to remove everything up to and including the last '/' or the first '.'
    stripped = re.sub(r"^(.*/|[^.]*\.)", "", model_name)
    return stripped

def get_model_id_for_cache(__metadata__: dict[str, Any], body: dict[str, Any]) -> str:
    """
    Get a simplified model identifier for caching purposes.
    The function checks for a "__model__" entry in the metadata, which may contain both "id" and "base_model_id". 
    It prioritizes "id" for the workspace-specific model, but falls back to "base_model_id" if "id" is not present. 
    If neither is available, it returns "unknown_model". This allows for more effective caching by using a consistent model identifier.
    e.g., "lumina-gemini-3-pro", "-kimi-k25", etc.
    """
    model_obj = __metadata__.get("__model__", {})
    workspace_model_id = model_obj.get("id") if model_obj else body.get("model")
    base_model_id = model_obj.get("base_model_id") or workspace_model_id
    return workspace_model_id or base_model_id or "unknown_model"