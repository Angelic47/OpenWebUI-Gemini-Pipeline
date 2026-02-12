"""Token counting utilities for cache management in the Gemini Pipeline."""

"""Token counting utilities for cache management in the Gemini Pipeline."""

import logging
from typing import Any

from google.genai import types


class TokenCounter:
    """Handles token counting for cache management decisions.
    
    This class helps determine when to rebuild caches based on token counts:
    - Calculates baseline system_prompt token count
    - Calculates conversation token count (total)
    - Determines if non-cached content exceeds threshold
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.log = logging.getLogger("google_ai.pipe")
        self.system_prompt_tokens: int | None = None

    def count_system_prompt_tokens(
        self, 
        client, 
        model: str, 
        system_instruction: str
    ) -> int:
        """Calculate token count for system prompt (baseline).
        
        Args:
            client: Gemini client instance
            model: Model name (e.g., 'gemini-2.5-flash')
            system_instruction: System prompt text
            
        Returns:
            Token count for system prompt
        """
        try:
            response = client.models.count_tokens(
                model=model,
                contents=[],  # Empty content, only count system_instruction
                config=types.CountTokensConfig(system_instruction=system_instruction)
            )
            self.system_prompt_tokens = response.total_tokens
            self.log.debug(f"System prompt tokens: {self.system_prompt_tokens}")
            return self.system_prompt_tokens
        except Exception as e:
            self.log.exception(f"Failed to count system prompt tokens: {e}")
            raise RuntimeError(f"Failed to count system prompt tokens: {e}") from e

    def count_conversation_tokens(
        self, 
        client, 
        model: str, 
        contents: list[types.Content],
        system_instruction: str | None = None
    ) -> dict[str, Any]:
        """Calculate token count for conversation content.
        
        Supports multimodal content including images and text.
        
        Args:
            client: Gemini client instance
            model: Model name (e.g., 'gemini-2.5-flash')
            contents: List of Content objects (conversation history with text/images)
            system_instruction: Optional system prompt
            
        Returns:
            Dictionary with:
            - total_tokens: Total token count
            - cached_tokens: Cached content token count (if applicable)
            - non_cached_tokens: Non-cached token count
        """
        try:
            config = types.CountTokensConfig()
            if system_instruction:
                config.system_instruction = system_instruction
            
            response = client.models.count_tokens(
                model=model,
                contents=contents,
                config=config
            )
            
            total = response.total_tokens
            cached = getattr(response, 'cached_content_token_count', 0) or 0
            non_cached = total - cached
            
            result = {
                "total_tokens": total,
                "cached_tokens": cached,
                "non_cached_tokens": non_cached
            }
            
            self.log.debug(
                f"Token count - Total: {total}, Cached: {cached}, Non-cached: {non_cached}"
            )
            return result
        except Exception as e:
            self.log.exception(f"Failed to count conversation tokens: {e}")
            raise

    def should_rebuild_cache(
        self, 
        non_cached_tokens: int, 
        threshold_multiplier: float = 1.0
    ) -> bool:
        """Determine if cache should be rebuilt based on token threshold.
        
        Strategy: Rebuild cache when non-cached content exceeds system_prompt
        token count multiplied by threshold.
        
        Args:
            non_cached_tokens: Token count of non-cached content
            threshold_multiplier: Multiplier for system_prompt baseline (default: 1.0)
            
        Returns:
            True if cache should be rebuilt, False otherwise
            
        Raises:
            ValueError: If system_prompt token count not set
        """
        if self.system_prompt_tokens is None:
            raise ValueError(
                "System prompt token count not set. "
                "Call count_system_prompt_tokens() first."
            )
        
        threshold = int(self.system_prompt_tokens * threshold_multiplier)
        should_rebuild = non_cached_tokens > threshold
        
        self.log.debug(
            f"Cache check - Non-cached: {non_cached_tokens}, "
            f"Threshold ({threshold_multiplier}x): {threshold}, "
            f"Rebuild needed: {should_rebuild}"
        )
        
        if should_rebuild:
            self.log.info(
                f"Cache rebuild triggered: {non_cached_tokens} tokens exceeds "
                f"threshold of {threshold}"
            )
        
        return should_rebuild
