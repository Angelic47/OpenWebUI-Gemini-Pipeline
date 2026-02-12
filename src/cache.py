"""Cache management for the Gemini Pipeline."""

import datetime
import hashlib
import logging
from google.genai import types
from typing import Any
from .utils import get_model_id_for_cache


class CacheManager:
    """Manages Gemini cache for system prompts."""

    def __init__(self, valves):
        self.valves = valves
        self.log = logging.getLogger("google_ai.pipe")

    def process_gemini_cache(self, client, __metadata__: dict[str, Any], body: dict[str, Any], gen_config: dict) -> Any:
        """Process and manage Gemini cache for system prompts."""
        gen_tool_list: list[types.Tool] = gen_config.get("tools", None)
        gen_toolConfig: types.ToolConfig = gen_config.get("tool_config", None)
        cache_ttl = datetime.timedelta(hours=1)

        model = get_model_id_for_cache(__metadata__, body)
        cache_name = "OpenWebUI Lumina System Prompt Cache - " + model
        cache_name_tool_signature = ""
        if gen_tool_list is not None and len(gen_tool_list) > 0:
            # generate a simple signature for the tool list
            tool_names = [tool.name for tool in gen_tool_list]
            cache_name_tool_signature = ",".join(tool_names)
            cache_name_tool_signature = hashlib.md5(
                cache_name_tool_signature.encode("utf-8")
            ).hexdigest()[:8]
            cache_name += " - Tools: " + cache_name_tool_signature
            cache_ttl = datetime.timedelta(minutes=20)  # shorter ttl for tool caches
        if gen_toolConfig is not None:
            # include tool config in the signature
            tool_config_str = str(gen_toolConfig)
            tool_config_signature = hashlib.md5(
                tool_config_str.encode("utf-8")
            ).hexdigest()[:8]
            cache_name += " - ToolConfig: " + tool_config_signature
            cache_ttl = datetime.timedelta(minutes=20)  # shorter ttl for tool caches

        cache = None
        for _cache in client.caches.list():
            if _cache.display_name == cache_name:
                self.log.debug(f"Using existing cache: {_cache}")
                cache = _cache
                break

        cache_ttl_next = datetime.datetime.now() + cache_ttl
        cache_ttl_nextday = datetime.datetime.now().date() + datetime.timedelta(days=1)

        if cache is None:
            # Check if larger than tomorrow 00:00 Local Time
            if cache_ttl_next.date() >= cache_ttl_nextday:
                # convert local time to utc time
                cache_ttl_next = datetime.datetime.combine(
                    cache_ttl_nextday, datetime.time.min
                )

            # Create a cache
            _cache_config = {
                "display_name": cache_name,
                "system_instruction": gen_config["system_instruction"],
                "expire_time": cache_ttl_next.astimezone(datetime.timezone.utc),
            }
            if gen_tool_list is not None and len(gen_tool_list) > 0:
                _cache_config["tool"] = gen_tool_list
            if gen_toolConfig is not None:
                _cache_config["tool_config"] = gen_toolConfig
            cache_config = types.CreateCachedContentConfig(**_cache_config)
            self.log.debug("Creating new cache for system prompt...")
            cache = client.caches.create(model=model, config=cache_config)
            self.log.debug(f"Created cache: {cache}")
        else:
            # update expire time (convert local time to utc time)
            self.log.debug(f"Updating cache expire time to {cache_ttl_next}...")
            _cache_ttl_next_utc = cache_ttl_next.astimezone(datetime.timezone.utc)
            cache = client.caches.update(
                name=cache.name,
                config=types.UpdateCachedContentConfig(expire_time=_cache_ttl_next_utc),
            )
            self.log.debug(f"Updated cache: {cache}")

        return cache
