"""Generation configuration for the Gemini Pipeline."""

import os
import logging
from google.genai import types
from typing import Any


class GenerationConfigurator:
    """Configures generation parameters and safety settings."""

    def __init__(self, pipe, model_manager):
        self.pipe = pipe
        self.model_manager = model_manager
        self.log = logging.getLogger("google_ai.pipe")

    def get_user_valve_value(self, __user__: dict | None, valve_name: str) -> str | None:
        """Get a user valve value, returning None if not set or set to 'default'."""
        if __user__ and "valves" in __user__:
            value = getattr(__user__["valves"], valve_name, None)
            if value and value != "default":
                return value
        return None

    def configure_generation(
        self,
        body: dict[str, Any],
        system_instruction: str | None,
        __metadata__: dict[str, Any],
        __tools__: dict[str, Any] | None = None,
        __user__: dict | None = None,
        enable_image_generation: bool = False,
        model_id: str = "",
        client: Any = None,
    ) -> types.GenerateContentConfig:
        """
        Configure generation parameters and safety settings.

        Args:
            body: The request body containing generation parameters
            system_instruction: Optional system instruction string
            enable_image_generation: Whether to enable image generation
            model_id: The model ID being used (for feature support checks)

        Returns:
            types.GenerateContentConfig
        """
        gen_config_params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "max_output_tokens": body.get("max_tokens"),
            "stop_sequences": body.get("stop") or None,
            "system_instruction": system_instruction,
        }

        # Enable image generation if requested
        if enable_image_generation:
            gen_config_params["response_modalities"] = ["TEXT", "IMAGE"]

            # Configure image generation parameters (aspect ratio and resolution)
            # ImageConfig is only supported by Gemini 3 models
            if self.model_manager._check_image_config_support(model_id):
                # Body parameters override valve defaults for per-request customization
                # Get aspect_ratio: body -> user_valves (if not default) -> system valves
                user_aspect_ratio = self.get_user_valve_value(
                    __user__, "IMAGE_GENERATION_ASPECT_RATIO"
                )
                aspect_ratio = body.get(
                    "aspect_ratio",
                    user_aspect_ratio or self.pipe.valves.IMAGE_GENERATION_ASPECT_RATIO,
                )

                # Get resolution: body -> user_valves (if not default) -> system valves
                user_resolution = self.get_user_valve_value(
                    __user__, "IMAGE_GENERATION_RESOLUTION"
                )
                resolution = body.get(
                    "resolution",
                    user_resolution or self.pipe.valves.IMAGE_GENERATION_RESOLUTION,
                )

                # Validate and normalize the values
                validated_aspect_ratio = self.model_manager.validate_aspect_ratio(aspect_ratio)
                validated_resolution = self.model_manager.validate_resolution(resolution)

                # Create image config if we have at least one valid value
                if validated_aspect_ratio or validated_resolution:
                    try:
                        image_config_params = {}
                        if validated_aspect_ratio:
                            image_config_params["aspect_ratio"] = validated_aspect_ratio
                        if validated_resolution:
                            image_config_params["image_size"] = validated_resolution
                        gen_config_params["image_config"] = types.ImageConfig(
                            **image_config_params
                        )
                        self.log.debug(
                            f"Image generation config: aspect_ratio={validated_aspect_ratio}, resolution={validated_resolution}"
                        )
                    except (AttributeError, TypeError) as e:
                        # Fall back if SDK does not support ImageConfig
                        self.log.warning(
                            f"ImageConfig not supported by SDK version: {e}. Image generation will use default settings."
                        )
                    except Exception as e:
                        # Log unexpected errors but continue without image config
                        self.log.warning(
                            f"Unexpected error configuring ImageConfig: {e}"
                        )
            else:
                self.log.debug(
                    f"Model {model_id} does not support ImageConfig (aspect_ratio/resolution). "
                    "ImageConfig is only available for Gemini 3 image models."
                )

        # Configure Gemini thinking/reasoning for models that support it
        # This is independent of include_thoughts - thinking config controls HOW the model reasons,
        # while include_thoughts controls whether the reasoning is shown in the output
        if self.model_manager._check_thinking_support(model_id):
            try:
                thinking_config_params: dict[str, Any] = {}

                # Determine include_thoughts setting
                include_thoughts = body.get("include_thoughts", True)
                if not self.pipe.valves.INCLUDE_THOUGHTS:
                    include_thoughts = False
                    self.log.debug(
                        "Thoughts output disabled via GOOGLE_INCLUDE_THOUGHTS"
                    )
                thinking_config_params["include_thoughts"] = include_thoughts

                # Check if model supports thinking_level (Gemini 3 models)
                if self.model_manager._check_thinking_level_support(model_id):
                    # For Gemini 3 models, use thinking_level (not thinking_budget)
                    # Per-chat reasoning_effort overrides environment-level THINKING_LEVEL
                    reasoning_effort = body.get("reasoning_effort")
                    validated_level = None
                    source = None

                    if reasoning_effort:
                        validated_level = self.model_manager.validate_thinking_level(
                            reasoning_effort
                        )
                        if validated_level:
                            source = "per-chat reasoning_effort"
                        else:
                            self.log.debug(
                                f"Invalid reasoning_effort '{reasoning_effort}', falling back to THINKING_LEVEL"
                            )

                    # Fall back to environment-level THINKING_LEVEL if no valid reasoning_effort
                    if not validated_level:
                        validated_level = self.model_manager.validate_thinking_level(
                            self.pipe.valves.THINKING_LEVEL
                        )
                        if validated_level:
                            source = "THINKING_LEVEL"

                    if validated_level:
                        thinking_config_params["thinking_level"] = validated_level
                        self.log.debug(
                            f"Using thinking_level='{validated_level}' from {source} for model {model_id}"
                        )
                    else:
                        self.log.debug(
                            f"Using default thinking level for model {model_id}"
                        )
                else:
                    # For non-Gemini 3 models (e.g., Gemini 2.5), use thinking_budget
                    # Body-level thinking_budget overrides environment-level THINKING_BUDGET
                    body_thinking_budget = body.get("thinking_budget")
                    validated_budget = None
                    source = None

                    if body_thinking_budget is not None:
                        validated_budget = self.model_manager.validate_thinking_budget(
                            body_thinking_budget
                        )
                        if validated_budget is not None:
                            source = "body thinking_budget"
                        else:
                            self.log.debug(
                                f"Invalid body thinking_budget '{body_thinking_budget}', falling back to THINKING_BUDGET"
                            )

                    # Fall back to environment-level THINKING_BUDGET
                    if validated_budget is None:
                        validated_budget = self.model_manager.validate_thinking_budget(
                            self.pipe.valves.THINKING_BUDGET
                        )
                        if validated_budget is not None:
                            source = "THINKING_BUDGET"

                    if validated_budget == 0:
                        # Disable thinking if budget is 0
                        thinking_config_params["thinking_budget"] = 0
                        self.log.debug(
                            f"Thinking disabled via thinking_budget=0 from {source} for model {model_id}"
                        )
                    elif validated_budget is not None and validated_budget > 0:
                        thinking_config_params["thinking_budget"] = validated_budget
                        self.log.debug(
                            f"Using thinking_budget={validated_budget} from {source} for model {model_id}"
                        )
                    else:
                        # -1 or None means dynamic thinking
                        thinking_config_params["thinking_budget"] = -1
                        self.log.debug(
                            f"Using dynamic thinking (model decides) for model {model_id}"
                        )

                gen_config_params["thinking_config"] = types.ThinkingConfig(
                    **thinking_config_params
                )
            except (AttributeError, TypeError) as e:
                # Fall back if SDK/model does not support ThinkingConfig
                self.log.debug(f"ThinkingConfig not supported: {e}")
            except Exception as e:
                # Log unexpected errors but continue without thinking config
                self.log.warning(f"Unexpected error configuring ThinkingConfig: {e}")

        # Configure safety settings
        if self.pipe.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
                ),
            ]
            gen_config_params["safety_settings"] = safety_settings

        # Add various tools to Gemini as required
        features = __metadata__.get("features", {})
        params = __metadata__.get("params", {})
        tools = []

        if features.get("google_search_tool", False):
            self.log.debug("Enabling Google search grounding")
            tools.append(types.Tool(google_search=types.GoogleSearch()))
            self.log.debug("Enabling URL context grounding")
            tools.append(types.Tool(url_context=types.UrlContext()))

        if features.get("vertex_ai_search", False) or (
            self.pipe.valves.USE_VERTEX_AI
            and (self.pipe.valves.VERTEX_AI_RAG_STORE or os.getenv("VERTEX_AI_RAG_STORE"))
        ):
            vertex_rag_store = (
                params.get("vertex_rag_store")
                or self.pipe.valves.VERTEX_AI_RAG_STORE
                or os.getenv("VERTEX_AI_RAG_STORE")
            )
            if vertex_rag_store:
                self.log.debug(
                    f"Enabling Vertex AI Search grounding: {vertex_rag_store}"
                )
                tools.append(
                    types.Tool(
                        retrieval=types.Retrieval(
                            vertex_ai_search=types.VertexAISearch(
                                datastore=vertex_rag_store
                            )
                        )
                    )
                )
            else:
                self.log.warning(
                    "Vertex AI Search requested but vertex_rag_store not provided in params, valves, or env"
                )

        if __tools__ is not None and params.get("function_calling") == "native":
            for name, tool_def in __tools__.items():
                if not name.startswith("_"):
                    tool = tool_def["callable"]
                    if hasattr(tool, "__signature__"):
                        self.log.debug(
                            f"Adding tool '{name}' with signature {tool.__signature__}"
                        )
                    else:
                        self.log.debug(f"Adding tool '{name}' (no signature)")
                    tools.append(tool)

        if tools:
            gen_config_params["tools"] = tools

        # Filter out None values for generation config
        filtered_params = {k: v for k, v in gen_config_params.items() if v is not None}

        # Process Gemini cache
        if "tools" in filtered_params or "tool_config" in filtered_params:
            # FIXME: gemini api 400 when with tools and cached_content
            #  CachedContent can not be used with GenerateContent request setting system_instruction, tools or tool_config.
            #  Proposed fix: move those values to CachedContent from GenerateContent request
            #
            # But with automatic tools calling, we cannot pass real tools functions into cache
            # So for now, skip caching when tools are present
            self.log.debug(
                "Skipping Gemini cache processing due to presence of tools in generation config"
            )
            return types.GenerateContentConfig(**filtered_params)

        from .cache import CacheManager
        cache_manager = CacheManager(self.pipe.valves)
        cache = cache_manager.process_gemini_cache(client, model_id, __metadata__, body, filtered_params)
        self.log.debug(f"Using cache: {cache}")
        del filtered_params[
            "system_instruction"
        ]  # Remove system_instruction before caching
        filtered_params["cached_content"] = cache.name

        return types.GenerateContentConfig(**filtered_params)
