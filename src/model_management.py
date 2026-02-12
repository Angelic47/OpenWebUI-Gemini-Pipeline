"""Model management and validation for the Gemini Pipeline."""

import re
import time
import logging
from google import genai
from google.genai import types
from typing import Any
from .encryption import EncryptedStr
from .utils import strip_prefix, sanitize_header_value


class ModelManager:
    """Manages model listing, caching, and validation."""

    # Known image generation models (both Gemini 2.5 and Gemini 3)
    IMAGE_GENERATION_MODELS = [
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-image-preview",
        "gemini-3-flash-image",
        "gemini-3-flash-image-preview",
        "gemini-3-pro-image",
        "gemini-3-pro-image-preview",
    ]

    # Models that do NOT support thinking
    NON_THINKING_MODELS = [
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash-image",
    ]

    def __init__(self, pipe):
        self.pipe = pipe
        self.log = logging.getLogger("google_ai.pipe")
        self._model_cache: list[dict[str, str]] | None = None
        self._model_cache_time: float = 0

    def _validate_api_key(self) -> None:
        """
        Validates that the necessary Google API credentials are set.

        Raises:
            ValueError: If the required credentials are not set.
        """
        if self.pipe.valves.USE_VERTEX_AI:
            if not self.pipe.valves.VERTEX_PROJECT:
                self.log.error("USE_VERTEX_AI is true, but VERTEX_PROJECT is not set.")
                raise ValueError(
                    "VERTEX_PROJECT is not set. Please provide the Google Cloud project ID."
                )
            # For Vertex AI, location has a default, so project is the main thing to check.
            # Actual authentication will be handled by ADC or environment.
            self.log.debug(
                "Using Vertex AI. Ensure ADC or service account is configured."
            )
        else:
            if not self.pipe.valves.GOOGLE_API_KEY:
                self.log.error("GOOGLE_API_KEY is not set (and not using Vertex AI).")
                raise ValueError(
                    "GOOGLE_API_KEY is not set. Please provide the API key in the environment variables or valves."
                )
            self.log.debug("Using Google Generative AI API with API Key.")

    def _get_client(self) -> genai.Client:
        """
        Validates API credentials and returns a genai.Client instance.
        """
        self._validate_api_key()

        if self.pipe.valves.USE_VERTEX_AI:
            self.log.debug(
                f"Initializing Vertex AI client (Project: {self.pipe.valves.VERTEX_PROJECT}, Location: {self.pipe.valves.VERTEX_LOCATION})"
            )
            return genai.Client(
                vertexai=True,
                project=self.pipe.valves.VERTEX_PROJECT,
                location=self.pipe.valves.VERTEX_LOCATION,
            )
        else:
            self.log.debug("Initializing Google Generative AI client with API Key")
            headers = {}
            if (
                self.pipe.valves.ENABLE_FORWARD_USER_INFO_HEADERS
                and hasattr(self.pipe, "user")
                and self.pipe.user
            ):
                user_attrs = {
                    "X-OpenWebUI-User-Name": sanitize_header_value(
                        getattr(self.pipe.user, "name", None)
                    ),
                    "X-OpenWebUI-User-Id": sanitize_header_value(
                        getattr(self.pipe.user, "id", None)
                    ),
                    "X-OpenWebUI-User-Email": sanitize_header_value(
                        getattr(self.pipe.user, "email", None)
                    ),
                    "X-OpenWebUI-User-Role": sanitize_header_value(
                        getattr(self.pipe.user, "role", None)
                    ),
                }
                headers = {k: v for k, v in user_attrs.items() if v not in (None, "")}
            options = types.HttpOptions(
                api_version=self.pipe.valves.API_VERSION,
                base_url=self.pipe.valves.BASE_URL,
                headers=headers,
            )
            return genai.Client(
                api_key=EncryptedStr.decrypt(self.pipe.valves.GOOGLE_API_KEY),
                http_options=options,
            )

    def get_google_models(self, force_refresh: bool = False) -> list[dict[str, str]]:
        """
        Retrieve available Google models suitable for content generation.
        Uses caching to reduce API calls.

        Args:
            force_refresh: Whether to force refreshing the model cache

        Returns:
            List of dictionaries containing model id and name.
        """
        # Check cache first
        current_time = time.time()
        if (
            not force_refresh
            and self._model_cache is not None
            and (current_time - self._model_cache_time) < self.pipe.valves.MODEL_CACHE_TTL
        ):
            self.log.debug("Using cached model list")
            return self._model_cache

        try:
            client = self._get_client()
            self.log.debug("Fetching models from Google API")
            models = client.models.list()
            available_models = []
            for model in models:
                actions = model.supported_actions
                if actions is None or "generateContent" in actions:
                    model_id = strip_prefix(model.name)
                    model_name = model.display_name or model_id

                    # Check if model supports image generation
                    supports_image_generation = self._check_image_generation_support(
                        model_id
                    )
                    if supports_image_generation:
                        model_name += " 🎨"  # Add image generation indicator

                    available_models.append(
                        {
                            "id": model_id,
                            "name": model_name,
                            "image_generation": supports_image_generation,
                        }
                    )

            model_map = {model["id"]: model for model in available_models}

            # Filter map to only include models starting with 'gemini-'
            filtered_models = {
                k: v for k, v in model_map.items() if k.startswith("gemini-")
            }

            # Update cache
            self._model_cache = list(filtered_models.values())
            self._model_cache_time = current_time
            self.log.debug(f"Found {len(self._model_cache)} Gemini models")
            return self._model_cache

        except Exception as e:
            self.log.exception(f"Could not fetch models from Google: {str(e)}")
            # Return a specific error entry for the UI
            return [{"id": "error", "name": f"Could not fetch models: {str(e)}"}]

    def _check_image_generation_support(self, model_id: str) -> bool:
        """
        Check if a model supports image generation.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports image generation, False otherwise
        """
        # Check for exact matches or pattern matches
        for pattern in self.IMAGE_GENERATION_MODELS:
            if model_id == pattern or pattern in model_id:
                return True

        # Additional pattern checking for future models
        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return True

        return False

    def _check_image_config_support(self, model_id: str) -> bool:
        """
        Check if a model supports ImageConfig (aspect_ratio and image_size parameters).

        ImageConfig is only supported by Gemini 3 image generation models.
        Gemini 2.5 image models support image generation but not ImageConfig.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports ImageConfig, False otherwise
        """
        # ImageConfig is only supported by Gemini 3 models
        model_lower = model_id.lower()

        # Check if it's a Gemini 3 model
        if "gemini-3-" not in model_lower:
            return False

        # Check if it's an image generation model
        return self._check_image_generation_support(model_id)

    def _check_thinking_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking feature.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking, False otherwise
        """
        # Check for exact matches
        for pattern in self.NON_THINKING_MODELS:
            if model_id == pattern or pattern in model_id:
                return False

        # Additional pattern checking - image generation models typically don't support thinking
        if "image" in model_id.lower() and (
            "generation" in model_id.lower() or "preview" in model_id.lower()
        ):
            return False

        # By default, assume models support thinking
        return True

    def _check_thinking_level_support(self, model_id: str) -> bool:
        """
        Check if a model supports the thinking_level parameter.

        Gemini 3 models support thinking_level and should NOT use thinking_budget.
        Other models (like Gemini 2.5) use thinking_budget instead.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model supports thinking_level, False otherwise
        """
        # Gemini 3 models support thinking_level (not thinking_budget)
        gemini_3_patterns = [
            "gemini-3-",
        ]

        model_lower = model_id.lower()
        for pattern in gemini_3_patterns:
            if pattern in model_lower:
                return True

        return False

    def validate_thinking_level(self, level: str) -> str | None:
        """
        Validate and normalize the thinking level value.

        Args:
            level: The thinking level string to validate

        Returns:
            Normalized level string ('low', 'high') or None if invalid/empty
        """
        if not level:
            return None

        normalized = level.strip().lower()
        valid_levels = ["low", "high"]

        if normalized in valid_levels:
            return normalized

        self.log.warning(
            f"Invalid thinking level '{level}'. Valid values are: {', '.join(valid_levels)}. "
            "Falling back to model default."
        )
        return None

    def validate_thinking_budget(self, budget: int) -> int:
        """
        Validate and normalize the thinking budget value.

        Args:
            budget: The thinking budget integer to validate

        Returns:
            Validated budget: -1 for dynamic, 0 to disable, or 1-32768 for fixed limit
        """
        # -1 means dynamic thinking (let the model decide)
        if budget == -1:
            return -1

        # 0 means disable thinking
        if budget == 0:
            return 0

        # Validate positive range (1-32768)
        if budget > 0:
            if budget > 32768:
                self.log.warning(
                    f"Thinking budget {budget} exceeds maximum of 32768. Clamping to 32768."
                )
                return 32768
            return budget

        # Negative values (except -1) are invalid, treat as -1 (dynamic)
        self.log.warning(
            f"Invalid thinking budget {budget}. Only -1 (dynamic), 0 (disabled), or 1-32768 are valid. "
            "Falling back to dynamic thinking."
        )
        return -1

    def validate_aspect_ratio(self, aspect_ratio: str) -> str | None:
        """
        Validate and normalize the aspect ratio value.

        Args:
            aspect_ratio: The aspect ratio string to validate

        Returns:
            Validated aspect ratio string, None for default, or '1:1' as fallback for invalid values
        """
        from .constants import ASPECT_RATIO_OPTIONS

        if not aspect_ratio or aspect_ratio == "default":
            self.log.debug("Using default aspect ratio (None)")
            return None

        normalized = aspect_ratio.strip()
        valid_ratios = [r for r in ASPECT_RATIO_OPTIONS if r != "default"]

        if normalized in valid_ratios:
            return normalized

        self.log.warning(
            f"Invalid aspect ratio '{aspect_ratio}'. Valid values are: {', '.join(valid_ratios)}. "
            "Using default '1:1'."
        )
        return "1:1"

    def validate_resolution(self, resolution: str) -> str | None:
        """
        Validate and normalize the resolution value.

        Args:
            resolution: The resolution string to validate

        Returns:
            Validated resolution string, None for default, or '2K' as fallback for invalid values
        """
        from .constants import RESOLUTION_OPTIONS

        if not resolution or resolution.lower() == "default":
            self.log.debug("Using default resolution (None)")
            return None

        normalized = resolution.strip().upper()
        valid_resolutions = [r for r in RESOLUTION_OPTIONS if r.lower() != "default"]

        if normalized in valid_resolutions:
            return normalized

        self.log.warning(
            f"Invalid resolution '{resolution}'. Valid values are: {', '.join(valid_resolutions)}. "
            "Using default '2K'."
        )
        return "2K"

    def prepare_model_id(self, model_id: str) -> str:
        """
        Prepare and validate the model ID for use with the API.

        Args:
            model_id: The original model ID from the user

        Returns:
            Properly formatted model ID

        Raises:
            ValueError: If the model ID is invalid or unsupported
        """
        original_model_id = model_id
        model_id = strip_prefix(model_id)

        # If the model ID doesn't look like a Gemini model, try to find it by name
        if not model_id.startswith("gemini-"):
            models_list = self.get_google_models()
            found_model = next(
                (m["id"] for m in models_list if m["name"] == original_model_id), None
            )
            if found_model and found_model.startswith("gemini-"):
                model_id = found_model
                self.log.debug(
                    f"Mapped model name '{original_model_id}' to model ID '{model_id}'"
                )
            else:
                # If we still don't have a valid ID, raise an error
                if not model_id.startswith("gemini-"):
                    self.log.error(
                        f"Invalid or unsupported model ID '{original_model_id}'"
                    )
                    raise ValueError(
                        f"Invalid or unsupported Google model ID or name: '{original_model_id}'"
                    )

        return model_id
