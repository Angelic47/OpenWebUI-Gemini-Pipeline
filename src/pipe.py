"""Main Pipe class for the Gemini Pipeline."""

import os
import re
import time
import asyncio
import base64
import hashlib
import logging
import io
import uuid
from typing import List, Union, Optional, Dict, Any, Tuple, AsyncIterator, Callable

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError, APIError
from fastapi import Request
from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.users import Users
from pydantic import BaseModel, Field

from .encryption import EncryptedStr
from .constants import ASPECT_RATIO_OPTIONS, RESOLUTION_OPTIONS
from .utils import strip_prefix, sanitize_header_value
from .image_processing import ImageProcessor
from .model_management import ModelManager
from .content_preparation import ContentPreparator
from .generation_config import GenerationConfigurator
from .response_handlers import ResponseHandler
from .cache import CacheManager
from .image_upload import ImageUploader
from .mcp_patch import MCPPatcher


class Pipe:
    """
    Pipeline for interacting with Google Gemini models.
    """

    # User-overridable configuration valves
    class UserValves(BaseModel):
        """User-overridable configuration valves."""

        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )

    # Configuration valves for the pipeline
    class Valves(BaseModel):
        """Configuration valves for the pipeline."""

        BASE_URL: str = Field(
            default=os.getenv(
                "GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com"
            ),
            description="Base URL for the Google Generative AI API.",
        )
        GOOGLE_API_KEY: EncryptedStr = Field(
            default=os.getenv("GOOGLE_API_KEY", ""),
            description="API key for Google Generative AI (used if USE_VERTEX_AI is false).",
        )
        API_VERSION: str = Field(
            default=os.getenv("GOOGLE_API_VERSION", "v1alpha"),
            description="API version to use for Google Generative AI (e.g., v1alpha, v1beta, v1).",
        )
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("GOOGLE_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses (set false to force non-streaming mode).",
        )
        INCLUDE_THOUGHTS: bool = Field(
            default=os.getenv("GOOGLE_INCLUDE_THOUGHTS", "true").lower() == "true",
            description="Enable Gemini thoughts outputs (set false to disable).",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("GOOGLE_THINKING_BUDGET", -1)),
            description="""Thinking budget for Gemini 2.5 models (0=disabled, -1=dynamic, 1-32768=fixed token limit). 
            Not used for Gemini 3 models which use THINKING_LEVEL instead.""",
        )
        THINKING_LEVEL: str = Field(
            default=os.getenv("GOOGLE_THINKING_LEVEL", ""),
            description="""Thinking level for Gemini 3 models ('low' or 'high'). 
            Ignored for other models. Empty string means use model default.""",
        )
        USE_VERTEX_AI: bool = Field(
            default=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            description="Whether to use Google Cloud Vertex AI instead of the Google Generative AI API.",
        )
        VERTEX_PROJECT: str | None = Field(
            default=os.getenv("GOOGLE_CLOUD_PROJECT"),
            description="The Google Cloud project ID to use with Vertex AI.",
        )
        VERTEX_LOCATION: str = Field(
            default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            description="The Google Cloud region to use with Vertex AI.",
        )
        VERTEX_AI_RAG_STORE: str | None = Field(
            default=os.getenv("GOOGLE_VERTEX_AI_RAG_STORE"),
            description="Vertex AI RAG Store path for grounding (e.g., projects/PROJECT/locations/LOCATION/ragCorpora/DATA_STORE_ID). Only used when USE_VERTEX_AI is true.",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=os.getenv("GOOGLE_USE_PERMISSIVE_SAFETY", "false").lower()
            == "true",
            description="Use permissive safety settings for content generation.",
        )
        MODEL_CACHE_TTL: int = Field(
            default=int(os.getenv("GOOGLE_MODEL_CACHE_TTL", 600)),
            description="Time in seconds to cache the model list before refreshing",
        )
        RETRY_COUNT: int = Field(
            default=int(os.getenv("GOOGLE_RETRY_COUNT", 2)),
            description="Number of times to retry API calls on temporary failures",
        )
        DEFAULT_SYSTEM_PROMPT: str = Field(
            default=os.getenv("GOOGLE_DEFAULT_SYSTEM_PROMPT", ""),
            description="""Default system prompt applied to all chats. If a user-defined system prompt exists, 
            this is prepended to it. Leave empty to disable.""",
        )
        ENABLE_FORWARD_USER_INFO_HEADERS: bool = Field(
            default=os.getenv(
                "GOOGLE_ENABLE_FORWARD_USER_INFO_HEADERS", "false"
            ).lower()
            == "true",
            description="Whether to forward user information headers.",
        )

        # Image Processing Configuration
        IMAGE_GENERATION_ASPECT_RATIO: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_ASPECT_RATIO", "default"),
            description="Default aspect ratio for image generation.",
            json_schema_extra={"enum": ASPECT_RATIO_OPTIONS},
        )
        IMAGE_GENERATION_RESOLUTION: str = Field(
            default=os.getenv("GOOGLE_IMAGE_GENERATION_RESOLUTION", "default"),
            description="Default resolution for image generation.",
            json_schema_extra={"enum": RESOLUTION_OPTIONS},
        )
        IMAGE_MAX_SIZE_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_MAX_SIZE_MB", 15.0)),
            description="Maximum image size in MB before compression is applied",
        )
        IMAGE_MAX_DIMENSION: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_MAX_DIMENSION", 2048)),
            description="Maximum width or height in pixels before resizing",
        )
        IMAGE_COMPRESSION_QUALITY: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_COMPRESSION_QUALITY", 85)),
            description="JPEG compression quality (1-100, higher = better quality but larger size)",
        )
        IMAGE_ENABLE_OPTIMIZATION: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ENABLE_OPTIMIZATION", "true").lower()
            == "true",
            description="Enable intelligent image optimization for API compatibility",
        )
        IMAGE_PNG_COMPRESSION_THRESHOLD_MB: float = Field(
            default=float(os.getenv("GOOGLE_IMAGE_PNG_THRESHOLD_MB", 0.5)),
            description="PNG files above this size (MB) will be converted to JPEG for better compression",
        )
        IMAGE_HISTORY_MAX_REFERENCES: int = Field(
            default=int(os.getenv("GOOGLE_IMAGE_HISTORY_MAX_REFERENCES", 5)),
            description="Maximum total number of images (history + current message) to include in a generation call",
        )
        IMAGE_ADD_LABELS: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_ADD_LABELS", "true").lower() == "true",
            description="If true, add small text labels like [Image 1] before each image part so the model can reference them.",
        )
        IMAGE_DEDUP_HISTORY: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_DEDUP_HISTORY", "true").lower() == "true",
            description="If true, deduplicate identical images (by hash) when constructing history context",
        )
        IMAGE_HISTORY_FIRST: bool = Field(
            default=os.getenv("GOOGLE_IMAGE_HISTORY_FIRST", "true").lower() == "true",
            description="If true (default), history images precede current message images; if false, current images first.",
        )

    def __init__(self):
        """Initializes the Pipe instance and configures the genai library."""
        self.valves = self.Valves()
        self.name: str = "Google Gemini "

        # Setup logging
        self.log = logging.getLogger("google_ai.pipe")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

        # Initialize helper classes - pass self (Pipe instance) so they can access current valves
        self.image_processor = ImageProcessor(self)
        self.model_manager = ModelManager(self)
        self.content_preparator = ContentPreparator(self, self.image_processor)
        self.generation_configurator = GenerationConfigurator(self, self.model_manager)
        self.response_handler = ResponseHandler()
        self.image_uploader = ImageUploader()
        self.mcp_patcher = MCPPatcher()

    def _get_client(self) -> genai.Client:
        """Get configured genai client."""
        return self.model_manager._get_client()

    def get_google_models(self, force_refresh: bool = False) -> List[Dict[str, str]]:
        """Retrieve available Google models."""
        return self.model_manager.get_google_models(force_refresh)

    def pipes(self) -> List[Dict[str, str]]:
        """
        Returns a list of available Google Gemini models for the UI.
        """
        try:
            self.name = "Google Gemini "
            return self.get_google_models()
        except ValueError as e:
            self.log.error(f"Error during pipes listing (validation): {e}")
            return [{"id": "error", "name": str(e)}]
        except Exception as e:
            self.log.exception(
                f"An unexpected error occurred during pipes listing: {str(e)}"
            )
            return [{"id": "error", "name": f"An unexpected error occurred: {str(e)}"}]

    def _prepare_model_id(self, model_id: str) -> str:
        """Prepare and validate the model ID."""
        return self.model_manager.prepare_model_id(model_id)

    def _check_image_generation_support(self, model_id: str) -> bool:
        """Check if model supports image generation."""
        return self.model_manager._check_image_generation_support(model_id)

    def _configure_generation(
        self,
        body: Dict[str, Any],
        system_instruction: Optional[str],
        __metadata__: Dict[str, Any],
        __tools__: Optional[Dict[str, Any]] = None,
        __user__: Optional[Dict] = None,
        enable_image_generation: bool = False,
        model_id: str = "",
        client: Optional[Any] = None,
    ) -> types.GenerateContentConfig:
        """Configure generation parameters."""
        return self.generation_configurator.configure_generation(
            body, system_instruction, __metadata__, __tools__, __user__,
            enable_image_generation, model_id, client
        )

    async def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Retry a function with exponential backoff."""
        max_retries = self.valves.RETRY_COUNT
        retry_count = 0
        last_exception = None

        while retry_count <= max_retries:
            try:
                return await func(*args, **kwargs)
            except ServerError as e:
                retry_count += 1
                last_exception = e

                if retry_count <= max_retries:
                    wait_time = min(2**retry_count + (0.1 * retry_count), 10)
                    self.log.warning(
                        f"Temporary error from Google API: {e}. Retrying in {wait_time:.1f}s ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception:
                raise

        if last_exception is not None:
            raise last_exception

    async def pipe(
        self,
        body: Dict[str, Any],
        __metadata__: Dict[str, Any],
        __event_emitter__: Callable,
        __tools__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
        __user__: Optional[Dict] = None,
    ) -> Union[str, AsyncIterator[str]]:
        """
        Main method for sending requests to the Google Gemini endpoint.
        """
        # Apply MCP tool patch
        __tools__ = self.mcp_patcher.patch_mcp_tools(__metadata__, __tools__)

        # Setup logging for this request
        request_id = id(body)
        self.log.debug(f"Processing request: {request_id}")
        self.log.debug(f"User request body: {__user__}")
        self.user = Users.get_user_by_id(__user__["id"])

        try:
            # Parse and validate model ID
            model_id = body.get("model", "")
            try:
                model_id = self._prepare_model_id(model_id)
                self.log.debug(f"Using model: {model_id}")
            except ValueError as ve:
                return f"Model Error: {ve}"

            # Check if this model supports image generation
            supports_image_generation = self._check_image_generation_support(model_id)

            # Get stream flag
            stream = body.get("stream", False)
            if not self.valves.STREAMING_ENABLED:
                if stream:
                    self.log.debug("Streaming disabled via GOOGLE_STREAMING_ENABLED")
                stream = False
            messages = body.get("messages", [])

            # For image generation models, gather ALL images from the last user turn
            if supports_image_generation:
                try:
                    contents, system_instruction = await self.content_preparator.build_image_generation_contents(
                        messages, __event_emitter__
                    )
                    self.log.debug(
                        "Image generation mode: system instruction integrated into prompt"
                    )
                except ValueError as ve:
                    return f"Error: {ve}"
            else:
                contents, system_instruction = self.content_preparator.prepare_content(messages)
                if not contents:
                    return "Error: No valid message content found"
                self.log.debug(
                    f"Text generation mode: system instruction separate (value: {system_instruction})"
                )

            # Configure generation parameters and safety settings
            self.log.debug(f"Supports image generation: {supports_image_generation}")

            # Make the API call
            client = self._get_client()

            generation_config = self._configure_generation(
                body,
                system_instruction,
                __metadata__,
                __tools__,
                __user__,
                supports_image_generation,
                model_id,
                client,
            )

            if stream:
                # For image generation models, disable streaming to avoid chunk size issues
                if supports_image_generation:
                    self.log.debug(
                        "Disabling streaming for image generation model to avoid chunk size issues"
                    )
                    stream = False
                else:
                    try:
                        async def get_streaming_response():
                            return await client.aio.models.generate_content_stream(
                                model=model_id,
                                contents=contents,
                                config=generation_config,
                            )

                        response_iterator = await self._retry_with_backoff(
                            get_streaming_response
                        )
                        self.log.debug(f"Request {request_id}: Got streaming response")
                        return self.response_handler.handle_streaming_response(
                            response_iterator, __event_emitter__, __request__, __user__
                        )

                    except Exception as e:
                        self.log.exception(
                            f"Error in streaming request {request_id}: {e}"
                        )
                        return f"Error during streaming: {e}"

            # Non-streaming path
            if not stream or supports_image_generation:
                try:
                    async def get_response():
                        return await client.aio.models.generate_content(
                            model=model_id,
                            contents=contents,
                            config=generation_config,
                        )

                    start_ts = time.time()

                    if supports_image_generation:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing image request...",
                                    "done": False,
                                },
                            }
                        )

                    response = await self._retry_with_backoff(get_response)
                    self.log.debug(f"Request {request_id}: Got non-streaming response")

                    if supports_image_generation:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "image_processing",
                                    "description": "Processing complete",
                                    "done": True,
                                },
                            }
                        )

                    # Handle response
                    safety_message = self.response_handler.get_safety_block_message(response)
                    if safety_message:
                        return safety_message

                    candidate = response.candidates[0]
                    parts = getattr(getattr(candidate, "content", None), "parts", [])
                    if not parts:
                        return "[No content generated or unexpected response structure]"

                    answer_segments: list[str] = []
                    thought_segments: list[str] = []
                    generated_images: list[str] = []

                    for part in parts:
                        if getattr(part, "thought", False) and getattr(part, "text", None):
                            thought_segments.append(part.text)
                        elif getattr(part, "text", None):
                            answer_segments.append(part.text)
                        elif (
                            getattr(part, "inline_data", None)
                            and __request__
                            and __user__
                        ):
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data

                            self.log.debug(
                                f"Processing generated image: mime_type={mime_type}, data_type={type(image_data)}, data_length={len(image_data)}"
                            )

                            image_url = await self.image_uploader.upload_image_with_status(
                                image_data,
                                mime_type,
                                __request__,
                                __user__,
                                __event_emitter__,
                            )
                            generated_images.append(f"![Generated Image]({image_url})")

                        elif getattr(part, "inline_data", None):
                            mime_type = part.inline_data.mime_type
                            image_data = part.inline_data.data

                            if isinstance(image_data, bytes):
                                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
                            else:
                                image_data_b64 = str(image_data)

                            data_url = f"data:{mime_type};base64,{image_data_b64}"
                            generated_images.append(f"![Generated Image]({data_url})")

                    final_answer = "".join(answer_segments)

                    # Apply grounding
                    grounding_metadata_list = []
                    if getattr(candidate, "grounding_metadata", None):
                        grounding_metadata_list.append(candidate.grounding_metadata)
                    if grounding_metadata_list:
                        cited = await self.response_handler.process_grounding_metadata(
                            grounding_metadata_list,
                            final_answer,
                            __event_emitter__,
                        )
                        final_answer = cited or final_answer

                    # Combine all content
                    full_response = ""

                    if thought_segments:
                        duration_s = int(max(0, time.time() - start_ts))
                        thought_content = "".join(thought_segments).strip()
                        quoted_lines = []
                        for line in thought_content.split("\n"):
                            quoted_lines.append(f"> {line}")
                        quoted_content = "\n".join(quoted_lines)

                        details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()
                        full_response += details_block

                    full_response += final_answer

                    if generated_images:
                        if full_response:
                            full_response += "\n\n"
                        full_response += "\n\n".join(generated_images)

                    return full_response if full_response else "[No content generated]"

                except Exception as e:
                    self.log.exception(
                        f"Error in non-streaming request {request_id}: {e}"
                    )
                    return f"Error generating content: {e}"

        except (ClientError, ServerError, APIError) as api_error:
            error_type = type(api_error).__name__
            error_msg = f"{error_type}: {api_error}"
            self.log.error(error_msg)
            return error_msg

        except ValueError as ve:
            error_msg = f"Configuration error: {ve}"
            self.log.error(error_msg)
            return error_msg

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            self.log.exception(f"Unexpected error: {e}\n{error_trace}")
            return f"An error occurred while processing your request: {e}"
