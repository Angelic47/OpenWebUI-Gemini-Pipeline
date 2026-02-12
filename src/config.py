"""Configuration valves for the Gemini Pipeline."""

import os
from pydantic import BaseModel, Field
from .encryption import EncryptedStr
from .constants import ASPECT_RATIO_OPTIONS, RESOLUTION_OPTIONS


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
