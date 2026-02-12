"""
title: Google Gemini Pipeline
author: owndev, olivier-lacroix
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/sponsors/owndev
version: 1.10.0
required_open_webui_version: 0.6.26
license: Apache License 2.0
description: Highly optimized Google Gemini pipeline with advanced image generation capabilities, intelligent compression, and streamlined processing workflows.
features:
  - Optimized asynchronous API calls for maximum performance
  - Intelligent model caching with configurable TTL
  - Streamlined dynamic model specification with automatic prefix handling
  - Smart streaming response handling with safety checks
  - Advanced multimodal input support (text and images)
  - Unified image generation and editing with Gemini 2.5 Flash Image Preview
  - Intelligent image optimization with size-aware compression algorithms
  - Automated image upload to Open WebUI with robust fallback support
  - Optimized text-to-image and image-to-image workflows
  - Non-streaming mode for image generation to prevent chunk overflow
  - Progressive status updates for optimal user experience
  - Consolidated error handling and comprehensive logging
  - Seamless Google Generative AI and Vertex AI integration
  - Advanced generation parameters (temperature, max tokens, etc.)
  - Configurable safety settings with environment variable support
  - Military-grade encrypted storage of sensitive API keys
  - Intelligent grounding with Google search integration
  - Vertex AI Search grounding for RAG
  - Native tool calling support with automatic signature management
  - URL context grounding for specified web pages
  - Unified image processing with consolidated helper methods
  - Optimized payload creation for image generation models
  - Configurable image processing parameters (size, quality, compression)
  - Flexible upload fallback options and optimization controls
  - Configurable thinking levels (low/high) for Gemini 3 models
  - Configurable thinking budgets (0-32768 tokens) for Gemini 2.5 models
  - Configurable image generation aspect ratio (1:1, 16:9, etc.) and resolution (1K, 2K, 4K)
"""

from .pipe import Pipe

__all__ = ["Pipe"]
