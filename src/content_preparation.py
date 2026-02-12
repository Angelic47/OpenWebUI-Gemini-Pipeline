"""Content preparation and processing for the Gemini Pipeline."""

import re
import base64
import aiofiles
import logging
from typing import Any
from pathlib import Path


class ContentPreparator:
    """Prepares and processes message content for the Gemini API."""

    def __init__(self, pipe, image_processor):
        self.pipe = pipe
        self.image_processor = image_processor
        self.log = logging.getLogger("google_ai.pipe")

    def combine_system_prompts(self, user_system_prompt: str | None) -> str | None:
        """
        Combine default system prompt with user-defined system prompt.

        If DEFAULT_SYSTEM_PROMPT is set and user_system_prompt exists,
        the default is prepended to the user's prompt.
        If only DEFAULT_SYSTEM_PROMPT is set, it is used as the system prompt.
        If only user_system_prompt is set, it is used as-is.

        Args:
            user_system_prompt: The user-defined system prompt from messages (may be None)

        Returns:
            Combined system prompt or None if neither is set
        """
        default_prompt = self.pipe.valves.DEFAULT_SYSTEM_PROMPT.strip()
        user_prompt = user_system_prompt.strip() if user_system_prompt else ""

        if default_prompt and user_prompt:
            combined = f"{default_prompt}\n\n{user_prompt}"
            self.log.debug(
                f"Combined system prompts: default ({len(default_prompt)} chars) + "
                f"user ({len(user_prompt)} chars) = {len(combined)} chars"
            )
            return combined
        elif default_prompt:
            self.log.debug(f"Using default system prompt ({len(default_prompt)} chars)")
            return default_prompt
        elif user_prompt:
            return user_prompt
        return None

    def prepare_content(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Prepare messages content for the API and extract system message if present.

        Args:
            messages: List of message objects from the request

        Returns:
            Tuple of (prepared content list, system message string or None)
        """
        # Extract user-defined system message
        user_system_message = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Combine with default system prompt if configured
        system_message = self.combine_system_prompts(user_system_message)

        # Prepare contents for the API
        contents = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue  # Skip system messages, handled separately

            content = message.get("content", "")
            parts = []

            # Handle different content types
            if isinstance(content, list):  # Multimodal content
                parts.extend(self._process_multimodal_content(content))
            elif isinstance(content, str):  # Plain text content
                # remove thinking markers if present
                # e.g.: <details><summary>Thought (35s)</summary>\n...</details>
                if role == "assistant":
                    content = re.sub(
                        r"<details>\s*<summary>Thought \(\d+s\)</summary>.*?</details>",
                        "",
                        content,
                        flags=re.DOTALL,
                    ).strip()
                parts.append({"text": content})
            else:
                self.log.warning(f"Unsupported message content type: {type(content)}")
                continue  # Skip unsupported content

            # Map roles: 'assistant' -> 'model', 'user' -> 'user'
            api_role = "model" if role == "assistant" else "user"
            if parts:  # Only add if there are parts
                contents.append({"role": api_role, "parts": parts})

        return contents, system_message

    def _process_multimodal_content(
        self, content_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Process multimodal content (text and images).

        Args:
            content_list: List of content items

        Returns:
            List of processed parts for the Gemini API
        """
        parts = []

        for item in content_list:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")

                if image_url.startswith("data:image"):
                    # Handle base64 encoded image data with optimization
                    try:
                        # Optimize the image before processing
                        optimized_image = self.image_processor.optimize_image_for_api(image_url)
                        header, encoded = optimized_image.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]

                        # Basic validation for image types
                        if mime_type not in [
                            "image/jpeg",
                            "image/png",
                            "image/webp",
                            "image/heic",
                            "image/heif",
                        ]:
                            self.log.warning(
                                f"Unsupported image mime type: {mime_type}"
                            )
                            parts.append(
                                {"text": f"[Image type {mime_type} not supported]"}
                            )
                            continue

                        # Check if the encoded data is too large
                        if len(encoded) > 15 * 1024 * 1024:  # 15MB limit for base64
                            self.log.warning(
                                f"Image data too large: {len(encoded)} characters"
                            )
                            parts.append(
                                {
                                    "text": "[Image too large for processing - please use a smaller image]"
                                }
                            )
                            continue

                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded,
                                }
                            }
                        )
                    except Exception as img_ex:
                        self.log.exception(f"Could not parse image data URL: {img_ex}")
                        parts.append({"text": "[Image data could not be processed]"})
                else:
                    # Gemini API doesn't directly support image URLs
                    self.log.warning(f"Direct image URLs not supported: {image_url}")
                    parts.append({"text": f"[Image URL not processed: {image_url}]"})

        return parts

    async def extract_images_from_message(
        self,
        message: dict[str, Any],
        *,
        stats_list: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Extract prompt text and ALL images from a single user message.

        This replaces the previous single-image _find_image logic for image-capable
        models so that multi-image prompts are respected.

        Returns:
            (prompt_text, image_parts)
                prompt_text: concatenated text content (may be empty)
                image_parts: list of {"inline_data": {"mime_type", "data"}} dicts
        """
        content = message.get("content", "")
        text_segments: list[str] = []
        image_parts: list[dict[str, Any]] = []

        # Helper to process a data URL or fetched file and append inline_data
        def _add_image(data_url: str):
            try:
                optimized = self.image_processor.optimize_image_for_api(data_url, stats_list)
                header, b64 = optimized.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                image_parts.append({"inline_data": {"mime_type": mime, "data": b64}})
            except Exception as e:  # pragma: no cover - defensive
                self.log.warning(f"Skipping image (parse failure): {e}")

        # Regex to extract markdown image references
        md_pattern = re.compile(
            r"!\[([^\]]*)\]\((data:image/[^)]+|/files/[^)]+|/api/v1/files/[^)]+)"
        )

        # Structured multimodal array
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    text_segments.append(txt)
                    # Also parse any markdown images embedded in the text
                    for match in md_pattern.finditer(txt):
                        url = match.group(1)
                        if url.startswith("data:"):
                            _add_image(url)
                        else:
                            b64 = await self._fetch_file_as_base64(url)
                            if b64:
                                _add_image(b64)
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        _add_image(url)
                    elif "/files" in url or "/api/v1/files" in url:
                        b64 = await self._fetch_file_as_base64(url)
                        if b64:
                            _add_image(b64)
        # Plain string message (may include markdown images)
        elif isinstance(content, str):
            text_segments.append(content)
            for match in md_pattern.finditer(content):
                url = match.group(1)
                if url.startswith("data:"):
                    _add_image(url)
                else:
                    b64 = await self._fetch_file_as_base64(url)
                    if b64:
                        _add_image(b64)
        else:
            self.log.debug(
                f"Unsupported content type for image extraction: {type(content)}"
            )

        prompt_text = " ".join(s.strip() for s in text_segments if s.strip())
        return prompt_text, image_parts

    async def _fetch_file_as_base64(self, file_url: str) -> str | None:
        """
        Fetch a file from Open WebUI's file system and convert to base64.

        Args:
            file_url: File URL from Open WebUI

        Returns:
            Base64 encoded file data or None if file not found
        """
        try:
            if "/api/v1/files" in file_url:
                fid = file_url.split("/api/v1/files")[-1].split("/")[0].split("?")[0]
            else:
                fid = file_url.split("/files")[-1].split("/")[0].split("?")[0]

            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            file_obj = Files.get_file_by_id(fid)
            if file_obj and file_obj.path:
                file_path = Storage.get_file(file_obj.path)
                file_path = Path(file_path)
                if file_path.is_file():
                    async with aiofiles.open(file_path, "rb") as fp:
                        raw = await fp.read()
                    enc = base64.b64encode(raw).decode()
                    mime = file_obj.meta.get("content_type", "image/png")
                    return f"data:{mime};base64,{enc}"
        except Exception as e:
            self.log.warning(f"Could not fetch file {file_url}: {e}")
        return None

    async def gather_history_images(
        self,
        messages: list[dict[str, Any]],
        last_user_msg: dict[str, Any],
        optimization_stats: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Gather images from conversation history."""
        history_images: list[dict[str, Any]] = []
        for msg in messages:
            if msg is last_user_msg:
                continue
            if msg.get("role") not in {"user", "assistant"}:
                continue
            _p, parts = await self.extract_images_from_message(
                msg, stats_list=optimization_stats
            )
            if parts:
                history_images.extend(parts)
        return history_images

    async def build_image_generation_contents(
        self,
        messages: list[dict[str, Any]],
        __event_emitter__,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Construct the contents payload for image-capable models.

        Returns tuple (contents, system_instruction) where system_instruction is extracted from system messages.
        """
        # Extract user-defined system instruction first
        user_system_instruction = next(
            (msg["content"] for msg in messages if msg.get("role") == "system"),
            None,
        )

        # Combine with default system prompt if configured
        system_instruction = self.combine_system_prompts(user_system_instruction)

        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_msg:
            raise ValueError("No user message found")

        optimization_stats: list[dict[str, Any]] = []
        history_images = await self.gather_history_images(
            messages, last_user_msg, optimization_stats
        )
        prompt, current_images = await self.extract_images_from_message(
            last_user_msg, stats_list=optimization_stats
        )

        # Deduplicate
        history_images = self.image_processor.deduplicate_images(history_images)
        current_images = self.image_processor.deduplicate_images(current_images)

        combined, reused_flags = self.image_processor.apply_order_and_limit(
            history_images, current_images
        )

        if not prompt and not combined:
            raise ValueError("No prompt or images provided")
        if not prompt and combined:
            prompt = "Analyze and describe the provided images."

        # Build ordered stats aligned with combined list
        ordered_stats: list[dict[str, Any]] = []
        if optimization_stats:
            # Build map from final_hash -> stat (first wins)
            hash_map: dict[str, dict[str, Any]] = {}
            for s in optimization_stats:
                fh = s.get("final_hash")
                if fh and fh not in hash_map:
                    hash_map[fh] = s
            for part in combined:
                try:
                    fh = hashlib.sha256(
                        part["inline_data"]["data"].encode()
                    ).hexdigest()
                    ordered_stats.append(hash_map.get(fh) or {})
                except Exception:
                    ordered_stats.append({})
        # Emit stats AFTER final ordering so labels match
        await self._emit_image_stats(
            ordered_stats,
            reused_flags,
            self.pipe.valves.IMAGE_HISTORY_MAX_REFERENCES,
            __event_emitter__,
        )

        # Emit mapping
        if combined:
            mapping = [
                {
                    "index": i + 1,
                    "label": (
                        f"Image {i + 1}" if self.pipe.valves.IMAGE_ADD_LABELS else str(i + 1)
                    ),
                    "reused": reused_flags[i],
                    "origin": "history" if reused_flags[i] else "current",
                }
                for i in range(len(combined))
            ]
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_reference_map",
                        "description": f"{len(combined)} image(s) included (limit: {self.pipe.valves.IMAGE_HISTORY_MAX_REFERENCES}).",
                        "images": mapping,
                        "done": True,
                    },
                }
            )

        # Build parts
        parts: list[dict[str, Any]] = []

        # For image generation models, prepend system instruction to the prompt
        # since system_instruction parameter may not be supported
        final_prompt = prompt
        if system_instruction and prompt:
            final_prompt = f"{system_instruction}\n\n{prompt}"
            self.log.debug(
                f"Prepended system instruction to prompt for image generation. "
                f"System instruction length: {len(system_instruction)}, "
                f"Original prompt length: {len(prompt)}, "
                f"Final prompt length: {len(final_prompt)}"
            )
        elif system_instruction and not prompt:
            final_prompt = system_instruction
            self.log.debug(
                f"Using system instruction as prompt for image generation "
                f"(length: {len(system_instruction)})"
            )

        if final_prompt:
            parts.append({"text": final_prompt})
        if self.pipe.valves.IMAGE_ADD_LABELS:
            for idx, part in enumerate(combined, start=1):
                parts.append({"text": f"[Image {idx}]"})
                parts.append(part)
        else:
            parts.extend(combined)

        self.log.debug(
            f"Image-capable payload: history={len(history_images)} current={len(current_images)} used={len(combined)} limit={self.pipe.valves.IMAGE_HISTORY_MAX_REFERENCES} history_first={self.pipe.valves.IMAGE_HISTORY_FIRST}"
        )
        # Return None for system_instruction since we've incorporated it into the prompt
        return [{"role": "user", "parts": parts}], None

    async def _emit_image_stats(
        self,
        ordered_stats: list[dict[str, Any]],
        reused_flags: list[bool],
        total_limit: int,
        __event_emitter__,
    ) -> None:
        """
        Emit per-image optimization stats aligned with final combined order.

        ordered_stats: stats list in the exact order images will be sent (same length as combined image list)
        reused_flags: parallel list indicating whether image originated from history
        """
        if not ordered_stats:
            return
        for idx, stat in enumerate(ordered_stats, start=1):
            reused = reused_flags[idx - 1] if idx - 1 < len(reused_flags) else False
            stat_copy = dict(stat) if stat else {}
            stat_copy.update({"index": idx, "reused": reused})
            if stat and stat.get("original_size_mb") is not None:
                desc = f"Image {idx}: {stat['original_size_mb']:.2f}MB -> {stat['final_size_mb']:.2f}MB"
                if stat.get("quality") is not None:
                    desc += f" (Q{stat['quality']})"
            else:
                desc = f"Image {idx} (no metrics)"
            reasons = stat.get("reasons") if stat else None
            if reasons:
                desc += "    + " + ", ".join(reasons[:3])
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_optimization",
                        "description": desc,
                        "index": idx,
                        "done": False,
                        "details": stat_copy,
                    },
                }
            )
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "action": "image_optimization",
                    "description": f"{len(ordered_stats)} image(s) processed (limit: {total_limit}).",
                    "done": True,
                },
            }
        )
