"""Response handling for the Gemini Pipeline."""

import time
import logging
from typing import Any, AsyncIterator, Callable
from google.genai import types


class ResponseHandler:
    """Handles streaming and non-streaming responses from Gemini API."""

    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")

    @staticmethod
    def format_grounding_chunks_as_sources(
        grounding_chunks: list[types.GroundingChunk],
    ) -> list[dict]:
        """Format grounding chunks as sources for the UI."""
        formatted_sources = []
        for chunk in grounding_chunks:
            if hasattr(chunk, "retrieved_context") and chunk.retrieved_context:
                context = chunk.retrieved_context
                formatted_sources.append(
                    {
                        "source": {
                            "name": getattr(context, "title", None) or "Document",
                            "type": "vertex_ai_search",
                            "uri": getattr(context, "uri", None),
                        },
                        "document": [getattr(context, "chunk_text", None) or ""],
                        "metadata": [
                            {"source": getattr(context, "title", None) or "Document"}
                        ],
                    }
                )
            elif hasattr(chunk, "web") and chunk.web:
                context = chunk.web
                uri = context.uri
                title = context.title or "Source"

                formatted_sources.append(
                    {
                        "source": {
                            "name": title,
                            "type": "web_search_results",
                            "url": uri,
                        },
                        "document": ["Click the link to view the content."],
                        "metadata": [{"source": title}],
                    }
                )
        return formatted_sources

    async def process_grounding_metadata(
        self,
        grounding_metadata_list: list[types.GroundingMetadata],
        text: str,
        __event_emitter__: Callable,
    ) -> str:
        """Process and emit grounding metadata events."""
        grounding_chunks = []
        web_search_queries = []
        grounding_supports = []

        for metadata in grounding_metadata_list:
            if metadata.grounding_chunks:
                grounding_chunks.extend(metadata.grounding_chunks)
            if metadata.web_search_queries:
                web_search_queries.extend(metadata.web_search_queries)
            if metadata.grounding_supports:
                grounding_supports.extend(metadata.grounding_supports)

        # Add sources to the response
        if grounding_chunks:
            sources = self.format_grounding_chunks_as_sources(grounding_chunks)
            await __event_emitter__(
                {"type": "chat:completion", "data": {"sources": sources}}
            )

        # Add status specifying google queries used for grounding
        if web_search_queries:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "This response was grounded with Google Search",
                        "urls": [
                            f"https://www.google.com/search?q={query}"
                            for query in web_search_queries
                        ],
                    },
                }
            )

        # Add citations in the text body
        replaced_text: str | None = None
        if grounding_supports:
            # Citation indexes are in bytes
            ENCODING = "utf-8"
            text_bytes = text.encode(ENCODING)
            last_byte_index = 0
            cited_chunks = []

            for support in grounding_supports:
                cited_chunks.append(
                    text_bytes[last_byte_index : support.segment.end_index].decode(
                        ENCODING
                    )
                )

                # Generate and append citations (e.g., [1][2])
                footnotes = "".join(
                    [f"[{i + 1}]" for i in support.grounding_chunk_indices]
                )
                cited_chunks.append(f" {footnotes}")

                # Update index for the next segment
                last_byte_index = support.segment.end_index

            # Append any remaining text after the last citation
            if last_byte_index < len(text_bytes):
                cited_chunks.append(text_bytes[last_byte_index:].decode(ENCODING))

            replaced_text = "".join(cited_chunks)

        return replaced_text if replaced_text is not None else text

    async def handle_streaming_response(
        self,
        response_iterator: Any,
        __event_emitter__: Callable,
        __request__: Any = None,
        __user__: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Handle streaming response from Gemini API.

        Args:
            response_iterator: Iterator from generate_content
            __event_emitter__: Event emitter for status updates

        Returns:
            Generator yielding text chunks
        """

        async def emit_chat_event(event_type: str, data: dict[str, Any]) -> None:
            if not __event_emitter__:
                return
            try:
                await __event_emitter__({"type": event_type, "data": data})
            except Exception as emit_error:  # pragma: no cover - defensive
                self.log.warning(f"Failed to emit {event_type} event: {emit_error}")

        await emit_chat_event("chat:start", {"role": "assistant"})

        grounding_metadata_list = []
        # Accumulate content separately for answer and thoughts
        answer_chunks: list[str] = []
        thought_chunks: list[str] = []
        thinking_started_at: float | None = None

        try:
            async for chunk in response_iterator:
                # Check for safety feedback or empty chunks
                if not chunk.candidates:
                    # Check prompt feedback
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        block_reason = chunk.prompt_feedback.block_reason.name
                        message = f"[Blocked due to Prompt Safety: {block_reason}]"
                        await emit_chat_event(
                            "chat:finish",
                            {
                                "role": "assistant",
                                "content": message,
                                "done": True,
                                "error": True,
                            },
                        )
                        yield message
                    else:
                        message = "[Blocked by safety settings]"
                        await emit_chat_event(
                            "chat:finish",
                            {
                                "role": "assistant",
                                "content": message,
                                "done": True,
                                "error": True,
                            },
                        )
                        yield message
                    return  # Stop generation

                if chunk.candidates[0].grounding_metadata:
                    grounding_metadata_list.append(
                        chunk.candidates[0].grounding_metadata
                    )
                # Prefer fine-grained parts to split thoughts vs. normal text
                parts = []
                try:
                    parts = chunk.candidates[0].content.parts or []
                except Exception as parts_error:
                    # Fallback: use aggregated text if parts aren't accessible
                    self.log.warning(f"Failed to access content parts: {parts_error}")
                    if hasattr(chunk, "text") and chunk.text:
                        answer_chunks.append(chunk.text)
                        await __event_emitter__(
                            {
                                "type": "chat:message:delta",
                                "data": {
                                    "role": "assistant",
                                    "content": chunk.text,
                                },
                            }
                        )
                    continue

                for part in parts:
                    try:
                        # Thought parts (internal reasoning)
                        if getattr(part, "thought", False) and getattr(
                            part, "text", None
                        ):
                            if thinking_started_at is None:
                                thinking_started_at = time.time()
                            thought_chunks.append(part.text)
                            # Emit a live preview of what is currently being thought
                            preview = part.text.replace("\n", " ").strip()
                            MAX_PREVIEW = 120
                            if len(preview) > MAX_PREVIEW:
                                preview = preview[:MAX_PREVIEW].rstrip() + "…"
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "action": "thinking",
                                        "description": f"Thinking… {preview}",
                                        "done": False,
                                        "hidden": False,
                                    },
                                }
                            )

                        # Regular answer text
                        elif getattr(part, "text", None):
                            answer_chunks.append(part.text)
                            await __event_emitter__(
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "role": "assistant",
                                        "content": part.text,
                                    },
                                }
                            )
                    except Exception as part_error:
                        # Log part processing errors but continue with the stream
                        self.log.warning(f"Error processing content part: {part_error}")
                        continue

            # After processing all chunks, handle grounding data
            final_answer_text = "".join(answer_chunks)
            if grounding_metadata_list and __event_emitter__:
                cited = await self.process_grounding_metadata(
                    grounding_metadata_list,
                    final_answer_text,
                    __event_emitter__,
                )
                final_answer_text = cited or final_answer_text

            final_content = final_answer_text
            details_block: str | None = None

            if thought_chunks:
                duration_s = int(
                    max(0, time.time() - (thinking_started_at or time.time()))
                )
                # Format each line with "> " for blockquote while preserving formatting
                thought_content = "".join(thought_chunks).strip()
                quoted_lines = []
                for line in thought_content.split("\n"):
                    quoted_lines.append(f"> {line}")
                quoted_content = "\n".join(quoted_lines)

                details_block = f"""<details>
<summary>Thought ({duration_s}s)</summary>

{quoted_content}

</details>""".strip()
                final_content = f"{details_block}{final_answer_text}"

            if not final_content:
                final_content = ""

            # Ensure downstream consumers (UI, TTS) receive the complete response once streaming ends.
            await emit_chat_event(
                "replace", {"role": "assistant", "content": final_content}
            )
            await emit_chat_event(
                "chat:message",
                {"role": "assistant", "content": final_content, "done": True},
            )

            if thought_chunks:
                # Clear the thinking status without a summary in the status emitter
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"action": "thinking", "done": True, "hidden": True},
                    }
                )

            await emit_chat_event(
                "chat:finish",
                {"role": "assistant", "content": final_content, "done": True},
            )

            # Yield final content to ensure the async iterator completes properly.
            # This ensures the response is persisted even if the user navigates away.
            yield final_content

        except Exception as e:
            self.log.exception(f"Error during streaming: {e}")
            # Check if it's a chunk size error and provide specific guidance
            error_msg = str(e).lower()
            if "chunk too big" in error_msg or "chunk size" in error_msg:
                message = "Error: Image too large for processing. Please try with a smaller image (max 15 MB recommended) or reduce image quality."
            elif "quota" in error_msg or "rate limit" in error_msg:
                message = "Error: API quota exceeded. Please try again later."
            else:
                message = f"Error during streaming: {e}"
            await emit_chat_event(
                "chat:finish",
                {
                    "role": "assistant",
                    "content": message,
                    "done": True,
                    "error": True,
                },
            )
            yield message

    def get_safety_block_message(self, response: Any) -> str | None:
        """Check for safety blocks and return appropriate message."""
        # Check prompt feedback
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[Blocked due to Prompt Safety: {response.prompt_feedback.block_reason.name}]"

        # Check candidates
        if not response.candidates:
            return "[Blocked by safety settings or no candidates generated]"

        # Check candidate finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.SAFETY:
            blocking_rating = next(
                (r for r in candidate.safety_ratings if r.blocked), None
            )
            reason = f" ({blocking_rating.category.name})" if blocking_rating else ""
            return f"[Blocked by safety settings{reason}]"
        elif candidate.finish_reason == types.FinishReason.PROHIBITED_CONTENT:
            return "[Content blocked due to prohibited content policy violation]"

        return None
