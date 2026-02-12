"""Image upload handling for the Gemini Pipeline."""

import io
import uuid
import base64
import logging
from typing import Any, Callable
from fastapi import Request, UploadFile, BackgroundTasks
from starlette.datastructures import Headers
from open_webui.routers.files import upload_file
from open_webui.models.users import Users


class ImageUploader:
    """Handles uploading generated images to Open WebUI."""

    def __init__(self):
        self.log = logging.getLogger("google_ai.pipe")

    async def upload_image_with_status(
        self,
        image_data: Any,
        mime_type: str,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Callable,
    ) -> str:
        """
        Unified image upload method with status updates and fallback handling.

        Returns:
            URL to uploaded image or data URL fallback
        """
        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Uploading generated image to your library...",
                        "done": False,
                    },
                }
            )

            user = Users.get_user_by_id(__user__["id"])

            # Convert image data to base64 string if needed
            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            image_url = self._upload_image(
                __request__=__request__,
                user=user,
                image_data=image_data_b64,
                mime_type=mime_type,
            )

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Image uploaded successfully!",
                        "done": True,
                    },
                }
            )

            return image_url

        except Exception as e:
            self.log.warning(f"File upload failed, falling back to data URL: {e}")

            if isinstance(image_data, bytes):
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data_b64 = str(image_data)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "image_upload",
                        "description": "Using inline image (upload failed)",
                        "done": True,
                    },
                }
            )

            return f"data:{mime_type};base64,{image_data_b64}"

    def _upload_image(
        self, __request__: Request, user, image_data: str, mime_type: str
    ) -> str:
        """
        Upload generated image to Open WebUI's file system.
        Expects base64 encoded string input.

        Args:
            __request__: FastAPI request object
            user: User model object
            image_data: Base64 encoded image data string
            mime_type: MIME type of the image

        Returns:
            URL to the uploaded image or data URL fallback
        """
        try:
            self.log.debug(
                f"Processing image data, type: {type(image_data)}, length: {len(image_data)}"
            )

            # Decode base64 string to bytes
            try:
                decoded_data = base64.b64decode(image_data)
                self.log.debug(
                    f"Successfully decoded image data: {len(decoded_data)} bytes"
                )
            except Exception as decode_error:
                self.log.error(f"Failed to decode base64 data: {decode_error}")
                # Try to add padding if missing
                try:
                    missing_padding = len(image_data) % 4
                    if missing_padding:
                        image_data += "=" * (4 - missing_padding)
                    decoded_data = base64.b64decode(image_data)
                    self.log.debug(
                        f"Successfully decoded with padding: {len(decoded_data)} bytes"
                    )
                except Exception as second_decode_error:
                    self.log.error(f"Still failed to decode: {second_decode_error}")
                    return f"data:{mime_type};base64,{image_data}"

            bio = io.BytesIO(decoded_data)
            bio.seek(0)

            # Determine file extension
            extension = "png"
            if "jpeg" in mime_type or "jpg" in mime_type:
                extension = "jpg"
            elif "webp" in mime_type:
                extension = "webp"
            elif "gif" in mime_type:
                extension = "gif"

            # Create filename
            filename = f"gemini-generated-{uuid.uuid4().hex}.{extension}"

            # Upload with simple approach like reference
            up_obj = upload_file(
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=bio,
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,  # Matching reference - no heavy processing
                user=user,
                metadata={"mime_type": mime_type, "source": "gemini_image_generation"},
            )

            self.log.debug(
                f"Upload completed. File ID: {up_obj.id}, Decoded size: {len(decoded_data)} bytes"
            )

            # Generate URL using reference method
            return __request__.app.url_path_for("get_file_content_by_id", id=up_obj.id)

        except Exception as e:
            self.log.exception(f"Image upload failed, using data URL fallback: {e}")
            # Fallback to data URL if upload fails
            return f"data:{mime_type};base64,{image_data}"
