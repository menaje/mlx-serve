"""Images API router - OpenAI compatible image generation."""

import asyncio
import base64
import io
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])

# Temp directory for serving images via URL
_TEMP_IMAGE_DIR = Path("/tmp/mlx-serve-images")
_TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Image TTL (1 hour)
_IMAGE_TTL_SECONDS = 3600


def _cleanup_old_images() -> None:
    """Remove images older than TTL."""
    try:
        now = time.time()
        for img_path in _TEMP_IMAGE_DIR.iterdir():
            if img_path.is_file():
                age = now - img_path.stat().st_mtime
                if age > _IMAGE_TTL_SECONDS:
                    try:
                        img_path.unlink()
                        logger.debug(f"Cleaned up old image: {img_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old image {img_path}: {e}")
    except Exception as e:
        logger.warning(f"Image cleanup error: {e}")


def _start_cleanup_thread() -> None:
    """Start background thread for periodic image cleanup."""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Run every 5 minutes
            _cleanup_old_images()

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("Image cleanup thread started")


# Start cleanup thread when module loads
_start_cleanup_thread()


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image generation request."""

    model: str = Field(default="flux-schnell", description="Image generation model")
    prompt: str = Field(..., description="Text description of the image")
    n: int = Field(default=1, ge=1, le=1, description="Number of images (only 1 supported)")
    size: str = Field(
        default="1024x1024",
        description="Image size (width x height)",
    )
    quality: Literal["standard", "hd"] = Field(
        default="standard",
        description="Image quality",
    )
    response_format: Literal["url", "b64_json"] = Field(
        default="b64_json",
        description="Response format",
    )
    style: Literal["vivid", "natural"] | None = Field(
        default=None,
        description="Style of the generated image",
    )
    background: Literal["transparent", "opaque"] | None = Field(
        default=None,
        description="Background style (transparent requires PNG output)",
    )
    output_format: Literal["png", "jpeg", "webp"] | None = Field(
        default=None,
        description="Output image format",
    )
    user: str | None = None


class ImageData(BaseModel):
    """Generated image data."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    """OpenAI-compatible image generation response."""

    created: int
    data: list[ImageData]


def _parse_size(size: str) -> tuple[int, int]:
    """Parse size string to width and height.

    Raises:
        ValueError: If size format is invalid
    """
    try:
        width_str, height_str = size.lower().split("x")
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        if width > 4096 or height > 4096:
            raise ValueError("Dimensions cannot exceed 4096")
        return width, height
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid size format '{size}'. Expected format: WIDTHxHEIGHT (e.g., 1024x1024)") from e


async def _generate_image(
    model,
    prompt: str,
    width: int,
    height: int,
    num_steps: int = 4,
    guidance: float = 3.5,
    output_format: str = "png",
) -> bytes:
    """Generate image from prompt."""
    loop = asyncio.get_running_loop()

    def _generate():
        # mflux Flux1 model generate
        image = model.generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance=guidance,
        )
        return image

    image = await loop.run_in_executor(None, _generate)

    # Convert PIL Image to requested format
    def _to_bytes():
        buffer = io.BytesIO()
        # Map format names
        format_map = {
            "png": "PNG",
            "jpeg": "JPEG",
            "jpg": "JPEG",
            "webp": "WEBP",
        }
        img_format = format_map.get(output_format.lower(), "PNG")

        # For JPEG, need to convert RGBA to RGB
        if img_format == "JPEG" and image.mode == "RGBA":
            rgb_image = image.convert("RGB")
            rgb_image.save(buffer, format=img_format, quality=95)
        else:
            image.save(buffer, format=img_format)

        buffer.seek(0)
        return buffer.read()

    return await loop.run_in_executor(None, _to_bytes)


@router.get("/v1/images/{image_id}")
async def get_image(image_id: str):
    """Serve a generated image by ID."""
    # Try different extensions
    for ext in ["png", "jpeg", "webp"]:
        image_path = _TEMP_IMAGE_DIR / f"{image_id}.{ext}"
        if image_path.exists():
            media_types = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }
            return FileResponse(image_path, media_type=media_types[ext])

    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Image '{image_id}' not found or expired",
                "type": "invalid_request_error",
                "code": "image_not_found",
            }
        },
    )


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: ImageGenerationRequest, http_request: Request):
    """Generate an image from a text prompt.

    Uses FLUX.1 models via mflux for high-quality image generation.
    """
    try:
        model = model_manager.get_image_gen_model(request.model)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Image generation model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        ) from e

    # Parse size
    try:
        width, height = _parse_size(request.size)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "invalid_size",
                }
            },
        ) from e

    # Adjust steps based on quality
    num_steps = 4 if request.quality == "standard" else 8

    # Determine output format (default to png)
    output_format = request.output_format or "png"

    try:
        image_bytes = await _generate_image(
            model=model,
            prompt=request.prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            output_format=output_format,
        )

        if request.response_format == "url":
            # Save image and return URL with secure random ID
            image_id = uuid.uuid4().hex
            image_path = _TEMP_IMAGE_DIR / f"{image_id}.{output_format}"
            image_path.write_bytes(image_bytes)

            # Build URL based on request
            base_url = str(http_request.base_url).rstrip("/")
            image_url = f"{base_url}/v1/images/{image_id}"

            return ImageGenerationResponse(
                created=int(time.time()),
                data=[
                    ImageData(
                        url=image_url,
                        revised_prompt=request.prompt,
                    )
                ],
            )
        else:
            # Return base64 encoded image
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            return ImageGenerationResponse(
                created=int(time.time()),
                data=[
                    ImageData(
                        b64_json=b64_image,
                        revised_prompt=request.prompt,
                    )
                ],
            )

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Image generation failed: {str(e)}",
                    "type": "server_error",
                    "code": "generation_failed",
                }
            },
        ) from e
