"""Images API router - OpenAI compatible image generation."""

import asyncio
import base64
import io
import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["images"])


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
    """Parse size string to width and height."""
    try:
        width, height = size.lower().split("x")
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 1024, 1024


async def _generate_image(
    model,
    prompt: str,
    width: int,
    height: int,
    num_steps: int = 4,
    guidance: float = 3.5,
) -> bytes:
    """Generate image from prompt."""
    loop = asyncio.get_event_loop()

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

    # Convert PIL Image to PNG bytes
    def _to_png():
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.read()

    return await loop.run_in_executor(None, _to_png)


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: ImageGenerationRequest):
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
    width, height = _parse_size(request.size)

    # Adjust steps based on quality
    num_steps = 4 if request.quality == "standard" else 8

    try:
        image_bytes = await _generate_image(
            model=model,
            prompt=request.prompt,
            width=width,
            height=height,
            num_steps=num_steps,
        )

        # Encode as base64
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
