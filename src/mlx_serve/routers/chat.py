"""Chat completions API router - OpenAI compatible."""

import asyncio
import base64
import json
import logging
import time
import uuid
from io import BytesIO
from typing import Any, Literal
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# Request/Response Models
class FunctionDefinition(BaseModel):
    """Function definition for tool use."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """Function call in assistant message."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ImageUrl(BaseModel):
    """Image URL for vision models."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ContentPartText(BaseModel):
    """Text content part."""

    type: Literal["text"] = "text"
    text: str


class ContentPartImage(BaseModel):
    """Image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = ContentPartText | ContentPartImage


class ChatMessage(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model name to use")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    stream: bool = Field(default=False)
    stop: str | list[str] | None = Field(default=None)
    tools: list[ToolDefinition] | None = Field(default=None)
    tool_choice: str | dict | None = Field(default=None)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    n: int = Field(default=1, ge=1, le=1)  # Only support n=1 for now
    user: str | None = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


def _load_image_from_url(url: str) -> "Image.Image":
    """Load an image from a URL or base64 data URI."""
    if not HAS_PIL:
        raise ImportError("PIL is required for vision features. Install with: pip install Pillow")

    if url.startswith("data:"):
        # Parse base64 data URI
        # Format: data:image/png;base64,<data>
        header, data = url.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))
    else:
        # Load from URL
        with urlopen(url) as response:
            return Image.open(BytesIO(response.read()))


def _extract_images_from_messages(messages: list[ChatMessage]) -> list["Image.Image"]:
    """Extract all images from messages."""
    images = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if hasattr(part, "image_url") and part.image_url:
                    try:
                        img = _load_image_from_url(part.image_url.url)
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image: {e}")
    return images


def _extract_text_from_content(content: str | list[ContentPart] | None) -> str:
    """Extract text from content (handles both string and list formats)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Extract text parts from list
    text_parts = []
    for part in content:
        if hasattr(part, "text"):
            text_parts.append(part.text)
    return " ".join(text_parts)


def _has_images(messages: list[ChatMessage]) -> bool:
    """Check if any message contains images."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if hasattr(part, "image_url") and part.image_url:
                    return True
    return False


def _format_messages_for_model(
    messages: list[ChatMessage],
    tokenizer: Any,
    tools: list[ToolDefinition] | None = None,
) -> str:
    """Format messages for the model using chat template."""
    # Convert to the format expected by the tokenizer
    formatted_messages = []
    for msg in messages:
        # Extract text content (handles both string and list formats)
        content = _extract_text_from_content(msg.content)
        message_dict = {"role": msg.role, "content": content}
        if msg.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            message_dict["tool_call_id"] = msg.tool_call_id
        formatted_messages.append(message_dict)

    # Add tools to the conversation if provided
    tools_dict = None
    if tools:
        tools_dict = [
            {"type": t.type, "function": t.function.model_dump()} for t in tools
        ]

    try:
        # Try to use chat template with tools
        if tools_dict and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                formatted_messages,
                tools=tools_dict,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        logger.warning(f"Chat template failed, using fallback: {e}")

    # Fallback: simple concatenation
    prompt_parts = []
    for msg in formatted_messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        elif role == "tool":
            prompt_parts.append(f"Tool Result: {content}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def _parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Parse tool calls from model output."""
    # Try to parse JSON tool calls from model output
    # Different models have different formats, try common ones

    # Try Qwen format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    import re

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(tool_call_pattern, text, re.DOTALL)

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match.strip())
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=data.get("name", ""),
                        arguments=json.dumps(data.get("arguments", {})),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Try function call format: {"function_call": {"name": ..., "arguments": ...}}
    if not tool_calls:
        try:
            # Look for JSON objects that might be function calls
            json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
            json_matches = re.findall(json_pattern, text)
            for i, match in enumerate(json_matches):
                try:
                    data = json.loads(match)
                    if "name" in data and "arguments" in data:
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=data["name"],
                                    arguments=(
                                        json.dumps(data["arguments"])
                                        if isinstance(data["arguments"], dict)
                                        else str(data["arguments"])
                                    ),
                                ),
                            )
                        )
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    return tool_calls if tool_calls else None


async def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> tuple[str, int, int]:
    """Generate completion synchronously."""
    from mlx_lm import generate

    loop = asyncio.get_event_loop()

    def _generate():
        prompt_tokens = len(tokenizer.encode(prompt))
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )
        completion_tokens = len(tokenizer.encode(response))
        return response, prompt_tokens, completion_tokens

    return await loop.run_in_executor(None, _generate)


async def _generate_vlm_completion(
    model: Any,
    processor: Any,
    prompt: str,
    images: list["Image.Image"],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    """Generate completion with vision model."""
    from mlx_vlm import generate

    loop = asyncio.get_event_loop()

    def _generate():
        # VLM generate expects images and prompt
        response = generate(
            model,
            processor,
            prompt,
            images[0] if images else None,  # Most VLMs handle single image
            max_tokens=max_tokens,
            temp=temperature,
        )
        # Approximate token counts
        prompt_tokens = len(prompt.split()) * 2  # Rough estimate
        completion_tokens = len(response.split()) * 2
        return response, prompt_tokens, completion_tokens

    return await loop.run_in_executor(None, _generate)


async def _stream_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    request_id: str,
    model_name: str,
    created: int,
):
    """Stream completion tokens."""
    from mlx_lm import stream_generate

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream tokens
    loop = asyncio.get_event_loop()

    def _stream():
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            yield chunk

    # Run stream in thread and yield results
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    finished = threading.Event()

    def producer():
        try:
            for chunk in _stream():
                q.put(chunk)
        finally:
            finished.set()

    thread = threading.Thread(target=producer)
    thread.start()

    full_response = ""
    while not finished.is_set() or not q.empty():
        try:
            chunk = q.get(timeout=0.1)
            if hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            full_response += text

            content_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=text),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
        except queue.Empty:
            continue

    thread.join()

    # Send final chunk
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion.

    Supports streaming, tool/function calling, and vision (image inputs).
    """
    # Check if request contains images
    has_vision = _has_images(request.messages)
    images = []

    if has_vision:
        # Use VLM model for vision requests
        try:
            model, processor = model_manager.get_vlm_model(request.model)
            images = _extract_images_from_messages(request.messages)
        except ValueError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Vision model '{request.model}' not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            ) from e
        tokenizer = processor  # VLM uses processor instead of tokenizer
    else:
        # Use LLM model for text-only requests
        try:
            model, tokenizer = model_manager.get_llm_model(request.model)
        except ValueError as e:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Model '{request.model}' not found",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            ) from e

    # Format messages
    prompt = _format_messages_for_model(request.messages, tokenizer, request.tools)

    # Prepare generation parameters
    max_tokens = request.max_tokens or 2048
    stop = [request.stop] if isinstance(request.stop, str) else request.stop

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Vision models don't support streaming yet
    if request.stream and not has_vision:
        # Streaming response (text-only)
        return StreamingResponse(
            _stream_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                request_id=request_id,
                model_name=request.model,
                created=created,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    try:
        if has_vision:
            # Use VLM generation
            response_text, prompt_tokens, completion_tokens = await _generate_vlm_completion(
                model=model,
                processor=tokenizer,
                prompt=prompt,
                images=images,
                max_tokens=max_tokens,
                temperature=request.temperature,
            )
        else:
            # Use LLM generation
            response_text, prompt_tokens, completion_tokens = await _generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
            )

        # Check for tool calls in response
        tool_calls = None
        finish_reason: Literal["stop", "length", "tool_calls"] = "stop"

        if request.tools:
            tool_calls = _parse_tool_calls(response_text)
            if tool_calls:
                finish_reason = "tool_calls"

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text if not tool_calls else None,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Chat completion failed: {str(e)}",
                    "type": "server_error",
                    "code": "completion_failed",
                }
            },
        ) from e


# Text Completions API (Legacy)
class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = Field(..., description="Model name to use")
    prompt: str | list[str] = Field(..., description="Prompt(s) to complete")
    max_tokens: int | None = Field(default=16, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = Field(default=False)
    stop: str | list[str] | None = Field(default=None)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    logprobs: int | None = Field(default=None)
    echo: bool = Field(default=False)
    suffix: str | None = Field(default=None)
    user: str | None = None


class CompletionChoice(BaseModel):
    """Text completion choice."""

    text: str
    index: int
    logprobs: dict | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


class CompletionStreamChoice(BaseModel):
    """Streaming completion choice."""

    text: str
    index: int
    logprobs: dict | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionChunk(BaseModel):
    """OpenAI-compatible streaming completion chunk."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]


async def _stream_text_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    request_id: str,
    model_name: str,
    created: int,
    echo: bool = False,
):
    """Stream text completion tokens."""
    from mlx_lm import stream_generate

    # Echo the prompt if requested
    if echo:
        echo_chunk = CompletionChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                CompletionStreamChoice(
                    text=prompt,
                    index=0,
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {echo_chunk.model_dump_json()}\n\n"

    # Stream tokens
    import queue
    import threading

    def _stream():
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            yield chunk

    q: queue.Queue = queue.Queue()
    finished = threading.Event()

    def producer():
        try:
            for chunk in _stream():
                q.put(chunk)
        finally:
            finished.set()

    thread = threading.Thread(target=producer)
    thread.start()

    while not finished.is_set() or not q.empty():
        try:
            chunk = q.get(timeout=0.1)
            if hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            content_chunk = CompletionChunk(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    CompletionStreamChoice(
                        text=text,
                        index=0,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
        except queue.Empty:
            continue

    thread.join()

    # Send final chunk
    final_chunk = CompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[
            CompletionStreamChoice(
                text="",
                index=0,
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Create a text completion.

    This is the legacy completions API. For new applications,
    use /v1/chat/completions instead.
    """
    try:
        model, tokenizer = model_manager.get_llm_model(request.model)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        ) from e

    # Normalize prompt to string (take first if list)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    max_tokens = request.max_tokens or 16
    stop = [request.stop] if isinstance(request.stop, str) else request.stop

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_text_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop,
                request_id=request_id,
                model_name=request.model,
                created=created,
                echo=request.echo,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    try:
        response_text, prompt_tokens, completion_tokens = await _generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
        )

        # Handle echo option
        output_text = (prompt + response_text) if request.echo else response_text

        return CompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                CompletionChoice(
                    text=output_text,
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Text completion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Text completion failed: {str(e)}",
                    "type": "server_error",
                    "code": "completion_failed",
                }
            },
        ) from e
