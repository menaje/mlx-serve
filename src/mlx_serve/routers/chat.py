"""Chat completions API router - OpenAI compatible."""

import asyncio
import base64
import ipaddress
import json
import logging
import re
import socket
import time
import uuid
from io import BytesIO
from typing import Any, Literal
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mlx_serve.core.model_manager import model_manager

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

# Security constants
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB max image size
ALLOWED_IMAGE_HOSTS = None  # Set to list of allowed hosts, or None to allow all external
BLOCKED_IP_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                       "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                       "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                       "172.30.", "172.31.", "192.168.", "127.", "0.", "169.254.")

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


class ResponseFormatText(BaseModel):
    """Text response format."""

    type: Literal["text"] = "text"


class JsonSchemaDefinition(BaseModel):
    """JSON schema definition for structured outputs."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class ResponseFormatJsonSchema(BaseModel):
    """JSON schema response format for structured outputs."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchemaDefinition


class ResponseFormatJsonObject(BaseModel):
    """JSON object response format."""

    type: Literal["json_object"] = "json_object"


ResponseFormat = ResponseFormatText | ResponseFormatJsonSchema | ResponseFormatJsonObject


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = Field(default=False, description="Include usage statistics in stream")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model name to use")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    max_completion_tokens: int | None = Field(
        default=None, description="Maximum completion tokens (alternative to max_tokens)"
    )
    stream: bool = Field(default=False)
    stream_options: StreamOptions | None = Field(
        default=None, description="Options for streaming (e.g., include_usage)"
    )
    stop: str | list[str] | None = Field(default=None)
    logit_bias: dict[str, float] | None = Field(
        default=None, description="Token ID to bias value (-100 to 100)"
    )
    tools: list[ToolDefinition] | None = Field(default=None)
    tool_choice: str | dict | None = Field(default=None)
    parallel_tool_calls: bool | None = Field(
        default=None, description="Whether to enable parallel function calling"
    )
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    n: int = Field(default=1, ge=1, le=1)  # Only support n=1 for now
    user: str | None = None
    # New parameters for OpenAI API compatibility
    response_format: ResponseFormat | None = Field(
        default=None, description="Format of the response (text, json_object, json_schema)"
    )
    seed: int | None = Field(default=None, description="Seed for deterministic generation")
    logprobs: bool | None = Field(default=None, description="Whether to return log probabilities")
    top_logprobs: int | None = Field(
        default=None, ge=0, le=20, description="Number of most likely tokens to return (0-20)"
    )
    service_tier: Literal["auto", "default", "flex", "priority"] | None = Field(
        default=None, description="Service tier for request processing"
    )
    store: bool | None = Field(default=None, description="Whether to store the response")
    metadata: dict[str, str] | None = Field(default=None, description="Request metadata")


class TopLogprob(BaseModel):
    """Top logprob entry."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(BaseModel):
    """Log probability information for a token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] | None = None


class ChoiceLogprobs(BaseModel):
    """Log probabilities for a choice."""

    content: list[TokenLogprob] | None = None
    refusal: list[TokenLogprob] | None = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
    logprobs: ChoiceLogprobs | None = None


class CompletionTokensDetails(BaseModel):
    """Detailed breakdown of completion tokens."""

    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """Detailed breakdown of prompt tokens."""

    cached_tokens: int = 0


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    system_fingerprint: str | None = None
    service_tier: str | None = None


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict] | None = None
    refusal: str | None = None


class StreamChoiceLogprobs(BaseModel):
    """Log probabilities for streaming choice."""

    content: list[TokenLogprob] | None = None
    refusal: list[TokenLogprob] | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = None
    logprobs: StreamChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None
    service_tier: str | None = None
    usage: ChatCompletionUsage | None = None


def _is_private_ip(host: str) -> bool:
    """Check if host is a private/reserved IP address (IPv4 or IPv6)."""
    try:
        # Try to resolve hostname to IP
        # This also handles bracket notation for IPv6 like [::1]
        host_clean = host.strip("[]")
        ip = ipaddress.ip_address(host_clean)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        # Not a valid IP address, might be a hostname
        # Try to resolve it
        try:
            infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC)
            for info in infos:
                ip_str = info[4][0]
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_multicast
                    or ip.is_reserved
                    or ip.is_unspecified
                ):
                    return True
            return False
        except (socket.gaierror, socket.herror):
            # Cannot resolve - allow but let urlopen handle it
            return False


def _validate_image_url(url: str) -> None:
    """Validate image URL for security (SSRF prevention)."""
    parsed = urlparse(url)

    # Must be http or https
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

    host = parsed.hostname or ""

    # Check for localhost variants
    if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"):
        raise ValueError("Access to localhost is not allowed")

    # Check for private/reserved IP addresses (IPv4 and IPv6)
    if _is_private_ip(host):
        raise ValueError(f"Access to private/reserved network addresses is not allowed: {host}")

    # Check allowed hosts whitelist if configured
    if ALLOWED_IMAGE_HOSTS is not None and host not in ALLOWED_IMAGE_HOSTS:
        raise ValueError(f"Host {host} is not in the allowed hosts list")


def _load_image_from_url(url: str) -> "Image.Image":
    """Load an image from a URL or base64 data URI."""
    if not HAS_PIL:
        raise ImportError("PIL is required for vision features. Install with: pip install Pillow")

    if url.startswith("data:"):
        # Parse base64 data URI
        # Format: data:image/png;base64,<data>
        try:
            if "," not in url:
                raise ValueError("Invalid data URI format: missing comma separator")
            header, data = url.split(",", 1)
            image_data = base64.b64decode(data)

            # Check size limit
            if len(image_data) > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image size exceeds limit of {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB")

            return Image.open(BytesIO(image_data))
        except (ValueError, base64.binascii.Error) as e:
            raise ValueError(f"Invalid base64 data URI: {e}") from e
    else:
        # Validate URL for security
        _validate_image_url(url)

        # Load from URL with size limit
        try:
            with urlopen(url, timeout=30) as response:
                # Check content-length header if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_IMAGE_SIZE_BYTES:
                    raise ValueError(f"Image size exceeds limit of {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB")

                # Read with size limit
                image_data = response.read(MAX_IMAGE_SIZE_BYTES + 1)
                if len(image_data) > MAX_IMAGE_SIZE_BYTES:
                    raise ValueError(f"Image size exceeds limit of {MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB")

                return Image.open(BytesIO(image_data))
        except Exception as e:
            if "size exceeds" in str(e):
                raise
            raise ValueError(f"Failed to load image from URL: {e}") from e


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


def _build_response_format_instruction(response_format: ResponseFormat | None) -> str:
    """Build instruction text for response format."""
    if response_format is None:
        return ""

    if isinstance(response_format, ResponseFormatJsonObject):
        return "\n\nIMPORTANT: You must respond with a valid JSON object. Do not include any other text."
    elif isinstance(response_format, ResponseFormatJsonSchema):
        schema_json = json.dumps(response_format.json_schema.schema_, indent=2)
        instruction = f"\n\nIMPORTANT: You must respond with a valid JSON object that matches the following schema:\n{schema_json}\n\nDo not include any other text outside the JSON."
        return instruction

    return ""


def _validate_json_response(text: str, response_format: ResponseFormat | None) -> str:
    """Validate and potentially extract JSON from response."""
    if response_format is None:
        return text

    if isinstance(response_format, (ResponseFormatJsonObject, ResponseFormatJsonSchema)):
        # Try to extract JSON from the response
        text = text.strip()

        # Try to find JSON in the response
        if not text.startswith("{") and not text.startswith("["):
            # Look for JSON block in markdown code block
            json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
            if json_match:
                text = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                if json_match:
                    text = json_match.group(1)

        # Validate JSON
        try:
            json.loads(text)
        except json.JSONDecodeError:
            # Return as-is if not valid JSON
            pass

    return text


def _parse_tool_calls(text: str) -> list[ToolCall] | None:
    """Parse tool calls from model output."""
    # Try to parse JSON tool calls from model output
    # Different models have different formats, try common ones

    # Try Qwen format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
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


def _truncate_at_stop_sequences(text: str, stop: list[str] | None) -> tuple[str, bool]:
    """Truncate text at first occurrence of any stop sequence.

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not stop:
        return text, False

    earliest_pos = len(text)
    found = False
    for seq in stop:
        pos = text.find(seq)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos
            found = True

    if found:
        return text[:earliest_pos], True
    return text, False


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
    from mlx_lm.sample_utils import make_sampler

    loop = asyncio.get_running_loop()

    def _generate():
        prompt_tokens = len(tokenizer.encode(prompt))
        sampler = make_sampler(temp=temperature, top_p=top_p)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        # Apply stop sequence truncation
        response, _ = _truncate_at_stop_sequences(response, stop)
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

    loop = asyncio.get_running_loop()

    def _generate():
        # VLM generate expects images and prompt
        result = generate(
            model,
            processor,
            prompt,
            image=images[0] if images else None,  # Most VLMs handle single image
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # Extract text from GenerationResult
        response = result.text if hasattr(result, 'text') else str(result)
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
    system_fingerprint: str | None = None,
    service_tier: str | None = None,
    include_usage: bool = False,
):
    """Stream completion tokens."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    # Calculate prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt))

    # Create sampler for temperature and top_p
    sampler = make_sampler(temp=temperature, top_p=top_p)

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
        system_fingerprint=system_fingerprint,
        service_tier=service_tier,
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream tokens
    def _stream():
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
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
    completion_tokens = 0
    stop_detected = False
    while not finished.is_set() or not q.empty():
        if stop_detected:
            # Drain remaining queue items without yielding
            try:
                while not q.empty():
                    q.get_nowait()
            except queue.Empty:
                pass
            break

        try:
            chunk = q.get(timeout=0.1)
            if hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            # Check for stop sequences
            if stop:
                full_response += text
                truncated, was_truncated = _truncate_at_stop_sequences(full_response, stop)
                if was_truncated:
                    # Only yield the portion before stop sequence
                    text_to_yield = truncated[len(full_response) - len(text):]
                    if text_to_yield:
                        completion_tokens += 1
                        content_chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=model_name,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaMessage(content=text_to_yield),
                                    finish_reason=None,
                                )
                            ],
                            system_fingerprint=system_fingerprint,
                            service_tier=service_tier,
                        )
                        yield f"data: {content_chunk.model_dump_json()}\n\n"
                    stop_detected = True
                    continue
            else:
                full_response += text

            completion_tokens += 1  # Count tokens as they stream

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
                system_fingerprint=system_fingerprint,
                service_tier=service_tier,
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"
        except queue.Empty:
            continue

    thread.join()

    # Send final chunk with finish_reason
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
        system_fingerprint=system_fingerprint,
        service_tier=service_tier,
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    # Send usage chunk if requested
    if include_usage:
        usage_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[],
            system_fingerprint=system_fingerprint,
            service_tier=service_tier,
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion.

    Supports streaming, tool/function calling, and vision (image inputs).
    """
    # Check if request contains images
    has_vision = _has_images(request.messages)
    images = []

    # Check model type and validate it's suitable for chat completions
    model_type = model_manager.get_model_type(request.model)

    # Validate model type is suitable for chat completions
    if model_type is not None and model_type not in ("llm", "vlm"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Model '{request.model}' is type '{model_type}' which is not supported for chat completions. Use llm or vlm models.",
                    "type": "invalid_request_error",
                    "code": "unsupported_model_type",
                }
            },
        )

    is_vlm_model = model_type == "vlm"

    if has_vision or is_vlm_model:
        # Use VLM model for vision requests or VLM models
        try:
            model, processor = model_manager.get_vlm_model(request.model)
            if has_vision:
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

    # Add response format instruction to prompt
    response_format_instruction = _build_response_format_instruction(request.response_format)
    if response_format_instruction:
        prompt = prompt + response_format_instruction

    # Prepare generation parameters (max_completion_tokens takes precedence)
    max_tokens = request.max_completion_tokens or request.max_tokens or 2048
    stop = [request.stop] if isinstance(request.stop, str) else request.stop

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Generate system fingerprint based on model
    system_fingerprint = f"fp_{uuid.uuid4().hex[:12]}"

    # Vision models don't support streaming yet
    if request.stream and not has_vision and not is_vlm_model:
        # Check if usage should be included in stream
        include_usage = (
            request.stream_options.include_usage
            if request.stream_options
            else False
        )

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
                system_fingerprint=system_fingerprint,
                service_tier=request.service_tier or "default",
                include_usage=include_usage,
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
        if has_vision or is_vlm_model:
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

        # Validate and process response based on response_format
        response_text = _validate_json_response(response_text, request.response_format)

        # Check for tool calls in response
        tool_calls = None
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"

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
                completion_tokens_details=CompletionTokensDetails(),
                prompt_tokens_details=PromptTokensDetails(),
            ),
            system_fingerprint=system_fingerprint,
            service_tier=request.service_tier or "default",
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
    from mlx_lm.sample_utils import make_sampler

    # Create sampler
    sampler = make_sampler(temp=temperature, top_p=top_p)

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
            sampler=sampler,
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
