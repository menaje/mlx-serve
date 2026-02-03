"""Audio API router - OpenAI compatible TTS and STT."""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mlx_serve.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])

# Security constants
MAX_AUDIO_FILE_SIZE = 25 * 1024 * 1024  # 25MB max (OpenAI limit)


# TTS Models
class SpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request."""

    model: str = Field(default="kokoro", description="TTS model to use")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="af_heart", description="Voice to use")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio format",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed of speech")
    instructions: str | None = Field(
        default=None,
        description="Instructions for voice style (model-dependent)",
    )


async def _generate_speech(
    model,
    text: str,
    voice: str,
    speed: float,
    response_format: str,
) -> bytes:
    """Generate speech from text."""
    loop = asyncio.get_running_loop()

    def _synthesize():
        from mlx_audio.tts import generate

        # Generate audio
        audio = generate(
            model,
            text,
            voice=voice,
            speed=speed,
        )
        return audio

    audio_array = await loop.run_in_executor(None, _synthesize)

    # Convert to requested format
    def _encode():
        import numpy as np

        try:
            import soundfile as sf
        except ImportError:
            # Fallback: return raw PCM if soundfile not available
            if hasattr(audio_array, "numpy"):
                return audio_array.numpy().tobytes()
            return np.array(audio_array).tobytes()

        # Convert to numpy if needed
        if hasattr(audio_array, "numpy"):
            audio_np = audio_array.numpy()
        else:
            audio_np = np.array(audio_array)

        # Ensure correct shape and dtype
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(-1, 1)
        audio_np = audio_np.astype(np.float32)

        # Write to buffer
        buffer = io.BytesIO()
        # Map response_format to soundfile format
        format_map = {
            "mp3": "MP3",
            "wav": "WAV",
            "flac": "FLAC",
            "ogg": "OGG",
            "opus": "OGG",  # soundfile uses OGG container for opus
            "aac": "WAV",  # Fallback to WAV for unsupported formats
            "pcm": "RAW",
        }
        sf_format = format_map.get(response_format, "WAV")

        try:
            sf.write(buffer, audio_np, samplerate=24000, format=sf_format)
        except Exception:
            # Fallback to WAV
            sf.write(buffer, audio_np, samplerate=24000, format="WAV")

        buffer.seek(0)
        return buffer.read()

    return await loop.run_in_executor(None, _encode)


@router.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate speech from text.

    Returns audio in the requested format.
    """
    try:
        model = model_manager.get_tts_model(request.model)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"TTS model '{request.model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        ) from e

    try:
        audio_bytes = await _generate_speech(
            model=model,
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format,
        )

        # Determine content type
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        content_type = content_types.get(request.response_format, "audio/mpeg")

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
            },
        )

    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Speech synthesis failed: {str(e)}",
                    "type": "server_error",
                    "code": "synthesis_failed",
                }
            },
        ) from e


# STT Models
class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str


class TranscriptionVerboseResponse(BaseModel):
    """Verbose transcription response with segments."""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[dict] | None = None


async def _transcribe_audio(
    model,
    audio_path: str,
    language: str | None,
    response_format: str,
    timestamp_granularities: list[str] | None = None,
) -> dict:
    """Transcribe audio to text."""
    loop = asyncio.get_running_loop()

    def _transcribe():
        from mlx_audio.stt import transcribe

        # Determine if word-level timestamps are requested
        word_timestamps = (
            timestamp_granularities is not None and "word" in timestamp_granularities
        )

        result = transcribe(
            model,
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
        )
        return result

    return await loop.run_in_executor(None, _transcribe)


@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="whisper-large-v3-turbo", description="STT model to use"),
    language: str | None = Form(default=None, description="Language of the audio"),
    prompt: str | None = Form(default=None, description="Optional prompt to guide transcription"),
    response_format: str = Form(default="json", description="Response format"),
    temperature: float = Form(default=0.0, description="Sampling temperature"),
    timestamp_granularities: list[str] | None = Form(
        default=None,
        description="Timestamp granularities: 'word' and/or 'segment' (verbose_json only)",
    ),
):
    """Transcribe audio to text.

    Supports various audio formats (mp3, wav, m4a, etc.).
    """
    try:
        stt_model = model_manager.get_stt_model(model)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"STT model '{model}' not found",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        ) from e

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_AUDIO_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "message": f"Audio file too large. Maximum size is {MAX_AUDIO_FILE_SIZE // (1024 * 1024)}MB",
                    "type": "invalid_request_error",
                    "code": "file_too_large",
                }
            },
        )

    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await _transcribe_audio(
            model=stt_model,
            audio_path=tmp_path,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )

        # Handle different result formats
        if isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)

        if response_format == "verbose_json":
            return TranscriptionVerboseResponse(
                task="transcribe",
                language=language or "unknown",
                duration=0.0,  # TODO: calculate from audio
                text=text,
                segments=result.get("segments") if isinstance(result, dict) else None,
            )

        return TranscriptionResponse(text=text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Transcription failed: {str(e)}",
                    "type": "server_error",
                    "code": "transcription_failed",
                }
            },
        ) from e
    finally:
        # Clean up temp file
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
