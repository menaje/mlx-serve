"""Exercise the current MLX audio and mflux generation interfaces."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from mlx_serve.routers.audio import _generate_speech, _transcribe_audio
from mlx_serve.routers.images import _generate_image


@pytest.mark.asyncio
async def test_speech_joins_model_chunks_and_uses_their_sample_rate():
    class Model:
        def generate(self, text, voice, speed):
            assert (text, voice, speed) == ("hello", "voice", 1.25)
            yield SimpleNamespace(audio=np.array([0.1, 0.2]), sample_rate=16000)
            yield SimpleNamespace(audio=np.array([0.3]), sample_rate=16000)

    encoded = await _generate_speech(Model(), "hello", "voice", 1.25, "wav")
    samples, sample_rate = sf.read(io.BytesIO(encoded))
    assert sample_rate == 16000
    assert samples.tolist() == pytest.approx([0.1, 0.2, 0.3], abs=0.0001)


@pytest.mark.asyncio
async def test_transcription_converts_mlx_audio_output_and_preserves_timestamps():
    from mlx_audio.stt.models.base import STTOutput

    class Model:
        def generate(self, audio, language, word_timestamps):
            assert (audio, language, word_timestamps) == ("input.wav", "ko", True)
            return STTOutput(
                text="안녕하세요",
                language="ko",
                segments=[{"text": "안녕하세요", "start": 0, "end": 1}],
            )

    result = await _transcribe_audio(Model(), "input.wav", "ko", "verbose_json", ["word"])
    assert result["text"] == "안녕하세요"
    assert result["language"] == "ko"
    assert result["segments"][0]["end"] == 1


def test_verbose_transcription_endpoint_keeps_segment_details(client, monkeypatch):
    from mlx_audio.stt.models.base import STTOutput

    class Model:
        def generate(self, audio, language, word_timestamps):
            assert language == "ko"
            assert word_timestamps
            return STTOutput(
                text="안녕하세요",
                language="ko",
                segments=[{"text": "안녕하세요", "start": 0, "end": 1}],
            )

    monkeypatch.setattr("mlx_serve.routers.audio.model_manager.get_stt_model", lambda name: Model())
    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-small",
            "language": "ko",
            "response_format": "verbose_json",
            "timestamp_granularities": "word",
        },
        files={"file": ("input.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["language"] == "ko"
    assert response.json()["segments"][0]["text"] == "안녕하세요"


@pytest.mark.asyncio
async def test_image_generation_passes_seed_and_encodes_generated_image():
    class Model:
        def generate_image(self, *, seed, prompt, width, height, num_inference_steps, guidance):
            assert isinstance(seed, int)
            assert (prompt, width, height, num_inference_steps, guidance) == (
                "test",
                32,
                32,
                4,
                3.5,
            )
            return SimpleNamespace(image=Image.new("RGB", (width, height), "red"))

    encoded = await _generate_image(Model(), "test", 32, 32)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.format == "PNG"
        assert image.size == (32, 32)


def test_mflux_flux_entry_point_has_current_interface():
    import inspect

    from mflux.models.common.config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    assert ModelConfig.schnell().model_name == "black-forest-labs/FLUX.1-schnell"
    assert "model_config" in inspect.signature(Flux1).parameters
    assert "seed" in inspect.signature(Flux1.generate_image).parameters


def test_vlm_prompt_preserves_image_markers_for_the_official_template(monkeypatch):
    from mlx_serve.routers.chat import ChatMessage, _format_vlm_messages_for_model

    captured = {}

    def template(processor, config, messages, **kwargs):
        captured.update(config=config, messages=messages, kwargs=kwargs)
        return "formatted prompt"

    monkeypatch.setattr("mlx_vlm.prompt_utils.apply_chat_template", template)
    messages = [
        ChatMessage.model_validate(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
                    {"type": "text", "text": "this"},
                ],
            }
        )
    ]
    model = SimpleNamespace(config={"model_type": "idefics3"})
    assert _format_vlm_messages_for_model(messages, object(), model, None) == "formatted prompt"
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "Describe"},
        {"type": "image"},
        {"type": "text", "text": "this"},
    ]
    assert captured["kwargs"]["num_images"] == 1
