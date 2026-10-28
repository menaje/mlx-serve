"""Import a local Ollama D3 checkpoint into mlx-serve model storage."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from mlx_serve.config import settings
from mlx_serve.core.model_manager import model_manager, resolve_decision_install_name

_REQUIRED_JSON = {"config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json"}
_EXTRA_FILES = {"vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"}


def import_ollama_decision(model: str, name: str, source_root: Path | None = None) -> Path:
    """Import without changing Ollama's blobs, using hard links when possible."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", model):
        raise ValueError("Ollama model must be a library model with an explicit tag")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in (".", ".."):
        raise ValueError("Invalid destination model name")
    model_base, tag = model.split(":", 1)
    name = resolve_decision_install_name(name, f"vllm-sr/{model_base}")
    source_root = Path(source_root or os.getenv("OLLAMA_MODELS", Path.home() / ".ollama/models"))
    manifest_path = source_root / "manifests-v2" / "ollama.com" / "library" / model_base / tag
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ollama MLX model '{model}' is not installed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("runner") != "mlx" or manifest.get("format") != "safetensors":
        raise ValueError("Only Ollama MLX safetensors decision models are supported")

    layers = manifest.get("layers", [])
    files = {layer.get("name"): layer for layer in layers}
    if not _REQUIRED_JSON.issubset(files) or "weight" not in files:
        raise ValueError("Ollama checkpoint does not contain a complete D3 decision model")

    def blob(layer: dict) -> Path:
        digest = layer.get("digest", "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Invalid Ollama blob digest")
        path = source_root / "blobs" / digest.replace(":", "-")
        if not path.is_file():
            raise FileNotFoundError(f"Missing Ollama blob for {layer.get('name', 'model')}")
        return path

    decision = json.loads(blob(files["decision_config.json"]).read_text())
    if decision.get("format_id") != "d3-code-readout-v1":
        raise ValueError("Only D3 code-readout decision models are supported")
    destination = model_manager._get_model_dir(name)
    if destination.exists():
        raise FileExistsError(f"Model '{name}' already exists in mlx-serve")

    def link_or_copy(source: Path, target: Path) -> None:
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            shutil.copyfile(source, target)

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".decision-import-", dir=settings.models_dir) as temp:
        staging = Path(temp)
        weight_dir = staging / "weights"
        weight_dir.mkdir()
        number = 0
        for layer in layers:
            filename = layer.get("name", "")
            if filename in _REQUIRED_JSON | _EXTRA_FILES:
                link_or_copy(blob(layer), staging / filename)
            elif layer.get("mediaType") == "application/vnd.ollama.image.tensor":
                target = (
                    staging / "readout.safetensors"
                    if filename == "weight"
                    else weight_dir / f"{number:04d}.safetensors"
                )
                link_or_copy(blob(layer), target)
                number += 1
        metadata = {
            "type": "decision",
            "hf_repo": f"vllm-sr/{model_base}",
            "size": model_manager._get_dir_size(staging),
            "modified_at": datetime.now().isoformat(),
            "source": f"ollama:{model}",
        }
        # Other processes can publish models while blobs are being linked.
        # Lock the shared registry, reload it, and publish as one operation.
        lock_path = settings.models_dir.parent / ".decision-metadata.lock"
        with lock_path.open("a+") as registry_lock:
            fcntl.flock(registry_lock.fileno(), fcntl.LOCK_EX)
            try:
                model_manager._load_metadata()
                if destination.exists() or name in model_manager._metadata:
                    raise FileExistsError(f"Model '{name}' already exists in mlx-serve")
                staging.rename(destination)
                try:
                    model_manager._metadata[name] = metadata
                    model_manager._save_metadata()
                except BaseException:
                    model_manager._metadata.pop(name, None)
                    shutil.rmtree(destination, ignore_errors=True)
                    raise
            finally:
                fcntl.flock(registry_lock.fileno(), fcntl.LOCK_UN)
    return destination
