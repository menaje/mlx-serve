"""Isolated, bounded and atomic preparation of supported d1 checkpoints."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from mlx_serve.config import settings
from mlx_serve.core.decision_prepare_worker import MANIFEST, MLX_REPO, SOURCE_REPO, file_hash
from mlx_serve.core.storage_access import model_volume_mounted

logger = logging.getLogger(__name__)
SUPPORTED_REPOS = {SOURCE_REPO, MLX_REPO}


def read_verified_manifest(path: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    manifest = json.loads((path / MANIFEST).read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("source_repo") not in SUPPORTED_REPOS
        or manifest.get("architecture") != "d1_omni"
        or manifest.get("precision") != "float16"
        or not manifest.get("revision")
        or not manifest.get("validation", {}).get("official_predict")
    ):
        raise ValueError("Decision model manifest is invalid")
    files = manifest.get("files_sha256")
    if (
        not isinstance(files, dict)
        or not {"config.json", "tokenizer.json", "LICENSE"} <= files.keys()
    ):
        raise ValueError("Decision model manifest is incomplete")
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Decision model manifest has no weights")
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != set(files) | {MANIFEST}:
        raise ValueError("Decision model contains unexpected or missing files")
    for name, digest in files.items():
        target = path / name
        if not target.resolve().is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError(f"Missing or invalid model file: {name}")
        if verify_hashes and file_hash(target) != digest:
            raise ValueError(f"Decision model file checksum mismatch: {name}")
    return manifest


def _rss_bytes(pid: int) -> int:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    try:
        return int(result.stdout.strip()) * 1024
    except ValueError:
        return 0


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            # Downloads may finish or rename temporary files while they are being counted.
            continue
    return total


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


async def prepare_d1(
    manager: Any,
    repo: str,
    name: str,
    requested_revision: str | None,
    *,
    keep_original: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Resolve an immutable revision, prepare it once, and publish only on success."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        yield {"status": "error", "message": "Invalid decision model name"}
        return
    from mlx_serve.core.model_manager import resolve_decision_install_name

    try:
        name = resolve_decision_install_name(name, repo)
    except ValueError as exc:
        yield {"status": "error", "message": str(exc)}
        return
    if not model_volume_mounted(settings.models_dir):
        yield {"status": "error", "message": "Model storage is unavailable"}
        return
    model_dir = manager._get_model_dir(name)
    lock_dir = settings.models_dir / ".decision-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "d1-preparation.lock"
    deadline = time.monotonic() + settings.decision_prepare_timeout_seconds
    process = None
    stage: Path | None = None
    with lock_path.open("a+") as lock:
        try:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for decision model preparation")
                    await asyncio.sleep(0.2)
            yield {"status": "resolving", "name": name, "hf_repo": repo}
            info = await asyncio.wait_for(
                asyncio.to_thread(
                    HfApi().model_info, repo, revision=requested_revision, files_metadata=True
                ),
                timeout=max(0.1, deadline - time.monotonic()),
            )
            revision = info.sha
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Unable to resolve an immutable model revision")
            if model_dir.exists():
                manifest = await asyncio.to_thread(read_verified_manifest, model_dir)
                if manifest["revision"] != revision or manifest["source_repo"] != repo:
                    raise ValueError(
                        f"Model '{name}' already contains a different revision; choose another name"
                    )
                # Refresh this process's metadata if a concurrent process installed it.
                manager._load_metadata()
                if name not in manager._metadata:
                    raise ValueError("Existing model is not registered")
                metadata = manager._metadata[name]
                if metadata.get("revision") != revision or metadata.get("hf_repo") != repo:
                    raise ValueError("Registered decision metadata does not match its manifest")
                if (
                    keep_original
                    and repo == SOURCE_REPO
                    and (
                        not metadata.get("original_path")
                        or not Path(metadata["original_path"]).is_dir()
                    )
                ):
                    raise ValueError(
                        "This installation did not retain its source; choose a new model name"
                    )
                yield {"status": "success", "name": name, "revision": revision, "reused": True}
                return
            # Serialize d1 preparations and reuse any verified artifact with this source revision,
            # including one registered under another local name.
            if manager._metadata_path.exists():
                current = json.loads(manager._metadata_path.read_text())
                if not isinstance(current, dict):
                    raise ValueError("Model metadata is invalid")
                manager._metadata = current
            reusable = None
            for candidate_name, candidate_meta in manager._metadata.items():
                if (
                    candidate_meta.get("type") != "decision"
                    or candidate_meta.get("hf_repo") != repo
                    or candidate_meta.get("revision") != revision
                ):
                    continue
                original_path = candidate_meta.get("original_path")
                if (
                    keep_original
                    and repo == SOURCE_REPO
                    and (not original_path or not Path(original_path).is_dir())
                ):
                    continue
                candidate_dir = manager._get_model_dir(candidate_name)
                try:
                    candidate_manifest = await asyncio.to_thread(
                        read_verified_manifest, candidate_dir
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if (
                    candidate_manifest["revision"] == revision
                    and candidate_manifest["source_repo"] == repo
                ):
                    reusable = (candidate_dir, original_path)
                    break
            job_id = os.environ.get("MLX_SERVE_PREPARE_JOB_ID", "")
            job_prefix = f"{job_id}-" if re.fullmatch(r"[0-9a-f]{32}", job_id) else ""
            stage = Path(
                tempfile.mkdtemp(prefix=f".decision-prepare-{job_prefix}", dir=settings.models_dir)
            )
            source, artifact = stage / "source", stage / "artifact"
            if reusable is not None:
                copy_size = await asyncio.to_thread(_directory_size, reusable[0])
                free = shutil.disk_usage(settings.models_dir).free
                yield {
                    "status": "checking", "name": name,
                    "estimated_disk_bytes": copy_size, "free_disk_bytes": free,
                    "estimate_basis": "verified_copy",
                }
                if copy_size > settings.decision_prepare_max_disk_bytes:
                    raise ValueError("Model preparation exceeds the configured disk limit")
                if free < copy_size:
                    raise ValueError("Insufficient free disk space to reuse decision model")
                yield {
                    "status": "validating",
                    "name": name,
                    "detail": "Reusing a verified model with the same revision",
                }
                await asyncio.to_thread(shutil.copytree, reusable[0], artifact)
                yield {
                    "status": "resources", "name": name,
                    "temporary_file_bytes": await asyncio.to_thread(_directory_size, stage),
                }
                if time.monotonic() >= deadline:
                    raise TimeoutError("Decision model preparation exceeded its time limit")
            else:
                siblings = info.siblings or []
                total = sum((getattr(item, "size", 0) or 0) for item in siblings)
                expected = int(total * (1.6 if repo == SOURCE_REPO else 1.1)) + 128 * 1024**2
                free = shutil.disk_usage(settings.models_dir).free
                complete_sizes = bool(siblings) and all(
                    getattr(item, "size", None) is not None for item in siblings
                )
                yield {
                    "status": "checking", "name": name,
                    "estimated_disk_bytes": expected, "free_disk_bytes": free,
                    "estimate_basis": "repository_metadata" if complete_sizes
                    else "partial_repository_metadata",
                }
                if expected > settings.decision_prepare_max_disk_bytes:
                    raise ValueError("Model preparation exceeds the configured disk limit")
                if free < expected:
                    raise ValueError("Insufficient free disk space for decision model preparation")
                environment = os.environ.copy()
                package_root = str(Path(__file__).resolve().parents[2])
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(None, (package_root, environment.get("PYTHONPATH")))
                )
                process = await asyncio.create_subprocess_exec(
                    settings.decision_prepare_python or sys.executable,
                    "-m",
                    "mlx_serve.core.decision_prepare_worker",
                    repo,
                    revision,
                    str(source),
                    str(artifact),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                )
                assert process.stdout is not None
                error = None
                last_check = 0.0
                last_report = 0.0
                last_reported_disk: int | None = None
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Decision model preparation exceeded its time limit")
                    if time.monotonic() - last_check >= 1:
                        rss, disk = await asyncio.gather(
                            asyncio.to_thread(_rss_bytes, process.pid),
                            asyncio.to_thread(_directory_size, stage),
                        )
                        if rss > settings.decision_prepare_max_memory_bytes:
                            raise MemoryError(
                                "Decision model preparation exceeded its memory limit"
                            )
                        if disk > settings.decision_prepare_max_disk_bytes:
                            raise ValueError("Decision model preparation exceeded its disk limit")
                        last_check = time.monotonic()
                        if last_check - last_report >= 10 and disk != last_reported_disk:
                            # A logical file size, not a download percentage or prediction.
                            yield {
                                "status": "resources", "name": name,
                                "temporary_file_bytes": disk,
                            }
                            last_report = last_check
                            last_reported_disk = disk
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=0.5)
                    except TimeoutError:
                        continue
                    if not line:
                        break
                    text = line.decode(errors="replace").strip()
                    if text.startswith("MLX_PREP_STATUS="):
                        event = json.loads(text.removeprefix("MLX_PREP_STATUS="))
                        if event["status"] == "error":
                            error = event.get("message", "Model preparation failed")
                        elif event["status"] != "ready":
                            yield {**event, "name": name}
                    elif text:
                        logger.debug("Decision preparation: %s", text)
                if await process.wait() != 0:
                    raise ValueError(error or "Decision model preparation failed")
            manifest = await asyncio.to_thread(read_verified_manifest, artifact)
            if manifest["revision"] != revision or manifest["source_repo"] != repo:
                raise ValueError("Prepared model revision does not match requested revision")
            metadata = {
                "type": "decision",
                "hf_repo": repo,
                "revision": revision,
                "size": manager._get_dir_size(artifact),
                "modified_at": datetime.now().isoformat(),
                "precision": "float16",
                "architecture": "d1_omni",
            }
            yield {"status": "registering", "name": name}
            metadata_lock = settings.models_dir.parent / ".decision-metadata.lock"
            with metadata_lock.open("a+") as registration_lock:
                while True:
                    try:
                        fcntl.flock(registration_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting to register decision model")
                        await asyncio.sleep(0.2)
                try:
                    if manager._metadata_path.exists():
                        current = json.loads(manager._metadata_path.read_text())
                        if not isinstance(current, dict):
                            raise ValueError("Model metadata is invalid")
                        manager._metadata = current
                    if name in manager._metadata or model_dir.exists():
                        raise ValueError(f"Model '{name}' was registered during preparation")
                    # The artifact and final directory share a filesystem, so rename is atomic.
                    artifact.rename(model_dir)
                    created_original = None
                    try:
                        if keep_original and repo == SOURCE_REPO and reusable is not None:
                            metadata["original_path"] = reusable[1]
                        elif keep_original and repo == SOURCE_REPO:
                            originals = settings.models_dir.parent / "decision-originals"
                            originals.mkdir(parents=True, exist_ok=True)
                            retained = originals / f"{name}-{revision}"
                            if retained.exists():
                                raise ValueError("The original model destination already exists")
                            source.rename(retained)
                            created_original = retained
                            metadata["original_path"] = str(retained)
                        manager._metadata[name] = metadata
                        manager._save_metadata()
                    except Exception:
                        manager._metadata.pop(name, None)
                        shutil.rmtree(model_dir, ignore_errors=True)
                        if created_original is not None:
                            shutil.rmtree(created_original, ignore_errors=True)
                        raise
                finally:
                    fcntl.flock(registration_lock.fileno(), fcntl.LOCK_UN)
            yield {
                "status": "success",
                "name": name,
                "revision": revision,
                "reused": reusable is not None,
            }
        except Exception as exc:
            logger.warning("Failed to prepare decision model %s: %s", repo, exc)
            yield {"status": "error", "message": str(exc)}
        finally:
            if process is not None:
                with contextlib.suppress(ProcessLookupError):
                    await _stop_process(process)
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


async def prepare_d3(manager: Any, repo: str, name: str) -> AsyncIterator[dict[str, Any]]:
    """Download D3 into a private staging directory before publishing it."""
    if not model_volume_mounted(settings.models_dir):
        yield {"status": "error", "message": "Model storage is unavailable"}
        return
    model_dir = manager._get_model_dir(name)
    if model_dir.exists():
        yield {
            "status": "error",
            "message": f"Model '{name}' already exists; remove it before pulling again",
        }
        return

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".decision-download-", dir=settings.models_dir
        ) as temp:
            artifact = Path(temp) / "artifact"
            artifact.mkdir()
            yield {"status": "downloading", "name": name, "hf_repo": repo}
            download = asyncio.create_task(
                asyncio.to_thread(snapshot_download, repo_id=repo, local_dir=str(artifact))
            )
            try:
                await asyncio.shield(download)
            except asyncio.CancelledError:
                # Do not remove the staging directory while its download thread is writing.
                with contextlib.suppress(Exception):
                    await download
                raise

            yield {
                "status": "converting",
                "name": name,
                "detail": "Decision model checkpoint ready",
            }
            metadata = {
                "type": "decision",
                "hf_repo": repo,
                "size": manager._get_dir_size(artifact),
                "modified_at": datetime.now().isoformat(),
            }
            deadline = time.monotonic() + settings.decision_prepare_timeout_seconds
            metadata_lock = settings.models_dir.parent / ".decision-metadata.lock"
            with metadata_lock.open("a+") as lock:
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting to register decision model")
                        await asyncio.sleep(0.2)
                try:
                    manager._load_metadata()
                    if name in manager._metadata or model_dir.exists():
                        raise ValueError(
                            f"Model '{name}' already exists; remove it before pulling again"
                        )
                    artifact.rename(model_dir)
                    try:
                        manager._metadata[name] = metadata
                        manager._save_metadata()
                    except BaseException:
                        manager._metadata.pop(name, None)
                        shutil.rmtree(model_dir, ignore_errors=True)
                        raise
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            yield {"status": "success", "name": name}
    except Exception as exc:
        logger.warning("Failed to prepare D3 model %s: %s", repo, exc)
        yield {"status": "error", "message": str(exc)}
