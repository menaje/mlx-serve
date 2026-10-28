"""Conservative, explicit cleanup of detached Hugging Face Hub revisions.

The Hub cache can be shared with unrelated applications.  Candidates must be
reviewed by the user, who must stop those applications before cleanup.  The app
never treats cache contents as exclusively owned or cleans them automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

from huggingface_hub import constants, scan_cache_dir
from huggingface_hub.errors import CacheNotFound

from mlx_serve.core.managed_originals import _cleanup_locks


def _registered_sources(models_dir: Path) -> tuple[set[str], list[Path]]:
    registry = models_dir.parent / "metadata.json"
    if registry.is_symlink():
        raise ValueError("Model registry could not be checked")
    if not registry.exists():
        if any(child.name != ".decision-locks" for child in models_dir.iterdir()):
            raise ValueError("Model registry could not be checked")
        return set(), []
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
        sources, paths = set(), []
        for record in data.values():
            if not isinstance(record, dict):
                raise ValueError
            source = record.get("hf_repo")
            if source is not None:
                if not isinstance(source, str):
                    raise ValueError
                sources.add(source.casefold())
            original = record.get("original_path")
            if original is not None:
                if not isinstance(original, str) or not Path(original).is_absolute():
                    raise ValueError
                paths.append(Path(original).resolve())
        return sources, paths
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Model registry could not be checked") from exc


def _repo_fingerprint(repo: Path, root: Path) -> str:
    """Fingerprint the whole repository, including unscanned cache entries."""
    if repo.is_symlink() or not repo.is_dir() or repo.is_mount():
        raise ValueError("Hugging Face cache repository is invalid")
    if repo.parent != root or repo.stat().st_dev != root.stat().st_dev:
        raise ValueError("Hugging Face cache repository is invalid")
    digest = hashlib.sha256()

    def fail(error: OSError) -> None:
        raise error

    for base, directories, files in os.walk(repo, followlinks=False, onerror=fail):
        directories.sort()
        files.sort()
        for item in [Path(base), *(Path(base) / name for name in directories + files)]:
            entry = item.lstat()
            if entry.st_dev != root.stat().st_dev:
                raise ValueError("Hugging Face cache contains a mounted volume")
            if not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)
                    or stat.S_ISLNK(entry.st_mode)):
                raise ValueError("Hugging Face cache contains an unsupported file")
            digest.update(os.fsencode(str(item.relative_to(repo))))
            digest.update(f"\0{entry.st_mode}\0{entry.st_ino}\0{entry.st_size}"
                          f"\0{entry.st_mtime_ns}\0{entry.st_ctime_ns}\0".encode())
            if stat.S_ISLNK(entry.st_mode):
                if item.name in directories:
                    raise ValueError("Hugging Face cache contains a linked directory")
                digest.update(os.fsencode(os.readlink(item)))
    return digest.hexdigest()


def _scan(models_dir: Path, cache_dir: Path | None) -> tuple[list[dict[str, Any]], Any]:
    requested_root = Path(cache_dir or constants.HF_HUB_CACHE).absolute()
    if requested_root.is_symlink():
        raise ValueError("Hugging Face cache path is invalid")
    # macOS /tmp and /var are commonly symlinked parent directories. The
    # Hub scanner returns canonical paths, so compare both sides canonically.
    root = requested_root.resolve()
    sources, originals = _registered_sources(models_dir)
    try:
        cache = scan_cache_dir(root)
    except CacheNotFound:
        return [], None
    if not root.is_dir():
        raise ValueError("Hugging Face cache path is invalid")
    if cache.warnings:
        raise ValueError("Hugging Face cache contains unreadable entries")
    candidates = []
    for repo in sorted(cache.repos, key=lambda item: (item.repo_type, item.repo_id)):
        # Registered models protect the entire source, including older revisions.
        if repo.repo_type != "model" or repo.repo_id.casefold() in sources:
            continue
        repo_path = Path(os.path.abspath(repo.repo_path))
        if any(repo_path == path or repo_path in path.parents or path in repo_path.parents
               for path in originals):
            continue
        # Keep at least one remaining revision and never remove a branch or tag.
        detached = [r for r in repo.revisions if not r.refs
                    and re.fullmatch(r"[0-9a-f]{40}", r.commit_hash)]
        if not detached or len(repo.revisions) < 2:
            continue
        try:
            fingerprint = _repo_fingerprint(repo_path, root)
        except (OSError, ValueError):
            continue
        one_repo = replace(cache, repos=frozenset({repo}))
        for revision in sorted(detached, key=lambda item: item.commit_hash):
            strategy = one_repo.delete_revisions(revision.commit_hash)
            if strategy.repos:
                continue
            candidates.append({
                "repo": repo.repo_id, "revision": revision.commit_hash,
                "size_bytes": strategy.expected_freed_size,
                "fingerprint": fingerprint,
            })
    return candidates, cache


def stale_hf_cache_revisions(models_dir: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    with _cleanup_locks(models_dir):
        candidates, _ = _scan(models_dir, cache_dir)
        return {"candidates": candidates[:100], "truncated": len(candidates) > 100}


def cleanup_stale_hf_cache_revision(
    models_dir: Path, expected: dict[str, Any], cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Recheck the preview, references and deletion strategy before one removal."""
    with _cleanup_locks(models_dir):
        candidates, cache = _scan(models_dir, cache_dir)
        if expected not in candidates or cache is None:
            raise ValueError("Cleanup targets changed; review them again")
        repo = next((r for r in cache.repos if r.repo_type == "model"
                     and r.repo_id == expected["repo"]), None)
        if repo is None:
            raise ValueError("Cleanup targets changed; review them again")
        strategy = replace(cache, repos=frozenset({repo})).delete_revisions(expected["revision"])
        if strategy.repos or strategy.expected_freed_size != expected["size_bytes"]:
            raise ValueError("Cleanup targets changed; review them again")
        root = Path(os.path.abspath(repo.repo_path))
        for path in (*strategy.snapshots, *strategy.refs, *strategy.blobs, *strategy.files):
            target = Path(os.path.abspath(path))
            if not target.is_relative_to(root):
                raise ValueError("Hugging Face cache deletion path is invalid")
            for parent in target.parents:
                if parent == root:
                    break
                if parent.is_symlink() or parent.is_mount():
                    raise ValueError("Hugging Face cache deletion path is invalid")
        strategy.execute()
        if any(path.exists() or path.is_symlink() for path in strategy.snapshots):
            raise OSError("Hugging Face cache cleanup was incomplete")
        return {"removed": 1, "estimated_bytes": expected["size_bytes"]}
