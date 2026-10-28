"""Read-only inventory of the Hugging Face Hub's shared cache."""

from pathlib import Path
from typing import Any

from huggingface_hub import constants, scan_cache_dir
from huggingface_hub.errors import CacheNotFound


def inspect_hf_cache(cache_dir: Path | None = None) -> dict[str, Any]:
    """Report cache usage without assuming that any revision is safe to delete."""
    root = Path(cache_dir or constants.HF_HUB_CACHE)
    try:
        cache = scan_cache_dir(root)
    except CacheNotFound:
        return {
            "path": str(root), "size_bytes": 0, "repository_count": 0,
            "warning_count": 0, "repositories": [], "truncated": False,
        }
    repositories = sorted(cache.repos, key=lambda repo: (repo.repo_type, repo.repo_id))
    return {
        "path": str(root), "size_bytes": cache.size_on_disk,
        "repository_count": len(repositories), "warning_count": len(cache.warnings),
        "truncated": len(repositories) > 100,
        "repositories": [
            {
                "id": repo.repo_id, "type": repo.repo_type, "size_bytes": repo.size_on_disk,
                "revision_count": len(repo.revisions),
            }
            for repo in repositories[:100]
        ],
    }
