"""Download and validate known MLX retrieval models without repository code."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mlx_serve.core.decision_prepare_worker import emit, file_hash

EMBEDDING_REPO = "mlx-community/all-MiniLM-L6-v2-bf16"
RERANKER_REPO = "mlx-community/Qwen3-Reranker-0.6B-4bit"
RETRIEVAL_PROFILES = {EMBEDDING_REPO: "embedding", RERANKER_REPO: "reranker"}
MANIFEST = "mlx_serve_retrieval_manifest.json"
DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"]


def _git_blob_hash(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_published_files(source: Path, siblings: list) -> dict[str, str]:
    """Verify every downloaded file against immutable Hub git/LFS metadata."""
    published = {item.rfilename: item for item in siblings}
    files = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked model files are not supported")
        if not path.is_file():
            continue
        name = str(path.relative_to(source))
        info = published.get(name)
        if info is None or getattr(info, "size", None) != path.stat().st_size:
            raise ValueError(f"Model file is missing from published metadata: {name}")
        sha256 = file_hash(path)
        lfs = getattr(info, "lfs", None)
        expected = getattr(lfs, "sha256", None) if lfs is not None else None
        if expected:
            if sha256 != expected:
                raise ValueError(f"Published model file checksum mismatch: {name}")
        elif not getattr(info, "blob_id", None) or _git_blob_hash(path) != info.blob_id:
            raise ValueError(f"Published model file checksum mismatch: {name}")
        files[name] = sha256
    if not {"config.json", "tokenizer.json"} <= files.keys() or not any(
        name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Missing required retrieval model files")
    expected_files = {
        name for name in published if any(fnmatch.fnmatchcase(name, p) for p in DOWNLOAD_PATTERNS)
    }
    if set(files) != expected_files:
        raise ValueError("Downloaded model files do not match the published revision")
    return files


def sample_prediction(repo: str, path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    if repo == EMBEDDING_REPO:
        from mlx_embeddings import load
        from safetensors import safe_open

        from mlx_serve.core.batch_processor import EmbeddingBatchProcessor

        if config.get("model_type") != "bert":
            raise ValueError("Unsupported embedding model architecture")
        weights = list(path.glob("*.safetensors"))
        if not weights:
            raise ValueError("Missing embedding model weights")
        dtypes = set()
        for weight in weights:
            with safe_open(weight, framework="numpy") as tensors:
                dtypes.update(tensors.get_slice(key).get_dtype() for key in tensors.keys())
        if "BF16" not in dtypes or not dtypes <= {"BF16", "I32", "I64", "U32", "U64", "BOOL"}:
            raise ValueError("Unsupported embedding weight precision")
        model, tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
        vectors = EmbeddingBatchProcessor(model, tokenizer)._generate_embeddings([["A small cat."]])
        if (len(vectors) != 1 or len(vectors[0]) != 1 or len(vectors[0][0]) != 384
                or not all(math.isfinite(float(item)) for item in vectors[0][0])):
            raise ValueError("Embedding validation returned an invalid vector")
        return {"mlx_predict": True, "published_file_hashes": True, "dimensions": 384}
    if repo == RERANKER_REPO:
        from mlx_lm import load

        from mlx_serve.core.batch_processor import RerankBatchProcessor

        if config.get("model_type") != "qwen3" or config.get("quantization", {}).get("bits") != 4:
            raise ValueError("Unsupported reranker architecture or quantization")
        model, tokenizer = load(str(path), trust_remote_code=False)
        scores, _ = RerankBatchProcessor(model, tokenizer).compute_scores(
            "A cat sat on the mat", ["A cat sat on the mat.", "The ocean is blue."]
        )
        if (len(scores) != 2
                or not all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
                or scores[0] <= scores[1]):
            raise ValueError("Reranker validation returned an invalid ranking")
        return {"mlx_predict": True, "published_file_hashes": True, "ranking_check": True}
    raise ValueError("Unsupported retrieval model")


def prepare(repo: str, revision: str, source: Path, artifact: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    emit("downloading", hf_repo=repo, revision=revision)
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(source),
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    shutil.rmtree(source / ".cache", ignore_errors=True)
    emit("checking", detail="Verifying published file hashes")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("The retrieved model revision changed")
    files = verify_published_files(source, info.siblings or [])
    emit("validating", detail="Running MLX inference")
    validation = sample_prediction(repo, source)
    manifest = {
        "schema_version": 1, "source_repo": repo, "revision": revision,
        "type": RETRIEVAL_PROFILES[repo],
        "architecture": "bert" if repo == EMBEDDING_REPO else "qwen3",
        "precision": "bfloat16" if repo == EMBEDDING_REPO else "4bit",
        "license": os.environ.get("MLX_SERVE_PREPARE_LICENSE"),
        "converted": False, "converter": None, "converter_options": {},
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "files_sha256": files, "validation": validation,
    }
    source.rename(artifact)
    (artifact / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    emit("ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", choices=list(RETRIEVAL_PROFILES))
    parser.add_argument("revision")
    parser.add_argument("source", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        prepare(args.repo, args.revision, args.source, args.artifact)
    except Exception as exc:
        emit("error", message=str(exc))
        raise


if __name__ == "__main__":
    main()
