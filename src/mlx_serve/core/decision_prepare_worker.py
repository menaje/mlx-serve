"""Prepare a decision checkpoint in a disposable process without repository code."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

MANIFEST = "mlx_serve_decision_manifest.json"
SOURCE_REPO = "LiquidAI/d1-omni-600M"
MLX_REPO = "LiquidAI/d1-omni-600M-MLX-fp16"
D3_REPO = "vllm-sr/d3-lite"


def emit(status: str, **data: Any) -> None:
    print("MLX_PREP_STATUS=" + json.dumps({"status": status, **data}), flush=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path, *, converted: bool) -> dict[str, Any]:
    """Check the architecture, head, calibration and every tensor's precision."""
    from safetensors import safe_open

    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if not (path / filename).is_file():
            raise ValueError(f"Missing required d1 file: {filename}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "d1_omni" or not config.get("temperatures"):
        raise ValueError("Checkpoint does not contain the d1 architecture and calibration")
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Checkpoint has no safetensors weights")
    types: set[str] = set()
    keys: set[str] = set()
    for weight in weights:
        with safe_open(weight, framework="numpy") as tensors:
            for key in tensors.keys():
                keys.add(key)
                types.add(tensors.get_slice(key).get_dtype())
    if not any(key.startswith("head.") for key in keys):
        raise ValueError("Checkpoint has no decision head")
    if converted and not any("head.head.layers.0.attention.query_proj.weight" in k for k in keys):
        raise ValueError("Checkpoint does not contain the converted MLX decision head")
    allowed = (
        {"F16", "I32", "I64", "U32", "U64", "BOOL"}
        if converted
        else {"F32", "I32", "I64", "U32", "U64", "BOOL"}
    )
    if not types <= allowed or ("F16" if converted else "F32") not in types:
        raise ValueError(f"Unexpected checkpoint tensor types: {sorted(types)}")
    return config


def sample_prediction(path: Path) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm import load, predict

    model, processor = load(str(path), trust_remote_code=False, strict=True)
    if getattr(model, "model_type", None) != "d1_omni":
        raise ValueError("Loaded model is not d1-omni")
    questions = {
        "animal": {
            "type": "choice",
            "instructions": "Which animal is mentioned?",
            "criteria": {"cat": "a cat", "dog": "a dog"},
        },
        "true": {"type": "bool", "instructions": "Is a cat mentioned?"},
        "score": {
            "type": "score",
            "instructions": "How positive is the statement?",
            "criteria": ["negative", "neutral", "positive"],
        },
    }
    result = predict(model, processor, "I love my cat. It makes me happy.", questions)
    if set(result.get("answers", {})) != set(questions):
        raise ValueError("Incomplete d1 validation response")
    for answer in result["answers"].values():
        if answer["type"] == "bool":
            probability = answer["probability"]
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid Boolean probability")
            if answer["value"] != (probability >= 0.5):
                raise ValueError("Boolean result does not match its probability")
            continue
        probabilities = answer["probabilities"]
        if not probabilities or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
        ):
            raise ValueError("Invalid decision probability distribution")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001):
            raise ValueError("Decision probabilities do not sum to one")
        if answer["type"] == "choice" and answer["value"] not in probabilities:
            raise ValueError("Selected option is absent from the probability distribution")
        if answer["type"] == "score":
            expected = sum(int(level) * p for level, p in probabilities.items())
            if not math.isclose(answer["value"], expected, abs_tol=0.002):
                raise ValueError("Expected score does not match its probability distribution")
    del model, processor
    gc.collect()
    mx.clear_cache()
    return result


def compare_predictions(before: dict[str, Any], after: dict[str, Any]) -> float:
    maximum = 0.0
    for key, original in before["answers"].items():
        converted = after["answers"][key]
        if original["type"] != converted["type"]:
            raise ValueError(f"Converted answer type differs for {key}")
        if original["type"] in ("choice", "bool") and original["value"] != converted["value"]:
            raise ValueError(f"Converted answer differs for {key}")
        if original["type"] == "score" and abs(original["value"] - converted["value"]) > 0.04:
            raise ValueError(f"Converted expected score differs for {key}")
        if "probability" in original:
            differences = [abs(original["probability"] - converted["probability"])]
        else:
            p, q = original["probabilities"], converted["probabilities"]
            if p.keys() != q.keys():
                raise ValueError(f"Converted options differ for {key}")
            differences = [abs(p[name] - q[name]) for name in p]
        maximum = max(maximum, *differences)
    if maximum > 0.02:
        raise ValueError(f"Converted probabilities differ by {maximum:.4f}; limit is 0.02")
    return maximum


def prepare_d3(repo: str, revision: str, source: Path, artifact: Path) -> None:
    """Verify the published checkpoint and run native MLX inference before publication."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from mlx_serve.core.decision import D3DecisionModel

    emit("downloading", hf_repo=repo, revision=revision)
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=str(source),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"],
    )
    shutil.rmtree(source / ".cache", ignore_errors=True)
    emit("checking", detail="Checking D3 architecture and published file hashes")
    published = json.loads((source / "MODEL_MANIFEST.json").read_text())
    if (
        published.get("repo_id") != repo
        or published.get("format_id") != "d3-code-readout-v1"
        or not isinstance(published.get("identity", {}).get("files_sha256"), dict)
    ):
        raise ValueError("Invalid D3 package manifest")
    hashes = published["identity"]["files_sha256"]
    required = {"tokenizer.json", "tokenizer_config.json", "readout.safetensors"}
    if not required <= hashes.keys() or not any(
        name.endswith(".safetensors") and name != "readout.safetensors" for name in hashes
    ):
        raise ValueError("Incomplete D3 package manifest")
    expected_weights = {name for name in hashes if name.endswith(".safetensors")}
    actual_weights = {
        str(path.relative_to(source))
        for path in (
            list(source.glob("*.safetensors")) + list((source / "weights").glob("*.safetensors"))
        )
    }
    if actual_weights != expected_weights:
        raise ValueError("D3 checkpoint has unexpected or missing weight files")
    for name, expected in hashes.items():
        path = source / name
        if (
            not isinstance(name, str) or not isinstance(expected, str)
            or not path.resolve().is_relative_to(source.resolve()) or not path.is_file()
            or file_hash(path) != expected
        ):
            raise ValueError(f"D3 file checksum mismatch: {name}")

    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("dtype") != "bfloat16":
        raise ValueError("Unsupported D3 architecture or precision")
    for name in hashes:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(source / name, framework="numpy") as tensors:
            expected_dtype = "F32" if name == "readout.safetensors" else "BF16"
            if not tensors.keys() or any(
                tensors.get_slice(key).get_dtype() != expected_dtype for key in tensors.keys()
            ):
                raise ValueError(f"Unsupported D3 tensor precision: {name}")

    emit("validating", detail="Running the D3 model with MLX")
    model = D3DecisionModel(source)
    questions = {
        "choice": {"type": "choice", "instructions": "Which animal is mentioned?",
                   "criteria": {"cat": "a cat", "dog": "a dog"}},
        "bool": {"type": "noul", "instructions": "Is a cat mentioned?"},
        "score": {"type": "score", "instructions": "How positive is the statement?",
                  "criteria": ["negative", "neutral", "positive"]},
    }
    result = model.system_one("I love my cat. It makes me happy.", questions)
    answers = result.get("answers", {})
    if set(answers) != set(questions):
        raise ValueError("Incomplete D3 validation response")
    for name, answer in answers.items():
        if name == "bool":
            probability = answer.get("noul")
            if (
                not isinstance(probability, (float, int)) or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Invalid D3 Boolean probability")
            continue
        probabilities = answer.get("probabilities", {})
        if (
            not probabilities
            or any(not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.001)
        ):
            raise ValueError("Invalid D3 decision probability distribution")
        if name == "choice" and answer.get("choice") not in probabilities:
            raise ValueError("Invalid D3 choice")
        if name == "score":
            expected = sum(int(level) * p for level, p in probabilities.items())
            if not math.isclose(answer.get("score", float("nan")), expected, abs_tol=0.002):
                raise ValueError("Invalid D3 expected score")
    del model
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()
    source.rename(artifact)
    files = {
        str(path.relative_to(artifact)): file_hash(path)
        for path in sorted(artifact.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": 1, "source_repo": repo, "revision": revision,
        "architecture": "qwen3_5", "license": os.environ.get("MLX_SERVE_PREPARE_LICENSE"),
        "precision": "bfloat16",
        "converter": None, "converter_options": {"trust_remote_code": False},
        "converted": False, "files_sha256": files, "source_files_sha256": files,
        "environment": {
            "python": sys.version.split()[0],
            **{
                name: version(name)
                for name in ("mlx", "mlx-vlm", "transformers", "huggingface-hub", "safetensors")
            },
        },
        "validation": {"mlx_predict": True, "published_file_hashes": True,
                       "sample_answers": answers},
        "prepared_at": int(time.time()),
    }
    (artifact / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    emit("ready")


def prepare(repo: str, revision: str, source: Path, artifact: Path) -> None:
    from huggingface_hub import snapshot_download

    if version("mlx-vlm") != "0.7.10":
        raise ValueError("Decision preparation requires mlx-vlm 0.7.10")
    if repo == D3_REPO:
        prepare_d3(repo, revision, source, artifact)
        return
    emit("downloading", hf_repo=repo, revision=revision)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(source),
        allow_patterns=["*.json", "*.safetensors", "*.py", "LICENSE", "README.md"],
    )
    # This metadata belongs to our temporary directory, not the shared HF cache.
    shutil.rmtree(source / ".cache", ignore_errors=True)
    emit("checking", detail="Checking architecture, weights and license")
    is_original = repo == SOURCE_REPO
    validate_checkpoint(source, converted=not is_original)
    source_files = {
        str(path.relative_to(source)): file_hash(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    before = None
    if is_original:
        emit("validating", detail="Running the original model")
        before = sample_prediction(source)
        emit("converting", detail="Converting d1 to MLX FP16")
        from mlx_vlm import convert

        convert(str(source), mlx_path=str(artifact), dtype="float16", trust_remote_code=False)
        shutil.copy2(source / "LICENSE", artifact / "LICENSE")
        for unsafe in artifact.glob("*.py"):
            unsafe.unlink()
        config_path = artifact / "config.json"
        config = json.loads(config_path.read_text())
        config["dtype"] = "float16"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    else:
        source.rename(artifact)
    config = validate_checkpoint(artifact, converted=True)
    emit("validating", detail="Running the prepared MLX model")
    after = sample_prediction(artifact)
    difference = compare_predictions(before, after) if before is not None else None
    files = {
        str(path.relative_to(artifact)): file_hash(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "source_repo": repo,
        "revision": revision,
        "architecture": config["model_type"],
        "license": "lfm1.0",
        "precision": "float16",
        "converter": "mlx-vlm" if is_original else None,
        "converter_version": version("mlx-vlm") if is_original else None,
        "environment": {
            "python": sys.version.split()[0],
            **{
                name: version(name)
                for name in ("mlx", "mlx-vlm", "transformers", "huggingface-hub", "safetensors")
            },
        },
        "converter_options": (
            {"dtype": "float16", "trust_remote_code": False}
            if is_original else {"trust_remote_code": False}
        ),
        "converted": is_original,
        "temperatures": config["temperatures"],
        "max_length": config.get("max_length"),
        "files_sha256": files,
        "source_files_sha256": source_files,
        "validation": {
            "official_predict": True,
            "max_probability_difference": difference,
            "tolerance": 0.02,
            "sample_answers": after["answers"],
        },
        "prepared_at": int(time.time()),
    }
    (artifact / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    emit("ready", max_probability_difference=difference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", choices=[SOURCE_REPO, MLX_REPO, D3_REPO])
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
