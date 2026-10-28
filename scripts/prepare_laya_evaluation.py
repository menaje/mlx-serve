"""Prepare a local Laya source checkpoint for the mlx-vlm evaluation script.

This creates an experimental standalone directory, not an mlx-serve installation.
See mlx-vlm's laya model README for the required checkpoint layout.
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Downloaded 2nugu/laya-ko source directory")
    parser.add_argument("destination", type=Path, help="New directory for the MLX evaluation")
    args = parser.parse_args()
    source, destination = args.source, args.destination
    encoder = json.loads((source / "encoder" / "config.json").read_text())
    decision = json.loads((source / "rl_agent_config.json").read_text())
    if encoder.get("model_type") != "modernbert" or "head_layers" not in decision:
        raise ValueError("Unsupported Laya source checkpoint")
    weights = source / "model.safetensors"
    tokenizer = source / "tokenizer"
    if not weights.is_file() or not (tokenizer / "tokenizer.json").is_file():
        raise ValueError("Laya weights or tokenizer are missing")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        config = {
            "model_type": "laya",
            "encoder_config": encoder,
            "decision_config": decision,
            "head_layers": decision["head_layers"],
        }
        (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
        for path in [weights, *tokenizer.iterdir()]:
            if not path.is_file():
                continue
            shutil.copy2(path, destination / path.name)
        if (source / "LICENSE").is_file():
            shutil.copy2(source / "LICENSE", destination / "LICENSE")
    except Exception:
        shutil.rmtree(destination)
        raise
    print(destination)


if __name__ == "__main__":
    main()
