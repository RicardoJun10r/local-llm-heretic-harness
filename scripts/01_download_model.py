#!/usr/bin/env python3
"""Baixa um modelo base (formato Hugging Face / safetensors) para uso com o Heretic.

Uso:
    python scripts/01_download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct
    python scripts/01_download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama-3.1-8b

Se o modelo exigir aceite de licença (ex: Llama), rode antes:
    huggingface-cli login
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="ID do modelo no Hugging Face Hub (padrão: Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Diretório local de destino (padrão: models/<nome-do-repo>).",
    )
    args = parser.parse_args()

    local_dir = args.local_dir or f"models/{args.repo_id.split('/')[-1].lower()}"
    Path(local_dir).parent.mkdir(parents=True, exist_ok=True)

    print(f">> Baixando {args.repo_id} para {local_dir} ...")
    path = snapshot_download(repo_id=args.repo_id, local_dir=local_dir)
    print(f"Concluído: {path}")


if __name__ == "__main__":
    main()
