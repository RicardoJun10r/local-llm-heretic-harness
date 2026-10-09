#!/usr/bin/env bash
# Prepara o ambiente: instala uv (se preciso), cria venv, instala torch com o
# índice certo (CUDA se detectar GPU NVIDIA, senão CPU), e as dependências do
# projeto (transformers, heretic-llm, llama-cpp-python).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo ">> Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> Criando venv..."
uv venv .venv
source .venv/bin/activate

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ">> GPU NVIDIA detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ">> Instalando torch com suporte CUDA (cu124)..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    export LLAMA_CUDA=1
    export CMAKE_ARGS="-DGGML_CUDA=on"
else
    echo ">> Nenhuma GPU NVIDIA detectada — instalando torch CPU-only."
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    unset CMAKE_ARGS
fi

echo ">> Instalando dependências do projeto..."
uv pip install -e .

echo ">> Instalando llama-cpp-python (harness de inferência)..."
uv pip install llama-cpp-python

echo ""
echo "=== Verificação ==="
python - <<'PYEOF'
import torch
print("torch:", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYEOF

echo ""
echo "Setup concluído. Ative o ambiente com: source .venv/bin/activate"
