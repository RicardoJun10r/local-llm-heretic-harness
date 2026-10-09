#!/usr/bin/env bash
# Converte um modelo (safetensors) para GGUF e quantiza (Q4_K_M por padrão),
# para inferência eficiente via llama.cpp / llama-cpp-python.
#
# Uso:
#   ./scripts/03_convert_gguf.sh heretic-checkpoints/qwen2.5-7b-instruct-abliterated
#   QUANT=Q5_K_M ./scripts/03_convert_gguf.sh <caminho-do-modelo-abliterado>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

MODEL_DIR="${1:?Uso: $0 <caminho-do-modelo-abliterado-em-safetensors>}"
QUANT="${QUANT:-Q4_K_M}"
MODEL_NAME="$(basename "$MODEL_DIR")"
OUT_DIR="models/gguf"
mkdir -p "$OUT_DIR"

if [ ! -d "llama.cpp" ]; then
    echo ">> Clonando llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp
fi

cd llama.cpp
uv pip install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
    echo ">> Compilando llama.cpp com suporte CUDA..."
    cmake -B build -DGGML_CUDA=ON
else
    echo ">> Compilando llama.cpp otimizado para CPU (AVX2/AVX512)..."
    cmake -B build -DGGML_NATIVE=ON
fi
cmake --build build --config Release -j"$(nproc)"
cd ..

F16_PATH="$OUT_DIR/${MODEL_NAME}-f16.gguf"
QUANT_PATH="$OUT_DIR/${MODEL_NAME}-${QUANT,,}.gguf"

echo ">> Convertendo para GGUF (f16)..."
python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$F16_PATH" --outtype f16

echo ">> Quantizando para $QUANT..."
./llama.cpp/build/bin/llama-quantize "$F16_PATH" "$QUANT_PATH" "$QUANT"

echo ""
echo "Pronto: $QUANT_PATH"
echo "Teste rápido:"
echo "  ./llama.cpp/build/bin/llama-cli -m $QUANT_PATH -p \"Olá, quem é você?\" -n 100"
