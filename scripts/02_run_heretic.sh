#!/usr/bin/env bash
# Roda a abliteração com o Heretic sobre um modelo local.
#
# Uso:
#   ./scripts/02_run_heretic.sh models/qwen2.5-7b-instruct
#   N_TRIALS=60 QUANTIZATION=BNB_4BIT ./scripts/02_run_heretic.sh models/qwen2.5-7b-instruct
#
# Variáveis de ambiente (todas opcionais, com defaults sensatos p/ GPU):
#   N_TRIALS        número de trials de otimização (default: 200, o padrão do Heretic)
#   QUANTIZATION    NONE | BNB_4BIT (use BNB_4BIT se a VRAM for curta para o modelo, ex: 8GB)
#   DEVICE_MAP      default: auto (deixa o Accelerate decidir GPU/CPU)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

MODEL_PATH="${1:?Uso: $0 <caminho-ou-repo-id-do-modelo>}"
N_TRIALS="${N_TRIALS:-200}"
QUANTIZATION="${QUANTIZATION:-NONE}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

echo ">> Modelo: $MODEL_PATH"
echo ">> Trials: $N_TRIALS | Quantização: $QUANTIZATION | Device map: $DEVICE_MAP"

heretic \
    --model "$MODEL_PATH" \
    --device-map "$DEVICE_MAP" \
    --quantization "$QUANTIZATION" \
    --n-trials "$N_TRIALS" \
    --export-strategy MERGE \
    --study-checkpoint-dir heretic-checkpoints

echo ""
echo "Abliteração concluída. O modelo exportado deve estar em uma pasta indicada"
echo "na saída acima (geralmente algo como heretic-checkpoints/<nome>-abliterated/)."
