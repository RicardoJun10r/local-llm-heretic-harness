# local-llm-heretic-harness

LLM local, com remoção de guardrails via [Heretic](https://github.com/p-e-w/heretic) e um
harness próprio em Python (chat loop, context elision, summarization, function calling e
planning). Pensado para rodar numa máquina com GPU NVIDIA, mas os scripts também funcionam
em CPU-only (mais lento).

## Visão geral do pipeline

```
1. Baixar modelo base (HF/safetensors)
         │
2. Heretic → remove guardrails (abliteração)
         │
3. Converter + quantizar → GGUF
         │
4. Harness próprio (chat/tools/planning) ← roda sobre o GGUF via llama-cpp-python
```

## Requisitos

- Linux (testado em Ubuntu) com Python 3.10+
- Para aceleração GPU: drivers NVIDIA + CUDA instalados (`nvidia-smi` funcionando)
- `git`, `cmake`, `build-essential` (ou equivalente)
- [`gh`](https://cli.github.com/) não é necessário para rodar o projeto, só para quem for
  contribuir

## 1. Setup

```bash
git clone <este-repo>
cd local-llm-heretic-harness
chmod +x scripts/*.sh
./scripts/00_setup.sh
source .venv/bin/activate
```

O script detecta automaticamente se há GPU NVIDIA (`nvidia-smi`) e instala o PyTorch com o
índice certo (CUDA ou CPU), além de `llama-cpp-python` já preparado para offload em GPU
quando aplicável.

## 2. Escolher e baixar o modelo base

Escolha o modelo pelo tamanho de VRAM/RAM disponível na máquina:

| VRAM da GPU | Modelo sugerido                     | Quantização no Heretic |
|-------------|--------------------------------------|-------------------------|
| ~8GB        | `Qwen/Qwen2.5-7B-Instruct`            | `BNB_4BIT`              |
| ~16GB+      | `Qwen/Qwen2.5-7B-Instruct` ou 14B     | `NONE`                  |
| CPU-only    | `Qwen/Qwen2.5-3B-Instruct` ou menor   | `NONE`                  |

Se o modelo exigir aceite de licença (ex: família Llama):

```bash
huggingface-cli login
```

Baixar:

```bash
python scripts/01_download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct
# salva em models/qwen2.5-7b-instruct por padrão
```

## 3. Abliteração com Heretic

```bash
./scripts/02_run_heretic.sh models/qwen2.5-7b-instruct
```

Variáveis de ambiente opcionais:

```bash
N_TRIALS=200 QUANTIZATION=BNB_4BIT ./scripts/02_run_heretic.sh models/qwen2.5-7b-instruct
```

- `N_TRIALS` (default `200`): número de trials de otimização do Heretic. Em GPU isso roda
  em uma fração do tempo que levaria em CPU — mas se a máquina for mais modesta, reduza
  (ex: `N_TRIALS=40`) para um resultado mais rápido e um pouco menos otimizado.
- `QUANTIZATION=BNB_4BIT`: use se a VRAM for curta para o modelo escolhido em fp16/bf16.

O modelo abliterado é exportado (merge) para uma pasta sob `heretic-checkpoints/` — o
caminho exato aparece no final da execução do comando.

**Checkpoint de validação:** teste o modelo resultante com um prompt que o modelo
original recusaria, e compare a resposta antes/depois. O próprio Heretic já imprime
métricas de taxa de recusa durante a otimização (`--print-responses` para ver os
prompts/respostas usados na avaliação).

## 4. Converter para GGUF (inferência eficiente)

```bash
./scripts/03_convert_gguf.sh heretic-checkpoints/<pasta-do-modelo-abliterado>
```

Gera um `.gguf` quantizado (Q4_K_M por padrão) em `models/gguf/`. Para outra quantização:

```bash
QUANT=Q5_K_M ./scripts/03_convert_gguf.sh heretic-checkpoints/<pasta-do-modelo-abliterado>
```

Se a máquina tem GPU, o `llama.cpp` é compilado com suporte CUDA automaticamente.

## 5. Harness próprio (chat + tools + planning)

```bash
export MODEL_PATH=models/gguf/<nome>-q4_k_m.gguf
export N_GPU_LAYERS=-1   # offload de todas as camadas para a GPU; use 0 para CPU-only
cd harness
python agent.py
```

O harness (`harness/`) implementa:

- **`model.py`** — carrega o GGUF via `llama-cpp-python`, com `N_GPU_LAYERS` configurável.
- **`tools.py`** — function calling via convenção de prompt (`tool_call` em JSON), com
  duas ferramentas de exemplo (`read_file`, `calculator`).
- **`memory.py`** — context elision: quando o histórico passa de `MAX_CONTEXT_TOKENS`
  (default 3000), as mensagens mais antigas são resumidas pelo próprio modelo e
  substituídas por uma nota de resumo, em vez de crescer sem limite.
- **`agent.py`** — loop de planning/execução: o system prompt instrui o modelo a esboçar
  um plano antes de agir; o harness itera chamando ferramentas até obter uma resposta
  final ou atingir `MAX_STEPS` (default 5).

Variáveis de ambiente do harness: `MODEL_PATH`, `N_CTX`, `N_THREADS`, `N_GPU_LAYERS`,
`MAX_CONTEXT_TOKENS`, `KEEP_RECENT_MESSAGES`, `MAX_STEPS`.

### Roteiro de teste sugerido

1. Pergunta simples (sem tool): `"O que é fotossíntese?"`
2. Pergunta que exige tool: `"Quanto é 123 * 456?"`
3. Pergunta multi-passo: `"Leia o arquivo /etc/hostname e me diga quantos caracteres tem."`
4. Conversa longa (15-20 turnos) para forçar a elision/summarization a disparar.

## Notas e limitações

- **Uso responsável**: o modelo abliterado é para experimentação/pesquisa local. A
  responsabilidade pelo uso do conteúdo gerado é de quem opera o sistema.
- **Qualidade**: modelos pequenos (1B-7B) têm function calling e raciocínio mais fracos que
  modelos grandes — o parsing de `tool_call` em `tools.py` já trata o caso de tool
  desconhecida/alucinada.
- **CPU-only**: funciona, mas o próprio Heretic já avisa que operações ficam lentas sem
  GPU. Para modelos 7B+, GPU é fortemente recomendado tanto para a abliteração quanto para
  a inferência depois.

## Referências

- Heretic: [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic) (pacote PyPI: `heretic-llm`)
- llama.cpp: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- llama-cpp-python: bindings Python sobre o llama.cpp
