"""Wrapper fino sobre llama-cpp-python: carrega o modelo GGUF e expõe chat()."""
import os

from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "models/gguf/model-q4_k_m.gguf")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_THREADS = int(os.environ.get("N_THREADS", str(os.cpu_count() or 4)))
# -1 offloda todas as camadas para a GPU (se o binário foi compilado com CUDA);
# 0 mantém tudo na CPU.
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)


def chat(messages: list[dict], max_tokens: int = 512, stop=None) -> str:
    """messages: lista de dicts {"role": ..., "content": ...}"""
    out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, stop=stop)
    return out["choices"][0]["message"]["content"]


def count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))
