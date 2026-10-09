"""Context elision + summarization, em dois limiares baseados em % da janela
de contexto do modelo (N_CTX):

- >= ELISION_LOWER_RATIO (default 60%): elision simples — descarta as
  mensagens mais antigas, mantendo só um marcador de quantas foram removidas.
- >= ELISION_UPPER_RATIO (default 80%): elision + summarization — antes de
  descartar, pede ao próprio modelo um resumo condensado das mensagens
  antigas e injeta esse resumo no lugar delas.

Em ambos os casos, o system prompt original e as últimas KEEP_RECENT
mensagens são preservados intactos.
"""
import os

from model import N_CTX, chat, count_tokens

LOWER_RATIO = float(os.environ.get("ELISION_LOWER_RATIO", "0.6"))
UPPER_RATIO = float(os.environ.get("ELISION_UPPER_RATIO", "0.8"))
KEEP_RECENT = int(os.environ.get("KEEP_RECENT_MESSAGES", "6"))

# Tag usada para identificar e substituir (em vez de acumular) a mensagem de
# elision/resumo entre chamadas sucessivas.
_MARKER_TAG = "[memória condensada]"


def total_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) for m in messages)


def summarize(messages_to_drop: list[dict]) -> str:
    """Pede ao próprio modelo um resumo condensado das mensagens descartadas."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_drop)
    summary_prompt = [
        {
            "role": "system",
            "content": "Resuma a conversa abaixo em até 5 frases, preservando fatos e decisões importantes.",
        },
        {"role": "user", "content": transcript},
    ]
    return chat(summary_prompt, max_tokens=200)


def _is_marker(msg: dict) -> bool:
    return msg["role"] == "system" and msg["content"].startswith(_MARKER_TAG)


def apply_elision(messages: list[dict]) -> list[dict]:
    tokens = total_tokens(messages)
    lower = N_CTX * LOWER_RATIO
    upper = N_CTX * UPPER_RATIO

    if tokens < lower:
        return messages

    # system prompt original (exclui marcadores de elision de rodadas anteriores)
    system_msgs = [m for m in messages if m["role"] == "system" and not _is_marker(m)]
    rest = [m for m in messages if m["role"] != "system" and not _is_marker(m)]

    keep_recent = rest[-KEEP_RECENT:]
    to_drop = rest[:-KEEP_RECENT]

    if not to_drop:
        return messages

    if tokens >= upper:
        # Pressão alta: elision + summarization.
        summary = summarize(to_drop)
        marker = {
            "role": "system",
            "content": f"{_MARKER_TAG} Resumo da conversa anterior ({len(to_drop)} mensagens): {summary}",
        }
    else:
        # Pressão moderada: elision simples, sem gastar uma chamada ao modelo.
        marker = {
            "role": "system",
            "content": f"{_MARKER_TAG} {len(to_drop)} mensagens antigas foram removidas para liberar espaço de contexto.",
        }

    return system_msgs + [marker] + keep_recent
