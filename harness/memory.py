"""Context elision + summarization: quando o histórico cresce demais, resume
as mensagens mais antigas em vez de deixar o contexto crescer sem limite."""
import os

from model import chat, count_tokens

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "3000"))
KEEP_RECENT = int(os.environ.get("KEEP_RECENT_MESSAGES", "6"))


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


def apply_elision(messages: list[dict]) -> list[dict]:
    """Se o histórico ficou grande demais, resume as mensagens mais antigas e
    substitui por uma única nota de resumo, mantendo o system prompt e as
    mensagens mais recentes intactas."""
    if total_tokens(messages) <= MAX_CONTEXT_TOKENS:
        return messages

    system_msgs = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]

    keep_recent = rest[-KEEP_RECENT:]
    to_drop = rest[:-KEEP_RECENT]

    if not to_drop:
        return messages

    summary = summarize(to_drop)
    summary_msg = {"role": "system", "content": f"[Resumo da conversa anterior]: {summary}"}

    return system_msgs + [summary_msg] + keep_recent
