"""Loop principal: chat + planning + tool-use, com elision/summarization
aplicadas automaticamente quando o histórico cresce."""
import os

from memory import apply_elision
from model import chat
from tools import execute_tool, parse_tool_call, tools_system_prompt

MAX_STEPS = int(os.environ.get("MAX_STEPS", "5"))

SYSTEM_PROMPT = (
    "Você é um assistente local rodando na máquina do usuário. "
    "Quando a tarefa exigir múltiplos passos, primeiro esboce um plano breve "
    "(ex: '1. ... 2. ...') antes de agir, depois execute passo a passo usando "
    "ferramentas quando necessário.\n\n" + tools_system_prompt()
)


def run_turn(messages: list[dict], user_input: str) -> tuple[str, list[dict]]:
    """Processa uma entrada do usuário, rodando o loop de planning/tool-use
    até obter uma resposta final em texto. Retorna (resposta, histórico_atualizado)."""
    messages.append({"role": "user", "content": user_input})

    for _ in range(MAX_STEPS):
        messages = apply_elision(messages)
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})

        call = parse_tool_call(reply)
        if call is None:
            return reply, messages

        result = execute_tool(call)
        messages.append(
            {"role": "user", "content": f"[resultado da ferramenta '{call.get('tool')}']: {result}"}
        )

    return "Limite de passos atingido sem resposta final.", messages


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Harness local pronto. Digite 'sair' para encerrar.\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            break
        reply, messages = run_turn(messages, user_input)
        print(f"\nModelo: {reply}\n")


if __name__ == "__main__":
    main()
