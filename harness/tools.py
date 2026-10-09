"""Function calling via convenção baseada em prompt: o modelo emite um bloco
```tool_call``` em JSON, o harness faz o parse e executa a ferramenta localmente.
"""
import json
import os
import re

TOOLS = {
    "read_file": {
        "description": "Lê o conteúdo de um arquivo de texto local.",
        "params": {"path": "string"},
    },
    "calculator": {
        "description": "Avalia uma expressão aritmética simples.",
        "params": {"expression": "string"},
    },
}

TOOL_CALL_RE = re.compile(r"```tool_call\s*(\{.*?\})\s*```", re.DOTALL)


def tools_system_prompt() -> str:
    lines = ["Você tem acesso às seguintes ferramentas:"]
    for name, spec in TOOLS.items():
        lines.append(f"- {name}({spec['params']}): {spec['description']}")
    lines.append(
        "\nPara usar uma ferramenta, responda APENAS com um bloco:\n"
        "```tool_call\n"
        '{"tool": "<nome>", "args": {...}}\n'
        "```\n"
        "Se não precisar de ferramenta, responda normalmente em texto."
    )
    return "\n".join(lines)


def parse_tool_call(text: str):
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def execute_tool(call: dict) -> str:
    name = call.get("tool")
    args = call.get("args", {})

    if name == "read_file":
        path = args.get("path", "")
        if not os.path.isfile(path):
            return f"Erro: arquivo não encontrado: {path}"
        with open(path, "r", errors="replace") as f:
            return f.read()[:4000]  # limite para não explodir o contexto

    if name == "calculator":
        expr = args.get("expression", "")
        try:
            # eval restrito só a aritmética — nada de builtins/import.
            # Seguro apenas neste contexto de laboratório local; não exponha
            # esse padrão a entradas não confiáveis fora disso.
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Erro ao avaliar expressão: {e}"

    return f"Erro: tool desconhecida '{name}'"
