"""Function calling via convenção baseada em prompt: o modelo emite um bloco
```tool_call``` em JSON, o harness faz o parse e executa a ferramenta localmente.

Ferramentas: read, write, grep, glob — todas restritas a HARNESS_ROOT (por
padrão, o diretório de trabalho de onde o harness foi iniciado), para evitar
que o modelo leia/escreva fora da área permitida.
"""
import fnmatch
import json
import os
import re

ROOT = os.path.abspath(os.environ.get("HARNESS_ROOT", os.getcwd()))

TOOLS = {
    "read": {
        "description": "Lê um arquivo de texto. offset/limit (opcionais) selecionam um intervalo de linhas.",
        "params": {"path": "string", "offset": "int?", "limit": "int?"},
    },
    "write": {
        "description": "Cria ou sobrescreve um arquivo de texto com o conteúdo dado.",
        "params": {"path": "string", "content": "string"},
    },
    "grep": {
        "description": "Busca um padrão regex no conteúdo dos arquivos sob um diretório.",
        "params": {"pattern": "string", "path": "string?", "glob": "string?"},
    },
    "glob": {
        "description": "Lista arquivos que casam com um padrão glob (ex: '**/*.py').",
        "params": {"pattern": "string", "path": "string?"},
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


class PathEscapesRoot(Exception):
    pass


def _resolve(path: str) -> str:
    """Resolve `path` (relativo ou absoluto) para dentro de ROOT, recusando
    qualquer caminho que escape dele (ex: via '..')."""
    candidate = os.path.abspath(os.path.join(ROOT, path))
    if not (candidate == ROOT or candidate.startswith(ROOT + os.sep)):
        raise PathEscapesRoot(path)
    return candidate


def _tool_read(args: dict) -> str:
    full = _resolve(args.get("path", ""))
    if not os.path.isfile(full):
        return f"Erro: arquivo não encontrado: {args.get('path')}"

    offset = args.get("offset")
    limit = args.get("limit")
    with open(full, "r", errors="replace") as f:
        lines = f.readlines()

    start = max(offset - 1, 0) if offset else 0
    end = start + limit if limit else len(lines)
    selected = lines[start:end]

    numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(selected)]
    return "".join(numbered)[:8000]  # limite para não explodir o contexto


def _tool_write(args: dict) -> str:
    full = _resolve(args.get("path", ""))
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w") as f:
        f.write(args.get("content", ""))
    return f"Arquivo escrito: {args.get('path')} ({len(args.get('content', ''))} bytes)"


def _tool_grep(args: dict) -> str:
    base = _resolve(args.get("path", "."))
    pattern = re.compile(args.get("pattern", ""))
    name_glob = args.get("glob")

    matches = []
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            if name_glob and not fnmatch.fnmatch(name, name_glob):
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", errors="strict") as f:
                    for lineno, line in enumerate(f, start=1):
                        if pattern.search(line):
                            rel = os.path.relpath(full, ROOT)
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
            except (UnicodeDecodeError, OSError):
                continue  # pula arquivos binários/ilegíveis
            if len(matches) >= 200:
                break
        if len(matches) >= 200:
            break

    if not matches:
        return "Nenhum resultado."
    return "\n".join(matches)[:8000]


def _tool_glob(args: dict) -> str:
    base = _resolve(args.get("path", "."))
    pattern = args.get("pattern", "*")

    results = []
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, base)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                results.append(os.path.relpath(full, ROOT))

    if not results:
        return "Nenhum arquivo encontrado."
    return "\n".join(sorted(results)[:200])


_HANDLERS = {
    "read": _tool_read,
    "write": _tool_write,
    "grep": _tool_grep,
    "glob": _tool_glob,
}


def execute_tool(call: dict) -> str:
    name = call.get("tool")
    args = call.get("args", {})

    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Erro: tool desconhecida '{name}'"

    try:
        return handler(args)
    except PathEscapesRoot as e:
        return f"Erro: caminho fora da área permitida (HARNESS_ROOT): {e}"
    except Exception as e:
        return f"Erro ao executar '{name}': {e}"
