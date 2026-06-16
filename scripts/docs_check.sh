#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$repo_root" <<'PY'
import pathlib
import re
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
docs_root = repo_root / "docs"

external_url = re.compile(r"^(?:[a-z][a-z0-9+.-]*:)?//|^[a-z][a-z0-9+.-]*:")
inline_link = re.compile(r"(?:!)?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
reference_def = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(\S+)", re.MULTILINE)
fence = re.compile(r"```([A-Za-z0-9_-]+)?\n(.*?)```", re.DOTALL)
heading = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
strip_markup = re.compile(r"<[^>]+>|`([^`]*)`|!\[[^\]]*\]\([^)]*\)|\[([^\]]*)\]\([^)]*\)")
punct = re.compile(r"[^\w\s-]")
space = re.compile(r"\s+")
supported_diagrams = {
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "mindmap",
    "gitGraph",
    "requirementDiagram",
    "block-beta",
    "quadrantChart",
    "xychart-beta",
}


def anchor_for(text: str) -> str:
    text = strip_markup.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = punct.sub("", text).strip().lower()
    return space.sub("-", text)


def collect_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in markdown.splitlines():
        match = heading.match(line)
        if not match:
            continue
        base = anchor_for(match.group(1))
        if not base:
            continue
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def resolve_link(source: pathlib.Path, raw: str) -> tuple[pathlib.Path | None, str | None]:
    if not raw:
        return None, None
    if raw.startswith("#"):
        return source, raw[1:]
    if external_url.match(raw):
        return None, None
    target = raw.split("#", 1)
    path_part = target[0]
    fragment = target[1] if len(target) == 2 else None
    if not path_part:
        return source, fragment
    path = (source.parent / path_part).resolve()
    return path, fragment


def check_links() -> list[str]:
    errors: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        anchors = collect_anchors(markdown)
        refs = {name: target for name, target in reference_def.findall(markdown)}
        link_targets = inline_link.findall(markdown) + list(refs.values())
        for raw in link_targets:
            raw = raw.split("#", 1)[0] if "#" in raw else raw
            is_generated_api_link = (
                raw.startswith("api/html/")
                or raw.startswith("generated/api/html/")
                or raw.startswith("generated/api/reference/html/")
            )
            if raw.startswith("#") or external_url.match(raw) or is_generated_api_link:
                continue
            resolved, fragment = resolve_link(path, raw)
            if resolved is None:
                continue
            if not resolved.is_file():
                errors.append(f"{path.relative_to(repo_root)}: broken link to {raw}")
                continue
            if fragment and fragment not in anchors:
                errors.append(f"{path.relative_to(repo_root)}: missing anchor #{fragment} for {raw}")
    return errors


def balanced(text: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_string: str | None = None
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
            continue
        if char in {'"', "'"}:
            in_string = char
            continue
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return not stack and in_string is None


def lint_mermaid(path: pathlib.Path, markdown: str) -> list[str]:
    errors: list[str] = []
    for index, (info, body) in enumerate(fence.findall(markdown), start=1):
        if (info or "").strip() != "mermaid":
            continue
        diagram = body.strip()
        location = f"{path.relative_to(repo_root)}:mermaid block {index}"
        if not diagram:
            errors.append(f"{location}: empty diagram")
            continue
        lines = [line for line in diagram.splitlines() if line.strip() and not line.lstrip().startswith("%%")]
        if not lines:
            errors.append(f"{location}: empty diagram")
            continue
        first = lines[0].strip().split()[0]
        if first not in supported_diagrams:
            errors.append(f"{location}: unsupported or missing diagram type '{first}'")
        if "\t" in diagram:
            errors.append(f"{location}: tabs are not allowed")
        if not balanced(diagram):
            errors.append(f"{location}: unbalanced parentheses, brackets, braces, or quotes")
        if first in {"flowchart", "graph"}:
            for line in lines[1:]:
                if any(marker in line for marker in ("-->", "---", "==>", "-.->", "==>")):
                    parts = re.split(r"-->|---|==>|-.->", line, maxsplit=1)
                    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                        errors.append(f"{location}: malformed edge line: {line}")
        if first == "mindmap":
            for line in lines[1:]:
                if line and not re.match(r"^[ ]+\S", line):
                    errors.append(f"{location}: mindmap child lines must be indented: {line}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        errors.extend(check_links())
        errors.extend(lint_mermaid(path, markdown))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("docs_check: links and Mermaid diagrams are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
