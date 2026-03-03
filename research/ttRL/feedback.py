from __future__ import annotations

import ast
import re


_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_NON_CODE_PREFIXES = (
    "do not",
    "don't",
    "start immediately",
    "provide only",
    "return only",
    "task:",
    "test failure:",
    "feedback:",
    "candidate",
    "corrected function body",
    "the corrected function body",
    "here is",
)
_CODE_LINE = re.compile(
    r"^((return|if|elif|for|while|except|with|assert|raise|import|from|pass|break|continue)\b|((else|try|finally)\s*:))"
)
_ASSIGN_LINE = re.compile(r"^[A-Za-z_]\w*\s*([+\-*/%&|^]?=)")
_CALL_LINE = re.compile(r"^[A-Za-z_]\w*\s*\(")
_PROSE_PREFIXES = (
    "explanation:",
    "analysis:",
    "reasoning:",
    "notes:",
    "note:",
    "why:",
)
_BAD_INLINE_COMMENT = re.compile(
    r"(corrected function body|corrected body|original body|prose|insert code|fill in|end of code)",
    re.IGNORECASE,
)
_BLOCK_HEADER = re.compile(r".*:\s*(#.*)?$")
_DEDENT_LEAD = re.compile(r"^(elif\b|else\s*:|except\b|finally\s*:)")
_TOPLEVEL_LIKELY = re.compile(r"^(return\b|raise\b|pass$)")
_TRIVIAL_RETURN = re.compile(
    r"^return\s+([A-Za-z_]\w*|none|true|false)\s*$",
    re.IGNORECASE,
)


def extract_code(text: str) -> str:
    """Extract code from a model response.

    If fenced code blocks exist, return the first non-empty block; otherwise
    return raw text.
    """
    matches = _CODE_FENCE.findall(text)
    for block in matches:
        stripped = block.strip()
        # Ignore malformed captures that still contain fence markers.
        if stripped and "```" not in stripped:
            return stripped
    if "```" in text:
        # Fallback for malformed fence sequences (e.g., repeated dangling fences).
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        stripped = "\n".join(lines).strip()
        if stripped:
            for chunk in re.split(r"\n\s*\n+", stripped):
                candidate = chunk.strip()
                if not candidate:
                    continue
                if re.search(
                    r"(?m)^\s*(for|if|while|return|def|import|from|assert)\b",
                    candidate,
                ):
                    return candidate
            return stripped
    return text.strip()


def _extract_function_body(code: str, entry_point: str) -> str | None:
    if not entry_point:
        return None

    # Handle one-line definitions such as:
    # def f(x): return x + 1
    inline_pat = re.compile(
        rf"^[ \t]*def\s+{re.escape(entry_point)}\s*\(.*\)\s*(?:->\s*[^:]+)?\s*:[ \t]*(.+?)\s*$",
        re.MULTILINE,
    )
    inline_match = inline_pat.search(code)
    if inline_match:
        inline_body = inline_match.group(1).strip()
        if inline_body:
            return "    " + inline_body

    pat = re.compile(
        rf"^([ \t]*)def\s+{re.escape(entry_point)}\s*\(.*\)\s*(?:->\s*[^:]+)?\s*:[ \t]*$",
        re.MULTILINE,
    )
    match = pat.search(code)
    if not match:
        return None

    start = match.start()
    lines = code[start:].splitlines()
    if not lines:
        return None

    def_line = lines[0]
    def_indent = len(def_line) - len(def_line.lstrip(" "))
    body_lines = []
    for line in lines[1:]:
        if not line.strip():
            body_lines.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= def_indent:
            break
        relative = line[def_indent:] if len(line) > def_indent else line
        if relative.startswith("    "):
            relative = relative[4:]
        body_lines.append(relative)

    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    if not body_lines:
        return None

    return "\n".join(("    " + ln) if ln else "" for ln in body_lines)


def _looks_like_python_line(stripped: str) -> bool:
    if not stripped:
        return False
    if _CODE_LINE.match(stripped):
        return True
    if _ASSIGN_LINE.match(stripped):
        return True
    if _CALL_LINE.match(stripped):
        return True
    if stripped.startswith(("#", "@", "[", "{", "(")):
        return True
    return False


def _body_parses(body: str) -> bool:
    try:
        ast.parse(f"def _f():\n{body}\n")
    except SyntaxError:
        return False
    return True


def _coerce_indentation(body: str) -> str:
    lines = []
    raw_lines = body.splitlines()
    for line in raw_lines:
        if not line.strip():
            lines.append("")
            continue
        expanded = line.expandtabs(4)
        stripped = expanded.lstrip(" ")
        indent = len(expanded) - len(stripped)
        if indent == 0:
            indent = 4
        if indent % 4 != 0:
            indent = max(4, 4 * (indent // 4))
        lines.append((" " * indent) + stripped)
    repaired = "\n".join(lines).strip("\n")
    if _body_parses(repaired):
        return repaired

    # Fallback: reconstruct block indentation from colon-led structure.
    # This is intentionally conservative and only used when regular repair fails.
    rebuilt: list[str] = []
    depth = 1
    saw_header = False
    prev_nonempty: str | None = None
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            rebuilt.append("")
            continue

        if _DEDENT_LEAD.match(stripped) and depth > 1:
            depth -= 1

        prev_is_header = bool(prev_nonempty and _BLOCK_HEADER.match(prev_nonempty))
        if _TOPLEVEL_LIKELY.match(stripped) and saw_header and not prev_is_header:
            indent_level = 1
        else:
            indent_level = max(1, depth)

        rebuilt.append((" " * (4 * indent_level)) + stripped)

        if _BLOCK_HEADER.match(stripped):
            depth = max(depth, indent_level) + 1
            saw_header = True
        prev_nonempty = stripped

    rebuilt_text = "\n".join(rebuilt).strip("\n")
    if _body_parses(rebuilt_text):
        return rebuilt_text
    return repaired


def _sanitize_body(body: str) -> str:
    """Trim common degenerate tails in model outputs.

    This strips repeated filler/comment tails and collapses duplicate-line loops
    that frequently appear in low-quality teacher generations.
    """
    out: list[str] = []
    prev_norm = ""
    repeat_count = 0

    def _is_repeat_junk(norm_line: str) -> bool:
        low = norm_line.lower()
        if low.startswith("#"):
            return True
        if low in {"pass", "continue", "break"}:
            return True
        if _TRIVIAL_RETURN.match(low):
            return True
        return False

    for raw in body.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            if out and out[-1] != "":
                out.append("")
            prev_norm = ""
            repeat_count = 0
            continue

        low = stripped.lower()
        if stripped.startswith(('"""', "'''")):
            break
        if any(low.startswith(prefix) for prefix in _NON_CODE_PREFIXES):
            break
        if any(low.startswith(prefix) for prefix in _PROSE_PREFIXES):
            break
        if re.match(r"^step\s+\d+[:\-]", low):
            break
        if re.match(r"^\d+\.\s+[A-Za-z]", stripped):
            break

        if "#" in line:
            code_part, comment = line.split("#", 1)
            if _BAD_INLINE_COMMENT.search(comment):
                line = code_part.rstrip()
                stripped = line.strip()
                if not stripped:
                    continue

        norm = stripped
        if norm == prev_norm:
            # Only suppress obvious junk loops. Real repeated ops (e.g., two
            # consecutive mish calls) are valid for kernel-style tasks.
            if _is_repeat_junk(norm):
                repeat_count += 1
                if repeat_count >= 2:
                    break
                continue
            out.append(line)
            prev_norm = norm
            repeat_count = 0
            continue

        prev_norm = norm
        repeat_count = 0
        out.append(line)

    while out and not out[-1].strip():
        out.pop()

    return "\n".join(out)


def normalize_completion(text: str, entry_point: str = "") -> str:
    """Normalize model text into a function-body completion."""
    code = extract_code(text)
    extracted = _extract_function_body(code, entry_point)
    if extracted:
        extracted = _sanitize_body(extracted)
    if extracted and _body_parses(extracted):
        return extracted.strip("\n")

    lines = code.splitlines()
    kept = []
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.startswith("###"):
            continue
        if stripped.upper().startswith("THE FINAL ANSWER"):
            continue
        if stripped in {"PASS", "FAIL"}:
            continue
        if re.match(r"^\s*(def|class)\s+\w+", line):
            if not started:
                # Ignore full definitions in fallback mode; extraction handles them.
                continue
            break
        if re.match(r"^\s*if\s+__name__\s*==", line):
            break
        if stripped.startswith("print("):
            break
        if not started:
            if not stripped:
                continue
            low = stripped.lower()
            if any(low.startswith(prefix) for prefix in _NON_CODE_PREFIXES):
                continue
            if not _looks_like_python_line(stripped):
                continue
            started = True
        else:
            low = stripped.lower()
            if any(low.startswith(prefix) for prefix in _NON_CODE_PREFIXES):
                break
            if any(low.startswith(prefix) for prefix in _PROSE_PREFIXES):
                break
            # Stop at clearly non-code numbered prose blocks.
            if re.match(r"^\d+\.\s+[A-Za-z]", stripped):
                break
        kept.append(line.rstrip())

    while kept and not kept[-1].strip():
        kept.pop()

    if not kept:
        return "    pass"

    normalized = []
    for line in kept:
        if not line.strip():
            normalized.append("")
        elif line.startswith((" ", "\t")):
            normalized.append(line)
        else:
            normalized.append("    " + line)
    body = "\n".join(normalized).strip("\n")
    body = _sanitize_body(body)
    if not body.strip():
        return "    pass"
    if not _body_parses(body):
        repaired = _coerce_indentation(body)
        if _body_parses(repaired):
            return repaired
    return body
