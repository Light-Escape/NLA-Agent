"""
Agent 输出格式规范化。

前端使用 remark-math + KaTeX 渲染，只稳定支持 Markdown 数学分隔符：
行内 `$...$`，块级 `$$...$$`。这里在模型返回前修复常见分隔符错误，
避免未闭合的 LaTeX 分隔符进入前端导致整段公式无法显示。
"""

from __future__ import annotations

import re

_PROTECTED_BLOCK_RE = re.compile(
    r"(```[\s\S]*?```|<nla-workspace>[\s\S]*?</nla-workspace>)",
    re.IGNORECASE,
)
_DISPLAY_BRACKET_RE = re.compile(r"\\\[([\s\S]*?)\\\]")
_INLINE_PAREN_RE = re.compile(r"\\\((.*?)\\\)")
_DISPLAY_MATH_HINT_RE = re.compile(
    r"(\\begin\{|\\end\{|\\\\|\\frac|\\sqrt|\\left|\\right|\\sum|\\prod|\\int|"
    r"\\quad|\\cdot|\\times|\\[A-Za-z]+|[&_^=])"
)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _find_math_delimiters(text: str) -> list[tuple[int, int]]:
    delimiters: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] != "$" or _is_escaped(text, index):
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            delimiters.append((index, 2))
            index += 2
            continue
        delimiters.append((index, 1))
        index += 1
    return delimiters


def _find_next_unescaped_display_delimiter(text: str, start: int) -> int:
    index = start
    while index < len(text) - 1:
        if text[index] == "$" and text[index + 1] == "$" and not _is_escaped(text, index):
            return index
        index += 1
    return -1


def _has_unescaped_single_dollar(text: str, start: int, end: int) -> bool:
    index = start
    while index < end:
        if text[index] == "$" and not _is_escaped(text, index):
            if index + 1 < len(text) and text[index + 1] == "$":
                index += 2
                continue
            return True
        index += 1
    return False


def _looks_like_display_math(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return False
    return "\n" in stripped or bool(_DISPLAY_MATH_HINT_RE.search(stripped)) or len(stripped) >= 24


def _is_line_prefix_whitespace(text: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and text[cursor] != "\n":
        if text[cursor] not in (" ", "\t"):
            return False
        cursor -= 1
    return True


def _repair_paragraph_start_display_openers(text: str) -> str:
    """修复模型常见的段首块级公式少输出一个 `$` 的情况。"""
    repaired: list[str] = []
    index = 0

    while index < len(text):
        at_paragraph_start = _is_line_prefix_whitespace(text, index)
        if not at_paragraph_start or text[index] != "$" or _is_escaped(text, index):
            repaired.append(text[index])
            index += 1
            continue

        if index + 1 < len(text) and text[index + 1] == "$":
            repaired.append("$$")
            index += 2
            continue

        close_index = _find_next_unescaped_display_delimiter(text, index + 1)
        if (
            close_index != -1
            and not _has_unescaped_single_dollar(text, index + 1, close_index)
            and _looks_like_display_math(text[index + 1 : close_index])
        ):
            repaired.append("$$")
        else:
            repaired.append("$")
        index += 1

    return "".join(repaired)


def _escape_unpaired_math_delimiters(text: str) -> str:
    delimiters = _find_math_delimiters(text)
    paired: set[int] = set()
    display_ranges: list[tuple[int, int]] = []
    open_display_token: int | None = None

    for token_index, (start, length) in enumerate(delimiters):
        if length != 2:
            continue
        if open_display_token is None:
            open_display_token = token_index
            continue
        paired.add(open_display_token)
        paired.add(token_index)
        display_ranges.append((delimiters[open_display_token][0], start + length))
        open_display_token = None

    open_inline_token: int | None = None
    for token_index, (start, length) in enumerate(delimiters):
        inside_display = any(range_start < start < range_end for range_start, range_end in display_ranges)
        if length != 1 or inside_display:
            continue
        if open_inline_token is None:
            open_inline_token = token_index
            continue
        paired.add(open_inline_token)
        paired.add(token_index)
        open_inline_token = None

    escape_positions: set[int] = set()
    for token_index, (start, length) in enumerate(delimiters):
        if token_index in paired:
            continue
        escape_positions.update(range(start, start + length))

    if not escape_positions:
        return text

    normalized: list[str] = []
    for index, char in enumerate(text):
        if index in escape_positions:
            normalized.append("\\")
        normalized.append(char)
    return "".join(normalized)


def _transform_outside_inline_code(text: str) -> str:
    normalized: list[str] = []
    plain_text: list[str] = []
    index = 0

    while index < len(text):
        if text[index] != "`":
            plain_text.append(text[index])
            index += 1
            continue

        tick_start = index
        while index + 1 < len(text) and text[index + 1] == "`":
            index += 1
        tick_count = index - tick_start + 1
        tick_fence = "`" * tick_count
        closing_index = text.find(tick_fence, index + 1)

        if closing_index == -1:
            plain_text.append(tick_fence)
            index += 1
            continue

        normalized.append(_escape_unpaired_math_delimiters("".join(plain_text)))
        plain_text = []
        normalized.append(text[tick_start : closing_index + tick_count])
        index = closing_index + tick_count

    normalized.append(_escape_unpaired_math_delimiters("".join(plain_text)))
    return "".join(normalized)


def _normalize_unprotected_segment(segment: str) -> str:
    segment = _DISPLAY_BRACKET_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", segment)
    segment = _INLINE_PAREN_RE.sub(lambda m: f"${m.group(1).strip()}$", segment)
    segment = _repair_paragraph_start_display_openers(segment)
    return _transform_outside_inline_code(segment)


def sanitize_latex_markdown(text: str) -> str:
    """把模型输出修正为前端可渲染的 LaTeX Markdown。"""
    if not isinstance(text, str) or not text:
        return text

    chunks = _PROTECTED_BLOCK_RE.split(text)
    for index, chunk in enumerate(chunks):
        if not chunk or _PROTECTED_BLOCK_RE.fullmatch(chunk):
            continue
        chunks[index] = _normalize_unprotected_segment(chunk)
    return "".join(chunks)
