"""
数值线性代数 Agent 入口：仅负责组装模型、回调与工具注册。
"""

from __future__ import annotations

import copy as _copy
import json as _std_json
import os
import re
import ast
from contextvars import ContextVar
from functools import wraps
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from scipy.sparse import issparse

try:
    # As a package module (e.g. `python -m NLA_Master.agent`)
    from .executors import run_python_snippet, run_python_snippet_unchecked
    from .parsers import load_matrix_csc_content, load_matrix_csc_file, load_matrix_mtx_gz
    from .policy import (
        analyze_matrix_properties,
        build_precondition_checklist,
        choose_nla_algorithm,
        infer_solution_preference,
        plan_coach_next_step,
        route_user_task,
    )
    from .retrieval import search_numpy_scipy_docs
    from .linalg_backend import (
        call_blas,
        call_lapack,
        gemm_blas,
        get_linalg_backend_info,
        least_squares_lapack,
        solve_linear_lapack,
    )
    from .sparse_backend import (
        cg_sparse,
        eigs_sparse,
        eigsh_sparse,
        get_sparse_backend_info,
        gmres_sparse,
        spsolve_sparse,
    )
    from .memory import NLAMemoryStore, load_memory_config, extract_memory_from_dialogue as _extract_memory_from_dialogue
    from .memory.agent_integration import MemoryAgentBridge
    from .output_format import sanitize_latex_markdown
    from .workspace import WorkspaceStore
except ImportError:
    # As a direct script (e.g. `python agent.py`)
    from executors import run_python_snippet, run_python_snippet_unchecked
    from parsers import load_matrix_csc_content, load_matrix_csc_file, load_matrix_mtx_gz
    from policy import (
        analyze_matrix_properties,
        build_precondition_checklist,
        choose_nla_algorithm,
        infer_solution_preference,
        plan_coach_next_step,
        route_user_task,
    )
    from retrieval import search_numpy_scipy_docs
    from linalg_backend import (
        call_blas,
        call_lapack,
        gemm_blas,
        get_linalg_backend_info,
        least_squares_lapack,
        solve_linear_lapack,
    )
    from sparse_backend import (
        cg_sparse,
        eigs_sparse,
        eigsh_sparse,
        get_sparse_backend_info,
        gmres_sparse,
        spsolve_sparse,
    )
    from memory import NLAMemoryStore, load_memory_config, extract_memory_from_dialogue as _extract_memory_from_dialogue
    from memory.agent_integration import MemoryAgentBridge
    from output_format import sanitize_latex_markdown
    from workspace import WorkspaceStore

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)
load_dotenv()

_llm = LiteLlm(model="deepseek/deepseek-chat")
_active_workspace_key: ContextVar[Optional[str]] = ContextVar("nla_active_workspace_key", default=None)
_last_workspace_key = "default"
_FRONTEND_SESSION_RE = re.compile(r'<frontend-session\s+id="([^"]+)"\s*/?>')


def _normalize_workspace_key(key: str | None) -> str:
    text = (key or "").strip()
    return text if text else "default"


def _set_active_workspace_key(key: str | None) -> str:
    global _last_workspace_key
    normalized = _normalize_workspace_key(key)
    _active_workspace_key.set(normalized)
    _last_workspace_key = normalized
    return normalized


def _current_workspace_key() -> str:
    return _active_workspace_key.get() or _last_workspace_key or "default"


class SessionWorkspaceStore:
    """按前端聊天会话分流 Workspace，避免不同聊天共享同一批变量。"""

    def __init__(self) -> None:
        self._stores: dict[str, WorkspaceStore] = {}

    def _store(self) -> WorkspaceStore:
        key = _current_workspace_key()
        if key not in self._stores:
            self._stores[key] = WorkspaceStore()
        return self._stores[key]

    def __getattr__(self, name: str):
        return getattr(self._store(), name)


_workspace_store = SessionWorkspaceStore()

_memory_store = None
_memory_bridge = None
_memory_init_error = ""
try:
    _memory_config = load_memory_config()
    if os.getenv("NLA_MEMORY_READONLY", "").strip():
        _memory_config.readonly_mode = os.getenv("NLA_MEMORY_READONLY", "0") == "1"
    if os.getenv("NLA_MEMORY_AUTO_WRITE", "").strip():
        _memory_config.auto_write_mode = os.getenv("NLA_MEMORY_AUTO_WRITE", "0") == "1"
    _memory_store = NLAMemoryStore(config=_memory_config)
    _memory_bridge = MemoryAgentBridge(store=_memory_store)
except Exception as exc:
    _memory_store = None
    _memory_bridge = None
    _memory_init_error = f"{type(exc).__name__}: {exc}"

_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\\S+")
_SUPPORTED_MIME_PREFIXES = ("text/", "image/jpeg", "image/png", "image/gif", "image/webp")
_VALID_JSON_ESCAPE_CHARS = frozenset('"\\/ bfnrtu')


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
    return raw


def _drop_trailing_commas(raw: str) -> str:
    # 把 `,}` / `,]` 修正成 `}` / `]`，仅在字符串外生效
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if in_str and ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if not in_str and ch == ",":
            j = i + 1
            while j < n and raw[j].isspace():
                j += 1
            if j < n and raw[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _close_unbalanced_json(raw: str) -> str:
    # 若末尾截断导致括号未闭合，尝试自动补全
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        raw += '"'
    if stack:
        raw += "".join(reversed(stack))
    return raw


def _extract_first_json_payload(raw: str) -> str:
    """从混合文本中提取首个 JSON 对象/数组片段。"""
    if not raw or not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return raw

    stack: list[str] = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return raw
            stack.pop()
            if not stack:
                return text[start : i + 1].strip()
    return text[start:].strip()


def _insert_missing_commas(raw: str) -> str:
    """
    尝试修复常见漏逗号场景：
    - {"a":1 "b":2}
    - {"a":{} "b":[]}
    - {"a":"x" "b":"y"}
    """
    if not raw or not isinstance(raw, str):
        return raw
    out: list[str] = []
    in_str = False
    esc = False
    n = len(raw)
    i = 0

    def _next_non_ws(pos: int) -> int:
        while pos < n and raw[pos].isspace():
            pos += 1
        return pos

    def _looks_like_key(pos: int) -> bool:
        pos = _next_non_ws(pos)
        if pos >= n:
            return False
        if raw[pos] == '"':
            pos += 1
            local_esc = False
            while pos < n:
                ch2 = raw[pos]
                if local_esc:
                    local_esc = False
                elif ch2 == "\\":
                    local_esc = True
                elif ch2 == '"':
                    return _next_non_ws(pos + 1) < n and raw[_next_non_ws(pos + 1)] == ":"
                pos += 1
            return False
        if raw[pos].isalpha() or raw[pos] == "_":
            pos += 1
            while pos < n and (raw[pos].isalnum() or raw[pos] in "_-"):
                pos += 1
            pos = _next_non_ws(pos)
            return pos < n and raw[pos] == ":"
        return False

    while i < n:
        ch = raw[i]
        out.append(ch)
        token_end = False

        if esc:
            esc = False
            i += 1
            continue
        if in_str and ch == "\\":
            esc = True
            i += 1
            continue
        if ch == '"':
            was_in_str = in_str
            in_str = not in_str
            if was_in_str and not in_str and not (
                _next_non_ws(i + 1) < n and raw[_next_non_ws(i + 1)] == ":"
            ):
                token_end = True
        elif in_str:
            i += 1
            continue
        elif ch in ("}", "]") or ch.isdigit() or ch in "eE.-+tfn":
            token_end = True

        if token_end and _looks_like_key(i + 1):
            prev_non_ws = ""
            p = len(out) - 1
            while p >= 0:
                if not out[p].isspace():
                    prev_non_ws = out[p]
                    break
                p -= 1
            if prev_non_ws not in "{[,:":
                out.append(",")

        i += 1

    return "".join(out)


def _insert_missing_colons(raw: str) -> str:
    """
    修复工具参数中常见的键值漏冒号：
    {"content" "1 2"} -> {"content": "1 2"}
    """
    if not raw or not isinstance(raw, str):
        return raw

    out: list[str] = []
    in_str = False
    esc = False
    stack: list[dict[str, object]] = []
    i = 0
    n = len(raw)

    def _next_non_ws(pos: int) -> int:
        while pos < n and raw[pos].isspace():
            pos += 1
        return pos

    while i < n:
        ch = raw[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue

        if in_str:
            out.append(ch)
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == "{":
            stack.append({"kind": "object", "expect_key": True})
            out.append(ch)
            i += 1
            continue
        if ch == "[":
            stack.append({"kind": "array", "expect_key": False})
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            if stack:
                stack.pop()
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            if stack and stack[-1].get("kind") == "object":
                stack[-1]["expect_key"] = True
            out.append(ch)
            i += 1
            continue
        if ch == ":":
            if stack and stack[-1].get("kind") == "object":
                stack[-1]["expect_key"] = False
            out.append(ch)
            i += 1
            continue

        if ch == '"':
            out.append(ch)
            i += 1
            local_esc = False
            while i < n:
                ch2 = raw[i]
                out.append(ch2)
                if local_esc:
                    local_esc = False
                elif ch2 == "\\":
                    local_esc = True
                elif ch2 == '"':
                    break
                i += 1

            if stack and stack[-1].get("kind") == "object" and stack[-1].get("expect_key"):
                j = _next_non_ws(i + 1)
                if j < n and raw[j] not in ":,}]":
                    out.append(":")
                stack[-1]["expect_key"] = False
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _try_python_literal_parse(raw: str):
    # 兜底：兼容 Python 字典风格参数（单引号/True/None）
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    return None


def _normalize_python_json_literals(raw: str) -> str:
    """把字符串外的 True/False/None 归一为 JSON 字面量。"""
    if not raw or not isinstance(raw, str):
        return raw
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(raw)
    replacements = {"True": "true", "False": "false", "None": "null"}
    while i < n:
        ch = raw[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if in_str and ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if not in_str:
            matched = False
            for src, dst in replacements.items():
                end = i + len(src)
                if raw.startswith(src, i):
                    prev_ok = i == 0 or not (raw[i - 1].isalnum() or raw[i - 1] == "_")
                    next_ok = end >= n or not (raw[end].isalnum() or raw[end] == "_")
                    if prev_ok and next_ok:
                        out.append(dst)
                        i = end
                        matched = True
                        break
            if matched:
                continue
        out.append(ch)
        i += 1
    return "".join(out)


_TOOL_STRING_FIELD_FOLLOWERS: dict[str, set[str]] = {
    "code": {"array_refs", "scalar_refs", "timeout_s"},
}


def _is_escaped_quote(raw: str, pos: int) -> bool:
    backslashes = 0
    i = pos - 1
    while i >= 0 and raw[i] == "\\":
        backslashes += 1
        i -= 1
    return bool(backslashes % 2)


def _next_non_ws(raw: str, pos: int) -> int:
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    return pos


def _read_jsonish_quoted_key(raw: str, pos: int) -> tuple[str | None, int]:
    if pos >= len(raw) or raw[pos] != '"':
        return None, pos
    out: list[str] = []
    i = pos + 1
    esc = False
    while i < len(raw):
        ch = raw[i]
        if esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            return "".join(out), i + 1
        else:
            out.append(ch)
        i += 1
    return None, pos


def _find_string_field_value_starts(raw: str, field_name: str) -> list[int]:
    starts: list[int] = []
    needle = f'"{field_name}"'
    pos = 0
    while True:
        key_pos = raw.find(needle, pos)
        if key_pos < 0:
            break
        colon_pos = _next_non_ws(raw, key_pos + len(needle))
        if colon_pos < len(raw) and raw[colon_pos] == ":":
            quote_pos = _next_non_ws(raw, colon_pos + 1)
            if quote_pos < len(raw) and raw[quote_pos] == '"':
                starts.append(quote_pos + 1)
        pos = key_pos + len(needle)
    return starts


def _string_field_end_candidates(raw: str, value_start: int, field_name: str) -> list[int]:
    followers = _TOOL_STRING_FIELD_FOLLOWERS.get(field_name, set())
    candidates: list[int] = []
    i = value_start
    while i < len(raw):
        if raw[i] != '"' or _is_escaped_quote(raw, i):
            i += 1
            continue

        after_quote = _next_non_ws(raw, i + 1)
        if after_quote >= len(raw):
            candidates.append(i)
        elif raw[after_quote] == ",":
            key_pos = _next_non_ws(raw, after_quote + 1)
            key, key_end = _read_jsonish_quoted_key(raw, key_pos)
            colon_pos = _next_non_ws(raw, key_end)
            if key in followers and colon_pos < len(raw) and raw[colon_pos] == ":":
                candidates.append(i)
        elif raw[after_quote] == "}":
            rest = raw[after_quote:]
            if all(ch.isspace() or ch in "}]" for ch in rest):
                candidates.append(i)
        i += 1
    return candidates


def _repair_tool_string_fields(raw: str) -> list[str]:
    """
    针对工具参数中的长代码字符串做结构化修复。

    LLM 经常把 Python 片段写成 `"code": "RESULT = {"x": 1}"`，
    内层双引号和裸换行会让外层 function.arguments 不是合法 JSON。
    这里只在识别到后续参数键或对象结束时截取完整 code 字段，再用
    json.dumps 重新编码该字段；不猜测任意 JSON 结构。
    """
    if not raw or not isinstance(raw, str):
        return []

    repaired: list[str] = []
    for field_name in _TOOL_STRING_FIELD_FOLLOWERS:
        for value_start in _find_string_field_value_starts(raw, field_name):
            for value_end in reversed(_string_field_end_candidates(raw, value_start, field_name)):
                value = raw[value_start:value_end]
                candidate = raw[: value_start - 1] + _std_json.dumps(value, ensure_ascii=False) + raw[value_end + 1 :]
                if candidate != raw:
                    repaired.append(candidate)
    return repaired


def _quote_unquoted_object_keys(raw: str) -> str:
    """修复常见的 `{foo: 1}` / `{foo_bar: 2}` 裸键写法。"""
    if not raw or not isinstance(raw, str):
        return raw
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if in_str and ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if not in_str and ch in "{,":
            out.append(ch)
            i += 1
            while i < n and raw[i].isspace():
                out.append(raw[i])
                i += 1
            key_start = i
            if i < n and (raw[i].isalpha() or raw[i] == "_"):
                i += 1
                while i < n and (raw[i].isalnum() or raw[i] in "_-"):
                    i += 1
                key = raw[key_start:i]
                j = i
                while j < n and raw[j].isspace():
                    j += 1
                if j < n and raw[j] == ":":
                    out.append(f'"{key}"')
                    continue
                out.append(key)
                continue
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_tool_args_with_repair(raw: str, **kwargs):
    # 1) 标准 JSON
    return _std_json.loads(raw, **kwargs)


def _repair_tool_call_json(raw: str) -> str:
    """修复工具参数 JSON 中的裸换行与非法反斜杠转义。"""
    if not raw or not isinstance(raw, str):
        return raw
    try:
        _std_json.loads(raw)
        return raw
    except _std_json.JSONDecodeError:
        pass

    out: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if esc:
            if ch in _VALID_JSON_ESCAPE_CHARS:
                out.append(ch)
            else:
                out.append("\\")
                out.append(ch)
            esc = False
        elif in_str and ch == "\\":
            out.append(ch)
            esc = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        elif in_str and ch == "\n":
            out.append("\\n")
        elif in_str and ch == "\r":
            out.append("\\r")
        elif in_str and ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)

    repaired = "".join(out)
    try:
        _std_json.loads(repaired)
        return repaired
    except _std_json.JSONDecodeError:
        return raw


class _JsonWithRepair:
    @staticmethod
    def loads(s, **kwargs):
        try:
            return _parse_tool_args_with_repair(s, **kwargs)
        except _std_json.JSONDecodeError as orig:
            if isinstance(s, str):
                candidates: list[str] = []
                base = _strip_code_fence(s)
                escaped = _repair_tool_call_json(base)
                string_field_repairs = _repair_tool_string_fields(base)
                normalized_literals = _normalize_python_json_literals(escaped)
                colon_fixed = _insert_missing_colons(normalized_literals)
                quoted_keys = _quote_unquoted_object_keys(colon_fixed)
                comma_fixed = _insert_missing_commas(quoted_keys)
                trailing_fixed = _drop_trailing_commas(comma_fixed)
                candidates.append(base)
                candidates.append(_extract_first_json_payload(base))
                candidates.extend(string_field_repairs)
                for repaired_string_fields in string_field_repairs:
                    candidates.append(_repair_tool_call_json(repaired_string_fields))
                    candidates.append(_drop_trailing_commas(_insert_missing_commas(repaired_string_fields)))
                candidates.append(escaped)
                candidates.append(normalized_literals)
                candidates.append(_insert_missing_colons(escaped))
                candidates.append(colon_fixed)
                candidates.append(_quote_unquoted_object_keys(escaped))
                candidates.append(quoted_keys)
                candidates.append(_insert_missing_commas(_insert_missing_colons(escaped)))
                candidates.append(comma_fixed)
                candidates.append(_drop_trailing_commas(_insert_missing_commas(_insert_missing_colons(escaped))))
                candidates.append(trailing_fixed)
                candidates.append(_close_unbalanced_json(trailing_fixed))
                seen = set()
                for repaired in candidates:
                    if not repaired or repaired in seen:
                        continue
                    seen.add(repaired)
                    try:
                        return _parse_tool_args_with_repair(repaired, **kwargs)
                    except _std_json.JSONDecodeError:
                        pass

                # 最后一层容错：把参数当作 Python literal 解析
                py_obj = _try_python_literal_parse(base)
                if py_obj is not None:
                    return py_obj
            raise orig

    def __getattr__(self, name: str):
        return getattr(_std_json, name)


def _obj_get(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _obj_set(obj, key: str, value) -> bool:
    if isinstance(obj, dict):
        obj[key] = value
        return True
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        return False


def _repair_tool_call_arguments_in_place(message) -> bool:
    """
    修复 OpenAI/LiteLLM 风格 tool_calls 中的 function.arguments。

    ADK 在把 LiteLLM 响应转成 Gemini Content 时会再次解析 arguments；
    这里先把模型生成的宽松 JSON 归一成严格 JSON 字符串，让工具调用继续执行。
    """
    tool_calls = _obj_get(message, "tool_calls")
    if not isinstance(tool_calls, list):
        return False

    repaired_any = False
    for tool_call in tool_calls:
        function = _obj_get(tool_call, "function")
        if function is None:
            continue

        arguments = _obj_get(function, "arguments")
        if isinstance(arguments, str):
            try:
                parsed = _JsonWithRepair.loads(arguments)
            except _std_json.JSONDecodeError:
                continue
        elif isinstance(arguments, (dict, list)):
            parsed = arguments
        else:
            continue

        normalized = _std_json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if normalized != arguments:
            repaired_any = _obj_set(function, "arguments", normalized) or repaired_any

    return repaired_any


try:
    import google.adk.models.lite_llm as _adk_lite_llm_mod

    _adk_lite_llm_mod.json = _JsonWithRepair()

    _orig_msg_to_resp = _adk_lite_llm_mod._message_to_generate_content_response

    def _safe_message_to_generate_content_response(
        message,
        *,
        is_partial: bool = False,
        model_version: str = None,
        thought_parts=None,
    ):
        """
        当工具参数 JSON 损坏时，先尝试修复 tool_calls[*].function.arguments，
        修复成功后重新走 ADK 原始转换，避免直接跳过工具调用。
        """
        try:
            return _orig_msg_to_resp(
                message,
                is_partial=is_partial,
                model_version=model_version,
                thought_parts=thought_parts,
            )
        except _std_json.JSONDecodeError as exc:
            try:
                repaired_message = _copy.deepcopy(message)
                if _repair_tool_call_arguments_in_place(repaired_message):
                    return _orig_msg_to_resp(
                        repaired_message,
                        is_partial=is_partial,
                        model_version=model_version,
                        thought_parts=thought_parts,
                    )
            except _std_json.JSONDecodeError:
                pass
            except Exception:
                pass

            parts = []
            if thought_parts:
                parts.extend(thought_parts)
            try:
                msg_text, _ = _adk_lite_llm_mod._split_message_content_and_tool_calls(message)
                if isinstance(msg_text, str) and msg_text:
                    parts.append(types.Part.from_text(text=msg_text))
            except Exception:
                raw_text = getattr(message, "content", None)
                if isinstance(raw_text, str) and raw_text:
                    parts.append(types.Part.from_text(text=raw_text))

            parts.append(
                types.Part.from_text(
                    text=(
                        "[检测到工具参数 JSON 格式异常，已尝试自动修复但仍失败；"
                        f"请重新生成严格 JSON 工具调用参数。错误：{exc}]"
                    )
                )
            )
            return LlmResponse(
                content=types.Content(role="model", parts=parts),
                partial=is_partial,
                model_version=model_version,
            )

    _adk_lite_llm_mod._message_to_generate_content_response = (
        _safe_message_to_generate_content_response
    )
except Exception:
    pass


def add_memory(item: dict, require_confirmation: bool = True) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error}
    return _memory_store.add_memory(item=item, require_confirmation=require_confirmation)


def search_memory(query: str, top_k: int = 5, filters: Optional[dict] = None) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error, "items": []}
    return _memory_store.search_memory(query=query, top_k=top_k, filters=filters)


def update_memory(memory_id: str, item: dict) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error}
    return _memory_store.update_memory(memory_id=memory_id, item=item)


def delete_memory(memory_id: str) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error}
    return _memory_store.delete_memory(memory_id=memory_id)


def summarize_memory(query_results: dict) -> str:
    if _memory_store is None:
        return f"memory store unavailable: {_memory_init_error}"
    return _memory_store.summarize_memory(query_results=query_results)


def list_pending_memories(limit: int = 20) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error, "items": []}
    return _memory_store.list_pending_memories(limit=limit)


def approve_pending_memory(memory_id: str) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error}
    return _memory_store.approve_pending_memory(memory_id=memory_id)


def get_memory(memory_id: str, include_pending: bool = True) -> dict:
    if _memory_store is None:
        return {"status": "error", "message": "memory store unavailable", "detail": _memory_init_error}
    return _memory_store.get_memory(memory_id=memory_id, include_pending=include_pending)


def extract_memory_from_dialogue(user_question: str, assistant_answer: str) -> dict:
    return _extract_memory_from_dialogue(user_question=user_question, assistant_answer=assistant_answer)


def workspace_set(name: str, value: object, source: str = "Agent", notes: str = "") -> dict:
    """设置轻量工作区变量；大矩阵/长向量必须通过加载器或计算工具注册。"""
    return _workspace_store.set_user_var(name=name, value=value, source=source, notes=notes)


def workspace_get(name: str) -> dict:
    """读取工作区变量。"""
    return _workspace_store.get_var(name=name)


def workspace_list(detail: bool = False) -> dict:
    """列出工作区变量；detail=True 时返回类型与shape。"""
    return _workspace_store.list_vars(detail=bool(detail))


def workspace_clear(var_name: Optional[str] = None) -> dict:
    """清空工作区或清理指定变量。"""
    return _workspace_store.clear(name=var_name)


def workspace_alias(name: str, ref: str, notes: str = "") -> dict:
    """为已有 Workspace 对象创建别名，不展开、不复制底层数据。"""
    return _workspace_store.alias(name=name, ref=ref, notes=notes)


def workspace_bind_role(role: str, ref: str) -> dict:
    """为已有 Workspace 对象绑定数值角色，如 system_matrix/rhs/preconditioner。"""
    return _workspace_store.bind_role(role=role, ref=ref)


def workspace_summary(name: str) -> dict:
    """返回 Workspace 对象的句柄、版本与短摘要，不展开完整数据。"""
    return _workspace_store.summary(name=name)


def workspace_stats(name: str) -> dict:
    """返回 Workspace 对象的受控数值统计信息。"""
    return _workspace_store.stats(name=name)


def workspace_structure(name: str, tol: float = 1e-10) -> dict:
    """返回 Workspace 矩阵/向量的结构摘要，不展开完整数据。"""
    return _workspace_store.structure(name=name, tol=tol)


def workspace_slice(name: str, rows: Optional[list[int]] = None, cols: Optional[list[int]] = None) -> dict:
    """受控读取 Workspace 对象的小切片，单次返回元素数有硬上限。"""
    return _workspace_store.read_slice(name=name, rows=rows, cols=cols)


def workspace_audit(limit: int = 50) -> dict:
    """返回最近 Workspace 访问与计算记录。"""
    return _workspace_store.audit(limit=limit)


def who() -> dict:
    """MATLAB 风格 who：仅返回变量名。"""
    return workspace_list(detail=False)


def whos() -> dict:
    """MATLAB 风格 whos：返回变量详细信息。"""
    return workspace_list(detail=True)


def clear(var_name: Optional[str] = None) -> dict:
    """MATLAB 风格 clear：clear / clear var_name。"""
    return workspace_clear(var_name=var_name)


_LARGE_RESULT_KEYS = {
    "x",
    "u",
    "v",
    "vh",
    "q",
    "r",
    "lu",
    "eigenvectors",
    "singular_vectors",
    "residual_history",
}


def _result_value_needs_handle(value: object) -> bool:
    try:
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return True
        return int(arr.size) > 64
    except Exception:
        return False


def _result_ref_name(field_name: str) -> str:
    if field_name == "x":
        return _workspace_store.allocate_ref("x")
    if field_name == "residual_history":
        return _workspace_store.allocate_ref("residual_history")
    return _workspace_store.allocate_ref(f"{field_name}_result")


def _slim_large_result_fields(result: object, source: str) -> object:
    if not isinstance(result, dict):
        return result
    slim = dict(result)
    saved_refs: dict[str, str] = {}
    for key in list(result.keys()):
        if key not in _LARGE_RESULT_KEYS:
            continue
        value = result.get(key)
        if not _result_value_needs_handle(value):
            continue
        ref_name = _result_ref_name(key)
        save_ret = _workspace_store.set_var(
            name=ref_name,
            value=value,
            source=source,
            notes=f"大结果字段 {key} 自动保存为 Workspace 对象",
            role=key,
            origin=source,
            created_by_tool=source,
        )
        variable = save_ret.get("variable", {})
        slim.pop(key, None)
        slim[f"{key}_ref"] = ref_name
        slim[f"{key}_handle"] = {
            "object_handle": True,
            "ref": ref_name,
            "name": ref_name,
            "kind": variable.get("kind"),
            "shape": variable.get("shape"),
            "dtype": variable.get("dtype"),
            "storage_type": variable.get("storage_type"),
            "version": variable.get("version"),
            "fingerprint": variable.get("fingerprint"),
            "summary": variable.get("summary"),
            "preview_policy": variable.get("preview_policy"),
        }
        saved_refs[key] = ref_name
    if saved_refs:
        slim["workspace_saved_results"] = saved_refs
        slim["message"] = slim.get("message", "大结果已保存到 Workspace，返回中仅保留句柄。")
    return slim


def _workspace_protocol_error(message: str, *, detail: Optional[dict] = None, preferred_ref: Optional[str] = None) -> dict:
    next_actions = [
        "调用 workspace_summary / workspace_stats / workspace_structure 获取摘要信息",
        "使用 *_ref 或 array_refs 引用 Workspace 对象完成计算",
        "停止当前失败路径，基于可用工具重新规划",
    ]
    if preferred_ref:
        next_actions.insert(0, f"改用 {preferred_ref} 引用已有 Workspace 变量")
    payload = {
        "status": "error",
        "error_type": "workspace_protocol_violation",
        "message": message,
        "fallback_required": True,
        "next_allowed_actions": next_actions,
    }
    if detail:
        payload["detail"] = detail
    try:
        _workspace_store._log("protocol_violation", None, {"reason": message, **(detail or {})})
    except Exception:
        pass
    return payload


def _normalize_tool_error(result: object, source: str) -> object:
    if not isinstance(result, dict) or result.get("status") != "error":
        return result
    normalized = dict(result)
    message = str(normalized.get("message", ""))
    if normalized.get("error_type") == "file_resolution_error" or "文件不存在" in message:
        normalized["error_type"] = "file_resolution_error"
        normalized["fallback_required"] = True
        normalized.setdefault(
            "next_allowed_actions",
            [
                "检查本轮 <frontend-files> 中是否有 uploadUri / file_id",
                "使用 nla-upload://<file_id> 调用 load_matrix_csc_file 或 load_matrix_mtx_gz",
                "若没有 uploadUri，请让用户重新上传文件或启动上传服务",
                "不要猜测浏览器路径、裸文件名、uploads 目录或临时文件路径",
            ],
        )
    else:
        normalized.setdefault("error_type", "tool_error")
        normalized.setdefault(
            "next_allowed_actions",
            [
                "停止当前失败路径",
                "调用 workspace_summary / workspace_stats / workspace_structure 重新确认对象状态",
                "选择可用后端工具，并优先使用 *_ref 或 array_refs",
            ],
        )
    normalized.setdefault("fallback_required", True)
    try:
        _workspace_store._log(
            "fallback_required",
            None,
            {"source": source, "error_type": normalized.get("error_type"), "message": normalized.get("message", "")},
        )
    except Exception:
        pass
    return normalized


def _value_element_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        shape = value.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            total = 1
            for dim in shape:
                try:
                    total *= max(int(dim), 0)
                except Exception:
                    return 0
            return int(total)
        if "A_csc" in value:
            return _value_element_count(value.get("A_csc"))
        if "A_rows" in value:
            return _value_element_count(value.get("A_rows"))
    try:
        return int(np.asarray(value).size)
    except Exception:
        return 0


def _workspace_ref_is_large(ref_name: str) -> bool:
    try:
        variable = _workspace_store.summary(ref_name).get("variable", {})
        shape = variable.get("shape") or []
        total = 1
        for dim in shape:
            total *= max(int(dim), 0)
        return int(total) > 64 and variable.get("kind") in {"matrix", "sparse_matrix", "vector"}
    except Exception:
        return False


def _guard_inline_workspace_matrix(value: object, ref_name: Optional[str], preferred_ref: str, field_name: str) -> Optional[dict]:
    if ref_name:
        return None
    if _value_element_count(value) <= 64:
        return None
    if not _workspace_ref_is_large(preferred_ref):
        return None
    return _workspace_protocol_error(
        f"Workspace 中已有大对象 {preferred_ref}，禁止把 {field_name} 以内联矩阵形式重新传入工具。",
        detail={"field": field_name, "workspace_ref": preferred_ref, "inline_elements": _value_element_count(value)},
        preferred_ref=f"{preferred_ref}_ref='{preferred_ref}'",
    )


def _guard_inline_arrays(arrays: Optional[dict], array_refs: Optional[dict[str, str]]) -> Optional[dict]:
    if not arrays:
        return None
    provided_refs = set(str(v) for v in (array_refs or {}).values())
    for arg_name, value in arrays.items():
        if _value_element_count(value) <= 64:
            continue
        for ref_name in ("A", "B", "C"):
            if ref_name in provided_refs:
                continue
            if _workspace_ref_is_large(ref_name):
                return _workspace_protocol_error(
                    f"Workspace 中已有大对象 {ref_name}，禁止在通用后端调用中内联大数组参数 {arg_name}。",
                    detail={
                        "array_arg": str(arg_name),
                        "workspace_ref": ref_name,
                        "inline_elements": _value_element_count(value),
                    },
                    preferred_ref=f"array_refs={{'{arg_name}': '{ref_name}'}}",
                )
    return None


def _tool_with_auto_ans(tool_fn):
    @wraps(tool_fn)
    def _wrapped(*args, **kwargs):
        result = tool_fn(*args, **kwargs)
        source = getattr(tool_fn, "__name__", "last_tool")
        result = _normalize_tool_error(result, source=source)
        result = _slim_large_result_fields(result, source=source)
        _workspace_store.write_ans(result, source=source)
        return result

    return _wrapped


def _stored_matrix_from_loader_result(result: dict):
    if "_matrix" in result:
        return result["_matrix"]
    if "A_csc" in result:
        return result["A_csc"]
    if "A_rows" in result:
        return result["A_rows"]
    return result


def _slim_matrix_loader_result(result: dict, *, saved_as: str, source: str) -> dict:
    public = {
        "status": result.get("status"),
        "message": result.get("message", "矩阵已载入后端 Workspace。"),
        "workspace_saved": True,
        "saved_as": saved_as,
        "matrix_ref": saved_as,
        "matrix_refs": [saved_as],
        "source": source,
    }
    variable = _workspace_store.get_var(saved_as).get("variable", {})
    handle = _workspace_store.get_var(saved_as).get("value", {})
    public["handle"] = handle
    public["shape"] = variable.get("shape", result.get("shape", []))
    public["format"] = handle.get("format", "matrix") if isinstance(handle, dict) else "matrix"
    public["dtype"] = variable.get("dtype", "unknown")
    public["role"] = variable.get("role", "")
    public["role_bindings"] = {variable.get("role", "matrix") or "matrix": saved_as}
    if isinstance(handle, dict):
        if "nnz" in handle:
            public["nnz"] = handle["nnz"]
        if "density" in handle:
            public["density"] = handle["density"]
    public["next_allowed_actions"] = [
        f"调用 workspace_summary('{saved_as}') 或 workspace_structure('{saved_as}') 获取摘要",
        f"后续计算使用 A_ref='{saved_as}' 或 array_refs 引用 Workspace 对象",
        "若本轮读取了多个矩阵，保留各自 matrix_ref；必要时调用 workspace_bind_role 绑定 system_matrix/rhs/preconditioner 等角色",
        "稀疏问题优先调用 spsolve_sparse / cg_sparse / gmres_sparse / eigsh_sparse / eigs_sparse",
        "不要把完整矩阵写回 A_rows / A_csc，也不要改用裸文件名或 run_python_snippet 重新读取文件",
    ]
    return public


def _resolve_workspace_value(ref: Optional[str]):
    if not ref:
        return None
    return _workspace_store.get_raw_var(ref)


def _resolve_dense_matrix(rows=None, ref: Optional[str] = None):
    if ref:
        return _workspace_store.get_matrix(ref, sparse=False)
    return rows


def _resolve_sparse_matrix(A_csc=None, A_rows=None, ref: Optional[str] = None):
    if ref:
        return _workspace_store.get_matrix(ref, sparse=True), None
    return A_csc, A_rows


_WORKSPACE_SNIPPET_BLOCKED_IMPORTS = {
    "builtins",
    "glob",
    "gzip",
    "io",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "tempfile",
    "urllib",
    "zipfile",
}
_WORKSPACE_SNIPPET_BLOCKED_ATTRS = {
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "load",
    "loadtxt",
    "listdir",
    "mmread",
    "open",
    "save",
    "savetxt",
    "savez",
    "scandir",
    "walk",
}
_WORKSPACE_SNIPPET_ALLOWED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _workspace_snippet_static_check(code: str) -> list[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return [f"代码语法错误: {exc}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.Raise)):
            errors.append(f"禁止语法结构: {type(node).__name__}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".")[0]
                if root in _WORKSPACE_SNIPPET_BLOCKED_IMPORTS or alias.name == "scipy.io":
                    errors.append(f"禁止导入可能访问文件或系统的模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in _WORKSPACE_SNIPPET_BLOCKED_IMPORTS or module == "scipy.io":
                errors.append(f"禁止导入可能访问文件或系统的模块: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "compile", "__import__"}:
                errors.append(f"禁止调用函数: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _WORKSPACE_SNIPPET_BLOCKED_ATTRS:
                errors.append(f"禁止调用可能访问文件的属性: {node.func.attr}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append("禁止访问 dunder 属性")
    return list(dict.fromkeys(errors))


def _jsonable_workspace_value(value: object, *, prefer_sparse: bool = False) -> dict:
    if issparse(value):
        matrix = value.tocsc()
        return {
            "kind": "sparse_csc",
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "indptr": matrix.indptr.tolist(),
            "indices": matrix.indices.tolist(),
            "data": matrix.data.astype(float, copy=False).tolist(),
        }
    if isinstance(value, dict) and (value.get("format") == "csc" or "A_csc" in value):
        payload = value.get("A_csc", value)
        if prefer_sparse:
            return {
                "kind": "sparse_csc",
                "shape": [int(x) for x in payload.get("shape", [])],
                "indptr": list(payload.get("indptr", [])),
                "indices": list(payload.get("indices", [])),
                "data": list(payload.get("data", [])),
            }
    arr = np.asarray(value)
    if arr.ndim > 0:
        return {"kind": "ndarray", "data": arr.tolist(), "dtype": str(arr.dtype)}
    return {"kind": "scalar", "data": arr.item() if hasattr(arr, "item") else value}


def _workspace_injection_prelude(array_refs: Optional[dict[str, str]], scalar_refs: Optional[dict[str, str]]) -> str | dict:
    payload: dict[str, dict] = {"arrays": {}, "scalars": {}}
    for var_name, ref_name in (array_refs or {}).items():
        var = str(var_name).strip()
        if not _WORKSPACE_SNIPPET_ALLOWED_NAME_RE.fullmatch(var):
            return {"status": "error", "message": f"非法 Python 变量名: {var}"}
        value = _workspace_store.get_matrix(str(ref_name), sparse=True)
        payload["arrays"][var] = _jsonable_workspace_value(value, prefer_sparse=True)

    for var_name, ref_name in (scalar_refs or {}).items():
        var = str(var_name).strip()
        if not _WORKSPACE_SNIPPET_ALLOWED_NAME_RE.fullmatch(var):
            return {"status": "error", "message": f"非法 Python 变量名: {var}"}
        value = _workspace_store.get_raw_var(str(ref_name))
        payload["scalars"][var] = _jsonable_workspace_value(value)

    payload_json = _std_json.dumps(payload, ensure_ascii=False)
    return f"""
import json as _nla_json
import numpy as np
from scipy.sparse import csc_matrix

_NLA_WORKSPACE_PAYLOAD = _nla_json.loads({payload_json!r})
for _nla_name, _nla_item in _NLA_WORKSPACE_PAYLOAD.get("arrays", {{}}).items():
    if _nla_item.get("kind") == "sparse_csc":
        globals()[_nla_name] = csc_matrix(
            (_nla_item["data"], _nla_item["indices"], _nla_item["indptr"]),
            shape=tuple(_nla_item["shape"]),
        )
    else:
        globals()[_nla_name] = np.asarray(_nla_item.get("data"), dtype=_nla_item.get("dtype") or None)
for _nla_name, _nla_item in _NLA_WORKSPACE_PAYLOAD.get("scalars", {{}}).items():
    if _nla_item.get("kind") == "ndarray":
        globals()[_nla_name] = np.asarray(_nla_item.get("data"), dtype=_nla_item.get("dtype") or None)
    else:
        globals()[_nla_name] = _nla_item.get("data")
""".strip()


def run_python_workspace_snippet(
    code: str,
    array_refs: Optional[dict[str, str]] = None,
    scalar_refs: Optional[dict[str, str]] = None,
    timeout_s: float = 10.0,
) -> dict:
    """在受控 Python 片段中注入 Workspace 引用；不要用文件名或路径读取矩阵。"""
    src = (code or "").strip()
    if not src:
        return {"status": "error", "message": "code 不能为空"}
    if not array_refs and not scalar_refs:
        return {
            "status": "error",
            "message": "请提供 array_refs 或 scalar_refs；普通无 Workspace 代码请使用 run_python_snippet。",
        }
    errors = _workspace_snippet_static_check(src)
    if errors:
        return {
            "status": "error",
            "error_type": "workspace_python_safety_error",
            "message": "Workspace Python 代码未通过安全检查",
            "detail": {"errors": errors},
            "fallback_required": True,
            "next_allowed_actions": [
                "移除文件 IO、路径扫描或系统模块访问",
                "通过 array_refs / scalar_refs 注入 Workspace 对象",
                "若已有专用后端工具，优先改用 *_ref 或 array_refs 工具",
            ],
        }
    try:
        prelude = _workspace_injection_prelude(array_refs, scalar_refs)
        if isinstance(prelude, dict):
            return prelude
        result = run_python_snippet_unchecked(code=prelude + "\n\n" + src, timeout_s=timeout_s)
        if isinstance(result, dict):
            result["workspace_refs_used"] = {
                "array_refs": dict(array_refs or {}),
                "scalar_refs": dict(scalar_refs or {}),
            }
            result.setdefault("message", "已执行 Workspace 感知 Python 片段。")
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def solve_linear_lapack_tool(
    A_rows: Optional[list[list[float]]] = None,
    b: list[float] | list[list[float]] = None,
    assume: str = "auto",
    A_ref: Optional[str] = None,
    b_ref: Optional[str] = None,
) -> dict:
    """用 LAPACK 求解 Ax=b；优先用 A_ref/b_ref 引用 Workspace 变量。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows, A_ref, "A", "A_rows")
        if guard:
            return guard
        A = _resolve_dense_matrix(A_rows, A_ref)
        rhs = _resolve_workspace_value(b_ref) if b_ref else b
        if A is None:
            return {"status": "error", "message": "请提供 A_ref 或 A_rows"}
        return solve_linear_lapack(A_rows=A, b=rhs, assume=assume)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def least_squares_lapack_tool(
    A_rows: Optional[list[list[float]]] = None,
    b: list[float] | list[list[float]] = None,
    driver: str = "gelsd",
    A_ref: Optional[str] = None,
    b_ref: Optional[str] = None,
) -> dict:
    """用 LAPACK driver 计算最小二乘；优先用 A_ref/b_ref 引用 Workspace 变量。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows, A_ref, "A", "A_rows")
        if guard:
            return guard
        A = _resolve_dense_matrix(A_rows, A_ref)
        rhs = _resolve_workspace_value(b_ref) if b_ref else b
        if A is None:
            return {"status": "error", "message": "请提供 A_ref 或 A_rows"}
        return least_squares_lapack(A_rows=A, b=rhs, driver=driver)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def gemm_blas_tool(
    A_rows: Optional[list[list[float]]] = None,
    B_rows: Optional[list[list[float]]] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    C_rows: Optional[list[list[float]]] = None,
    trans_a: bool = False,
    trans_b: bool = False,
    A_ref: Optional[str] = None,
    B_ref: Optional[str] = None,
    C_ref: Optional[str] = None,
) -> dict:
    """用 BLAS GEMM 计算矩阵乘法；优先用 *_ref 引用 Workspace 矩阵。"""
    try:
        for guard in (
            _guard_inline_workspace_matrix(A_rows, A_ref, "A", "A_rows"),
            _guard_inline_workspace_matrix(B_rows, B_ref, "B", "B_rows"),
            _guard_inline_workspace_matrix(C_rows, C_ref, "C", "C_rows"),
        ):
            if guard:
                return guard
        A = _resolve_dense_matrix(A_rows, A_ref)
        B = _resolve_dense_matrix(B_rows, B_ref)
        C = _resolve_dense_matrix(C_rows, C_ref) if C_ref else C_rows
        if A is None or B is None:
            return {"status": "error", "message": "请提供 A_ref/B_ref 或 A_rows/B_rows"}
        return gemm_blas(
            A_rows=A,
            B_rows=B,
            alpha=alpha,
            beta=beta,
            C_rows=C,
            trans_a=trans_a,
            trans_b=trans_b,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def analyze_matrix_properties_tool(
    A_rows: Optional[list[list[float]]] = None,
    A_csc: Optional[dict] = None,
    tol: float = 1e-10,
    A_ref: Optional[str] = None,
) -> dict:
    """分析矩阵性质；优先用 A_ref 引用 Workspace 矩阵。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows, A_ref, "A", "A_rows")
        if guard:
            return guard
        if A_ref:
            A_rows = _workspace_store.get_matrix(A_ref, sparse=False)
            A_csc = None
        return analyze_matrix_properties(A_rows=A_rows, A_csc=A_csc, tol=tol)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def choose_nla_algorithm_tool(
    task_type: str,
    A_rows: Optional[list[list[float]]] = None,
    A_csc: Optional[dict] = None,
    A_ref: Optional[str] = None,
) -> dict:
    """根据任务类型与矩阵性质推荐算法；优先用 A_ref 引用 Workspace 矩阵。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        if A_ref:
            A_rows = _workspace_store.get_matrix(A_ref, sparse=False)
            A_csc = None
        return choose_nla_algorithm(task_type=task_type, A_rows=A_rows, A_csc=A_csc)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def call_lapack_tool(
    func_name: str,
    arrays: Optional[dict] = None,
    kwargs: Optional[dict] = None,
    dtype: str = "float64",
    output_names: Optional[list[str]] = None,
    array_refs: Optional[dict[str, str]] = None,
) -> dict:
    """通用 LAPACK driver；array_refs 可把形如 {'a': 'A'} 的数组名解析为 Workspace 变量。"""
    try:
        guard = _guard_inline_arrays(arrays, array_refs)
        if guard:
            return guard
        merged = dict(arrays or {})
        for arg_name, ref_name in (array_refs or {}).items():
            merged[str(arg_name)] = _workspace_store.get_matrix(str(ref_name), sparse=False)
        return call_lapack(func_name=func_name, arrays=merged, kwargs=kwargs, dtype=dtype, output_names=output_names)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def call_blas_tool(
    func_name: str,
    arrays: Optional[dict] = None,
    kwargs: Optional[dict] = None,
    dtype: str = "float64",
    output_names: Optional[list[str]] = None,
    array_refs: Optional[dict[str, str]] = None,
) -> dict:
    """通用 BLAS routine；array_refs 可把形如 {'a': 'A'} 的数组名解析为 Workspace 变量。"""
    try:
        guard = _guard_inline_arrays(arrays, array_refs)
        if guard:
            return guard
        merged = dict(arrays or {})
        for arg_name, ref_name in (array_refs or {}).items():
            merged[str(arg_name)] = _workspace_store.get_matrix(str(ref_name), sparse=False)
        return call_blas(func_name=func_name, arrays=merged, kwargs=kwargs, dtype=dtype, output_names=output_names)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def spsolve_sparse_tool(
    A_csc: Optional[dict] = None,
    b: list[float] | list[list[float]] = None,
    A_rows: Optional[list[list[float]]] = None,
    A_ref: Optional[str] = None,
    b_ref: Optional[str] = None,
) -> dict:
    """用稀疏直接法求解 Ax=b；优先用 A_ref/b_ref 引用 Workspace 变量。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        resolved_csc, resolved_rows = _resolve_sparse_matrix(A_csc=A_csc, A_rows=A_rows, ref=A_ref)
        rhs = _resolve_workspace_value(b_ref) if b_ref else b
        return spsolve_sparse(A_csc=resolved_csc, A_rows=resolved_rows, b=rhs)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def cg_sparse_tool(
    A_csc: Optional[dict] = None,
    b: list[float] = None,
    A_rows: Optional[list[list[float]]] = None,
    tol: float = 1e-8,
    maxiter: Optional[int] = None,
    A_ref: Optional[str] = None,
    b_ref: Optional[str] = None,
) -> dict:
    """用共轭梯度法求解；优先用 A_ref/b_ref 引用 Workspace 变量。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        resolved_csc, resolved_rows = _resolve_sparse_matrix(A_csc=A_csc, A_rows=A_rows, ref=A_ref)
        rhs = _resolve_workspace_value(b_ref) if b_ref else b
        return cg_sparse(A_csc=resolved_csc, A_rows=resolved_rows, b=rhs, tol=tol, maxiter=maxiter)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def gmres_sparse_tool(
    A_csc: Optional[dict] = None,
    b: list[float] = None,
    A_rows: Optional[list[list[float]]] = None,
    tol: float = 1e-8,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    A_ref: Optional[str] = None,
    b_ref: Optional[str] = None,
) -> dict:
    """用 GMRES 求解；优先用 A_ref/b_ref 引用 Workspace 变量。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        resolved_csc, resolved_rows = _resolve_sparse_matrix(A_csc=A_csc, A_rows=A_rows, ref=A_ref)
        rhs = _resolve_workspace_value(b_ref) if b_ref else b
        return gmres_sparse(
            A_csc=resolved_csc,
            A_rows=resolved_rows,
            b=rhs,
            tol=tol,
            restart=restart,
            maxiter=maxiter,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def eigsh_sparse_tool(
    A_csc: Optional[dict] = None,
    A_rows: Optional[list[list[float]]] = None,
    k: int = 6,
    which: str = "LM",
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    return_eigenvectors: bool = True,
    A_ref: Optional[str] = None,
) -> dict:
    """用 eigsh 计算部分特征值；优先用 A_ref 引用 Workspace 矩阵。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        resolved_csc, resolved_rows = _resolve_sparse_matrix(A_csc=A_csc, A_rows=A_rows, ref=A_ref)
        return eigsh_sparse(
            A_csc=resolved_csc,
            A_rows=resolved_rows,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=return_eigenvectors,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def eigs_sparse_tool(
    A_csc: Optional[dict] = None,
    A_rows: Optional[list[list[float]]] = None,
    k: int = 6,
    which: str = "LM",
    tol: float = 0.0,
    maxiter: Optional[int] = None,
    return_eigenvectors: bool = True,
    A_ref: Optional[str] = None,
) -> dict:
    """用 eigs 计算部分特征值；优先用 A_ref 引用 Workspace 矩阵。"""
    try:
        guard = _guard_inline_workspace_matrix(A_rows if A_rows is not None else A_csc, A_ref, "A", "A_rows/A_csc")
        if guard:
            return guard
        resolved_csc, resolved_rows = _resolve_sparse_matrix(A_csc=A_csc, A_rows=A_rows, ref=A_ref)
        return eigs_sparse(
            A_csc=resolved_csc,
            A_rows=resolved_rows,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=return_eigenvectors,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


for _tool, _name in (
    (solve_linear_lapack_tool, "solve_linear_lapack"),
    (least_squares_lapack_tool, "least_squares_lapack"),
    (gemm_blas_tool, "gemm_blas"),
    (analyze_matrix_properties_tool, "analyze_matrix_properties"),
    (choose_nla_algorithm_tool, "choose_nla_algorithm"),
    (call_lapack_tool, "call_lapack"),
    (call_blas_tool, "call_blas"),
    (spsolve_sparse_tool, "spsolve_sparse"),
    (cg_sparse_tool, "cg_sparse"),
    (gmres_sparse_tool, "gmres_sparse"),
    (eigsh_sparse_tool, "eigsh_sparse"),
    (eigs_sparse_tool, "eigs_sparse"),
):
    _tool.__name__ = _name


def _matrix_tool_with_auto_workspace(tool_fn):
    @wraps(tool_fn)
    def _wrapped(*args, **kwargs):
        result = tool_fn(*args, **kwargs)
        source = getattr(tool_fn, "__name__", "matrix_loader")
        result = _normalize_tool_error(result, source=source)
        if isinstance(result, dict) and result.get("status") == "ok" and (
            "A_rows" in result or "A_csc" in result or "_matrix" in result
        ):
            matrix_value = _stored_matrix_from_loader_result(result)
            saved_as = _workspace_store.allocate_ref("A")
            _workspace_store.set_var(
                name=saved_as,
                value=matrix_value,
                source=source,
                notes="矩阵读取工具自动保存",
                role="system_matrix",
                origin=source,
                created_by_tool=source,
            )
            result = _slim_matrix_loader_result(result, saved_as=saved_as, source=source)
        elif isinstance(result, dict):
            result = {k: v for k, v in result.items() if k not in {"A_rows", "A_csc", "_matrix", "resolved_path", "checked_paths"}}
        _workspace_store.write_ans(result, source=source)
        return result

    return _wrapped


_load_matrix_csc_file = _matrix_tool_with_auto_workspace(load_matrix_csc_file)
_load_matrix_csc_content = _matrix_tool_with_auto_workspace(load_matrix_csc_content)
_load_matrix_mtx_gz = _matrix_tool_with_auto_workspace(load_matrix_mtx_gz)
_route_user_task = _tool_with_auto_ans(route_user_task)
_infer_solution_preference = _tool_with_auto_ans(infer_solution_preference)
_analyze_matrix_properties = _tool_with_auto_ans(analyze_matrix_properties_tool)
_build_precondition_checklist = _tool_with_auto_ans(build_precondition_checklist)
_choose_nla_algorithm = _tool_with_auto_ans(choose_nla_algorithm_tool)
_plan_coach_next_step = _tool_with_auto_ans(plan_coach_next_step)
_search_numpy_scipy_docs = _tool_with_auto_ans(search_numpy_scipy_docs)
_get_linalg_backend_info = _tool_with_auto_ans(get_linalg_backend_info)
_call_lapack = _tool_with_auto_ans(call_lapack_tool)
_call_blas = _tool_with_auto_ans(call_blas_tool)
_solve_linear_lapack = _tool_with_auto_ans(solve_linear_lapack_tool)
_least_squares_lapack = _tool_with_auto_ans(least_squares_lapack_tool)
_gemm_blas = _tool_with_auto_ans(gemm_blas_tool)
_get_sparse_backend_info = _tool_with_auto_ans(get_sparse_backend_info)
_spsolve_sparse = _tool_with_auto_ans(spsolve_sparse_tool)
_cg_sparse = _tool_with_auto_ans(cg_sparse_tool)
_gmres_sparse = _tool_with_auto_ans(gmres_sparse_tool)
_eigsh_sparse = _tool_with_auto_ans(eigsh_sparse_tool)
_eigs_sparse = _tool_with_auto_ans(eigs_sparse_tool)
_run_python_snippet = _tool_with_auto_ans(run_python_snippet)
_run_python_workspace_snippet = _tool_with_auto_ans(run_python_workspace_snippet)
_add_memory = _tool_with_auto_ans(add_memory)
_search_memory = _tool_with_auto_ans(search_memory)
_update_memory = _tool_with_auto_ans(update_memory)
_delete_memory = _tool_with_auto_ans(delete_memory)
_summarize_memory = _tool_with_auto_ans(summarize_memory)
_extract_memory_from_dialogue_tool = _tool_with_auto_ans(extract_memory_from_dialogue)
_list_pending_memories = _tool_with_auto_ans(list_pending_memories)
_approve_pending_memory = _tool_with_auto_ans(approve_pending_memory)
_get_memory = _tool_with_auto_ans(get_memory)


def _context_attr(obj: object, path: tuple[str, ...]) -> object | None:
    current = obj
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _workspace_key_from_callback_context(callback_context: CallbackContext) -> str | None:
    for path in (
        ("session_id",),
        ("session", "id"),
        ("_invocation_context", "session", "id"),
        ("invocation_context", "session", "id"),
    ):
        value = _context_attr(callback_context, path)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _workspace_key_from_request(llm_request: LlmRequest) -> str | None:
    for content in reversed(getattr(llm_request, "contents", None) or []):
        for part in reversed(getattr(content, "parts", None) or []):
            text = getattr(part, "text", None)
            if not isinstance(text, str):
                continue
            match = _FRONTEND_SESSION_RE.search(text)
            if match:
                return match.group(1)
    return None


def _activate_workspace_for_turn(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    _set_active_workspace_key(
        _workspace_key_from_request(llm_request) or _workspace_key_from_callback_context(callback_context)
    )


def _filter_unsupported_content_parts(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    _activate_workspace_for_turn(callback_context, llm_request)
    if not getattr(llm_request, "contents", None):
        return None

    if _memory_bridge is not None and _memory_store is not None:
        try:
            if bool(getattr(_memory_store.config, "auto_write_mode", False)):
                _memory_bridge.auto_extract_after_turns(llm_request.contents)
        except Exception:
            pass

    def _normalize_win_paths(text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        return _WIN_PATH_RE.sub(lambda m: m.group(0).replace("\\", "/"), text)

    user_indices = [
        idx
        for idx, content in enumerate(llm_request.contents)
        if (getattr(content, "role", "") or "") == "user"
    ]
    last_user_idx = user_indices[-1] if user_indices else -1

    new_contents = []
    for idx, content in enumerate(llm_request.contents):
        role = getattr(content, "role", "") or ""
        if not getattr(content, "parts", None):
            new_contents.append(content)
            continue

        new_parts = []
        for part in content.parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                mime = getattr(inline, "mime_type", "") or ""
                if not any(mime.startswith(prefix) for prefix in _SUPPORTED_MIME_PREFIXES):
                    new_parts.append(
                        types.Part(
                            text=(
                                "[已省略不支持的附件类型: "
                                + mime
                                + "。请提供文件路径（建议用 /）或粘贴文本内容。]"
                            )
                        )
                    )
                    continue
                new_parts.append(part)
                continue

            if role == "user":
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    if _memory_bridge is not None and idx == last_user_idx:
                        if "[历史记忆召回]" not in text:
                            try:
                                text = _memory_bridge.inject_memory_context(text, top_k=3)
                                if "[历史记忆召回]" in text:
                                    text += (
                                        "\n\n[记忆边界提示] 上述历史记忆只代表过往经验，"
                                        "不能证明当前会话存在同名文件、上传 URI、路径或 Workspace 变量；"
                                        "当前文件状态必须以 <frontend-files>、工具返回和 workspace_list 为准。"
                                    )
                            except Exception:
                                pass
                    if _WIN_PATH_RE.search(text):
                        text = _normalize_win_paths(text)
                    part = types.Part(text=text)
            new_parts.append(part)
        new_contents.append(types.Content(role=content.role, parts=new_parts))

    llm_request.contents = new_contents
    return None


def _enforce_model_output_format(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    del callback_context
    content = getattr(llm_response, "content", None)
    if content is None or not getattr(content, "parts", None):
        return None

    changed = False
    new_parts = []
    for part in content.parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            normalized = sanitize_latex_markdown(text)
            if normalized != text:
                part = types.Part(text=normalized)
                changed = True
        new_parts.append(part)

    if not changed:
        return None

    llm_response.content = types.Content(
        role=getattr(content, "role", None) or "model",
        parts=new_parts,
    )
    return llm_response


root_agent = Agent(
    model=_llm,
    name="nla_agent",
    description="数值线性代数助手：矩阵读取、算法策略、文档检索与代码验证。",
    before_model_callback=_filter_unsupported_content_parts,
    after_model_callback=_enforce_model_output_format,
    instruction="""
你是数值线性代数助手。采用“目标驱动 + 证据驱动”的轻量流程，不机械走固定阶段。

【自适应流程（可跳阶段）】
1) 每轮先用一句中文复述目标；优先调用 route_user_task。仅在确有复用价值时调用 search_memory。
2) 仅在缺信息时补问：优先补“求解目标/输出偏好/必要条件/矩阵关键性质”中当前最影响决策的一项，不做全量盘问。
3) 若用户已明确“直接给结果/代码/快速求解”，且前提充分，可直接进入 plan 或 solve；避免重复确认。
4) 仅在用户提供显式矩阵或矩阵文件时，读取并做性质分析（analyze_matrix_properties）；读取 CSC/Matrix Market 文件时只使用 file_id 对应的 `nla-upload://...`，不要使用或索要后端绝对路径，也不要为了读取文件而把完整文件内容塞进 JSON 参数；否则先给条件化方案与最小补充问题。
5) 若用户通过前端聊天框附加文件，前端会在 `<frontend-files>` 中提供本轮附件：Text 文件可直接阅读 content；Matrix 文件必须优先使用 file_id/uploadUri（形如 `nla-upload://<file_id>`）调用 load_matrix_csc_file / load_matrix_mtx_gz 载入并注册到 Workspace。若上传失败且只能看到 shortPreview，不要直接臆造完整矩阵；请提示用户启动上传服务或重新上传。小型 CSC 文本附件确需读取时，才可用 load_matrix_csc_content。
6) Workspace Protocol：Workspace 是数值对象注册表，不是大数组中转站。一旦矩阵、向量或分解结果已保存到 Workspace，必须按“摘要 → 角色绑定/决策 → 引用计算 → 汇报”执行。先用 workspace_summary / workspace_stats / workspace_structure 获取必要信息，计算时必须用 `A_ref`、`B_ref`、`C_ref`、`b_ref` 或 `array_refs`。禁止调用 workspace_get 试图展开大对象，禁止把 Workspace 中已有的大矩阵重新写入 `workspace_set` / `A_rows` / `A_csc` / `arrays`，禁止猜测临时文件、共享内存、base64、隐藏路径、逐行逐列导出等隐式访问通道。workspace_set 只用于标量、小向量和轻量变量；已有对象需要命名时用 workspace_alias，需要明确用途时用 workspace_bind_role。workspace_slice 只用于少量局部核验，不得通过多次切片拼回完整对象。
7) 工具返回 `workspace_protocol_violation`、`fallback_required` 或普通 error 后，立即停止当前失败路径：不要继续发明新的访问方式；先报告失败原因，再从 `next_allowed_actions`、Workspace 摘要和可用后端工具重新规划。若缺少关键输入，最多问一个最影响决策的补充问题。
8) 稠密常见问题优先使用 solve_linear_lapack / least_squares_lapack / gemm_blas；若用户指定 LAPACK/BLAS driver（如 gejsv/dgejsv/gesvd/geev/qr/gemm 等）或高层工具不覆盖，优先用 call_lapack / call_blas；稀疏问题优先使用 spsolve_sparse / cg_sparse / gmres_sparse / eigsh_sparse / eigs_sparse。涉及 Workspace 矩阵、向量或分解时，普通 run_python_snippet 不能访问 Workspace，禁止用它猜文件名、扫描 uploads/临时目录、从记忆复原矩阵或手动构造标准测试矩阵；若确需 Python 兜底，必须调用 run_python_workspace_snippet 并通过 array_refs / scalar_refs 注入对象。
9) run_python_workspace_snippet 只用于专用后端工具不覆盖的数值验证或 SciPy 高层调用；代码中不得读取文件、调用 scipy.io.mmread、使用浏览器路径/裸文件名/上传目录，也不得把 Workspace 对象导出再读回。若安全检查拒绝代码，必须回到专用后端工具或询问用户补充输入。
10) 不要在 Windows 上直接用 ctypes 查找 lapack.dll/.so 或 dgejsv_ 符号；LAPACK/BLAS 统一通过 scipy.linalg.lapack / scipy.linalg.blas 的 f2py 包装层调用。

【输出与记忆】
- 默认先给简洁结论与关键依据，再按需展开推导与代码。
- 历史记忆只能提供经验模式，不能作为当前会话中文件已上传、路径可读或 Workspace 变量存在的证据；当前状态必须以本轮 `<frontend-files>`、工具返回和 workspace_list 为准。
- 工作区是本次对话临时数值对象注册表，风格类似 MATLAB Workspace。用户要求“记住/存一下/以后用”标量、小向量或轻量变量时，才调用 workspace_set；矩阵文件读取工具成功后会自动注册为不冲突的引用（首个通常为 A，后续为 A2、A3 等）并写入 ans。返回中若有 saved_as/matrix_ref/workspace_saved=true，禁止再调用 workspace_set 重复保存同一矩阵；需要更清晰名称时调用 workspace_alias，需要标记用途时调用 workspace_bind_role。多矩阵任务必须保留每个 matrix_ref，并在计算前确认或推断它们分别对应 system_matrix/rhs/preconditioner/operand_left/operand_right 等角色。后续用户说“刚才的 A/b/x/ans”时，先用 workspace_list 或工具返回的 handle 确认变量名；计算时直接传 `A_ref=\"A\"` 等引用。workspace_get 对矩阵、大向量和分解结果只返回句柄摘要，不会返回完整元素。
- 若本轮使用了 Workspace 对象完成计算，最终回答要简短说明使用的引用参数（例如 `A_ref=\"A\"` 或 `array_refs={\"a\":\"A\"}`），不要展示或复原完整矩阵内容。
- 计算工具的最近结果会自动写入 ans；若结果中有明确变量名（如解向量 x、矩阵 A、右端项 b、残差 residual），应再调用 workspace_set 用清晰变量名保存，便于前端 Workspace 展示。
- 每轮最终回答末尾都要同步工作区：先调用 workspace_list(detail=true)，然后在回答最后追加一段机器可读标记 `<nla-workspace>{...}</nla-workspace>`，其中 JSON 就是 workspace_list(detail=true) 的返回内容。该标记只给前端解析，不要解释给用户。
- 需要输出数学公式时，必须使用前端可直接渲染的 LaTeX Markdown 格式：行内公式用 `$...$`，块级公式单独成段并用 `$$...$$` 包裹。
- 不要用 `\\(...\\)`、`\\[...\\]`、图片、Unicode 近似符号或纯文本排版替代公式；矩阵、向量、范数、残差、条件数、分解式等都应写成标准 LaTeX，例如 `$$A=QR,\\quad \\|r_k\\|_2=\\|b-Ax_k\\|_2$$`。
- 多行推导优先放在一个块级公式中，并使用 `aligned`、`bmatrix`、`pmatrix` 等 KaTeX 支持的环境，便于前端 Markdown + KaTeX 渲染。
- 输出最终回答前必须自检：任意未转义 `$` 必须成对出现，任意 `$$` 必须成对出现；禁止输出缺少起止 `$` 的公式片段。若公式尚未写完或不确定能否闭合，不要输出裸 `$`，改写为普通中文描述或完整的块级公式。
- 回合末调用 extract_memory_from_dialogue；自动写入模式可 add_memory(require_confirmation=false)。
- 若用户询问记忆库内容、待审核记忆或要求确认入库，优先使用 list_pending_memories / approve_pending_memory / get_memory。
- 全程中文；工具参数必须为严格 JSON（双引号键名与字符串）。
""",
    tools=[
        _load_matrix_csc_file,
        _load_matrix_csc_content,
        _load_matrix_mtx_gz,
        _route_user_task,
        _infer_solution_preference,
        _analyze_matrix_properties,
        _build_precondition_checklist,
        _choose_nla_algorithm,
        _plan_coach_next_step,
        _search_numpy_scipy_docs,
        _get_linalg_backend_info,
        _call_lapack,
        _call_blas,
        _solve_linear_lapack,
        _least_squares_lapack,
        _gemm_blas,
        _get_sparse_backend_info,
        _spsolve_sparse,
        _cg_sparse,
        _gmres_sparse,
        _eigsh_sparse,
        _eigs_sparse,
        _run_python_snippet,
        _run_python_workspace_snippet,
        _add_memory,
        _search_memory,
        _update_memory,
        _delete_memory,
        _summarize_memory,
        _extract_memory_from_dialogue_tool,
        _list_pending_memories,
        _approve_pending_memory,
        _get_memory,
        workspace_set,
        workspace_get,
        workspace_list,
        workspace_clear,
        workspace_alias,
        workspace_bind_role,
        workspace_summary,
        workspace_stats,
        workspace_structure,
        workspace_slice,
        workspace_audit,
        who,
        whos,
        clear,
    ],
)
