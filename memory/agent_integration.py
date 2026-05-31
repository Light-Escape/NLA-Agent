from __future__ import annotations

import json
import re
from typing import Any

from .store import NLAMemoryStore


class MemoryAgentBridge:
    MIN_INJECTION_SCORE = 0.08
    _INJECTED_MEMORY_BLOCK_RE = re.compile(
        r"\[历史记忆召回\][\s\S]*?\[当前用户问题\]\s*",
        re.IGNORECASE,
    )
    _JACOBI_SVD_QUERY_RE = re.compile(
        r"(jacobi\s*svd|svd.*jacobi|gejsv|gesvj).*(lapack|scipy)|"
        r"(lapack|scipy).*(jacobi\s*svd|svd.*jacobi|gejsv|gesvj)",
        re.IGNORECASE,
    )
    _EXEC_FAILURE_RE = re.compile(
        r"(status['\"]?\s*:\s*['\"]?error|traceback|exception|执行超时|syntaxerror|nameerror|valueerror|unicode)",
        re.IGNORECASE,
    )
    _FIX_HINT_RE = re.compile(
        r"(修复|解决|改为|避免|建议|根因|设置|检查|fix|resolved|workaround)",
        re.IGNORECASE,
    )

    def __init__(self, store: NLAMemoryStore):
        self.store = store

    @classmethod
    def _static_api_hints(cls, user_question: str) -> str:
        if not cls._JACOBI_SVD_QUERY_RE.search(user_question or ""):
            return ""
        return (
            "高风险 API 记忆: Jacobi SVD 通过 scipy.linalg.lapack 时，优先核对 "
            "gejsv/dgejsv/sgejsv 包装；常见正确模板是 "
            "gejsv = lapack.get_lapack_funcs('gejsv', (A_f,)); "
            "s, u, v, workout, iworkout, info = gejsv(A_f, jobu=1, jobv=1)。"
            "不要按 numpy.linalg.svd 的 (U, s, Vh) 返回顺序理解，并检查 info == 0。"
        )

    @classmethod
    def strip_injected_memory(cls, text: str) -> str:
        cleaned = cls._INJECTED_MEMORY_BLOCK_RE.sub("", text or "").strip()
        return cleaned or (text or "")

    def _relevant_recall(self, user_question: str, top_k: int = 3, filters: dict | None = None) -> dict:
        result = self.store.search_memory(user_question, top_k=top_k, filters=filters)
        min_score = float(getattr(self.store.config, "min_recall_score", self.MIN_INJECTION_SCORE))
        items = []
        for row in result.get("items", []) or []:
            score = float(row.get("score", row.get("similarity", 0.0)) or 0.0)
            if score >= min_score:
                items.append(row)
        return {"status": result.get("status", "ok"), "query": result.get("query", user_question), "items": items}

    def recall_context(self, user_question: str, top_k: int = 3, filters: dict | None = None) -> str:
        clean_question = self.strip_injected_memory(user_question)
        result = self._relevant_recall(clean_question, top_k=top_k, filters=filters)
        memory_ctx = self.store.summarize_memory(result)
        static_hint = self._static_api_hints(clean_question)
        if static_hint:
            if result.get("items"):
                return f"{memory_ctx}\n{static_hint}"
            return static_hint
        if not result.get("items"):
            return ""
        return memory_ctx

    def inject_memory_context(self, user_question: str, top_k: int = 3, filters: dict | None = None) -> str:
        clean_question = self.strip_injected_memory(user_question)
        memory_ctx = self.recall_context(user_question=clean_question, top_k=top_k, filters=filters)
        if not memory_ctx:
            return clean_question
        return (
            "[历史记忆召回]\n"
            f"{memory_ctx}\n\n"
            "[使用要求]\n"
            "若历史记忆包含 API 调用坑点、正确调用约定或返回值说明，先按该记忆核对方案；"
            "不要凭函数名猜测调用方式，必要时再查官方文档验证。\n\n"
            "[当前用户问题]\n"
            f"{clean_question}"
        )

    @staticmethod
    def _content_text(content: Any) -> str:
        parts = getattr(content, "parts", []) or []
        out: list[str] = []
        for p in parts:
            txt = getattr(p, "text", None)
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
            fn_resp = getattr(p, "function_response", None)
            if fn_resp is not None:
                fn_name = getattr(fn_resp, "name", "") or ""
                payload = getattr(fn_resp, "response", None)
                if payload is not None:
                    try:
                        out.append(f"{fn_name} {json.dumps(payload, ensure_ascii=False)}".strip())
                    except Exception:
                        out.append(f"{fn_name} {str(payload)}".strip())
        return "\n".join(chunk for chunk in out if chunk).strip()

    def _extract_latest_execution_issue(self, contents: list[Any]) -> dict:
        timeline: list[dict] = []
        for content in contents or []:
            role = (getattr(content, "role", "") or "").strip().lower()
            text = self.strip_injected_memory(self._content_text(content))
            if text:
                timeline.append({"role": role, "text": text})
        if not timeline:
            return {"status": "skipped", "reason": "empty_timeline"}

        for idx in range(len(timeline) - 1, -1, -1):
            row = timeline[idx]
            text = row["text"]
            role = row["role"]
            if role not in {"tool", "function", "model"}:
                continue
            if "run_python_snippet" not in text and "stderr" not in text and "exit_code" not in text:
                continue
            if not self._EXEC_FAILURE_RE.search(text):
                continue

            user_context = ""
            for j in range(idx - 1, -1, -1):
                if timeline[j]["role"] == "user":
                    user_context = timeline[j]["text"]
                    break

            fix_hint = ""
            for j in range(idx + 1, len(timeline)):
                if timeline[j]["role"] != "model":
                    continue
                cand = timeline[j]["text"]
                if self._FIX_HINT_RE.search(cand):
                    fix_hint = cand
                    break

            return self.store.ingest_execution_issue(
                tool_name="run_python_snippet",
                error_text=text,
                fix_text=fix_hint,
                context_text=user_context,
                require_confirmation=not self.store.config.auto_write_mode,
            )
        return {"status": "skipped", "reason": "no_execution_issue_detected"}

    def auto_extract_after_turns(self, contents: list[Any]) -> dict:
        """
        从对话历史中抓取最近一个 user->model 配对，自动抽取并入库。
        """
        issue_ret = self._extract_latest_execution_issue(contents)
        if len(contents) < 2:
            return {"status": "ok", "dialogue": {"status": "skipped", "reason": "not_enough_history"}, "execution_issue": issue_ret}
        last_user_text = ""
        last_model_text = ""
        for content in reversed(contents):
            role = getattr(content, "role", "")
            text = self.strip_injected_memory(self._content_text(content))
            if role == "model" and not last_model_text and text:
                last_model_text = text.strip()
            elif role == "user" and last_model_text and text:
                last_user_text = text.strip()
                break
        if not (last_user_text and last_model_text):
            return {"status": "ok", "dialogue": {"status": "skipped", "reason": "no_complete_turn"}, "execution_issue": issue_ret}
        turn_ret = self.store.auto_ingest_last_turn(last_user_text, last_model_text)
        return {"status": "ok", "dialogue": turn_ret, "execution_issue": issue_ret}
