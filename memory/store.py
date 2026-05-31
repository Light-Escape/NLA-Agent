from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MemoryConfig, load_memory_config
from .embedder import HashEmbeddingFunction
from .extractor import extract_memory_from_dialogue
from .normalizer import expand_query_text, make_embedding_text, memory_signature, normalize_problem

try:
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class _InMemoryCollection:
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn
        self.docs: dict[str, dict[str, Any]] = {}

    def add(self, ids, documents, metadatas):
        for i, _id in enumerate(ids):
            self.docs[_id] = {
                "id": _id,
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": self.embedding_fn([documents[i]])[0],
            }

    def get(self, ids=None):
        if ids is None:
            rows = list(self.docs.values())
        else:
            rows = [self.docs[_id] for _id in ids if _id in self.docs]
        return {
            "ids": [r["id"] for r in rows],
            "documents": [r["document"] for r in rows],
            "metadatas": [r["metadata"] for r in rows],
        }

    def update(self, ids, documents, metadatas):
        for i, _id in enumerate(ids):
            if _id not in self.docs:
                continue
            self.docs[_id].update(
                {
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "embedding": self.embedding_fn([documents[i]])[0],
                }
            )

    def delete(self, ids):
        for _id in ids:
            self.docs.pop(_id, None)

    def query(self, query_texts, n_results):
        q_emb = self.embedding_fn([query_texts[0]])[0]
        scored = []
        for row in self.docs.values():
            emb = row["embedding"]
            sim = sum(a * b for a, b in zip(q_emb, emb))
            scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:n_results]
        return {
            "ids": [[r["id"] for _, r in top]],
            "documents": [[r["document"] for _, r in top]],
            "metadatas": [[r["metadata"] for _, r in top]],
            "distances": [[1.0 - s for s, _ in top]],
        }


class NLAMemoryStore:
    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or load_memory_config()
        self.embedding_fn = HashEmbeddingFunction(self.config.embedding_dim)
        self.base_dir = Path(__file__).resolve().parent.parent
        self.persist_dir = self.base_dir / self.config.persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.version_log = self.persist_dir / "versions.jsonl"
        self._seen_dialogue_signatures: set[str] = set()

        self.client = None
        if chromadb is not None:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.pending_collection = self.client.get_or_create_collection(
                name=self.config.pending_collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chromadb"
        else:  # pragma: no cover
            self.collection = _InMemoryCollection(self.embedding_fn)
            self.pending_collection = _InMemoryCollection(self.embedding_fn)
            self.backend = "in-memory"

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is None:
            return
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
                return
        except Exception:
            pass
        try:
            system = getattr(client, "_system", None)
            stop = getattr(system, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass

    _ISSUE_QUERY_RE = re.compile(r"(报错|错误|异常|失败|崩溃|bug|error|exception|traceback)", re.IGNORECASE)
    _EXEC_ISSUE_QUERY_RE = re.compile(
        r"(代码执行|run_python_snippet|stderr|exit_code|超时|编码|中文显示|unicode|traceback|nameerror|syntaxerror)",
        re.IGNORECASE,
    )
    _API_USAGE_QUERY_RE = re.compile(
        r"(jacobi\s*svd|gejsv|gesvj|scipy\s*\.?\s*linalg\s*\.?\s*lapack|get_lapack_funcs|lapack)",
        re.IGNORECASE,
    )
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
    _TOKEN_RE = re.compile(r"[a-zA-Z0-9_+\-.]+|[\u4e00-\u9fff]+")

    @staticmethod
    def _is_issue_resolution_item(item: dict) -> bool:
        source_meta = item.get("source_meta", {}) or {}
        return source_meta.get("experience_type") == "issue_resolution"

    @staticmethod
    def _is_execution_issue_item(item: dict) -> bool:
        source_meta = item.get("source_meta", {}) or {}
        return source_meta.get("experience_type") == "execution_issue"

    @staticmethod
    def _is_api_usage_item(item: dict) -> bool:
        source_meta = item.get("source_meta", {}) or {}
        return source_meta.get("experience_type") == "api_usage"

    @staticmethod
    def _token_overlap_score(query: str, item: dict) -> float:
        query_tokens = set(normalize_problem(query).split())
        if not query_tokens:
            return 0.0
        item_text = " ".join(
            [
                item.get("problem_pattern", ""),
                " ".join(item.get("math_topic", []) or []),
                item.get("solution_pattern", ""),
                item.get("code_hint", ""),
                " ".join(item.get("failure_modes", []) or []),
            ]
        )
        item_tokens = set(normalize_problem(item_text).split())
        if not item_tokens:
            return 0.0
        important = {"jacobi", "svd", "gejsv", "gesvj", "lapack", "scipy", "get_lapack_funcs"}
        overlap = query_tokens & item_tokens
        weighted = len(overlap) + sum(1 for tok in overlap if tok in important)
        return min(0.12, 0.02 * weighted)

    @staticmethod
    def _short_text(text: str, max_len: int = 180) -> str:
        t = normalize_problem(text or "")
        if len(t) <= max_len:
            return t
        return t[: max_len - 3].rstrip() + "..."

    @classmethod
    def _retrieval_tokens(cls, text: str) -> list[str]:
        normalized = normalize_problem(text or "")
        tokens: list[str] = []
        for match in cls._TOKEN_RE.findall(normalized):
            if not match:
                continue
            tokens.append(match)
            if cls._CJK_RE.fullmatch(match):
                if len(match) <= 6:
                    tokens.extend(match)
                tokens.extend(match[i : i + 2] for i in range(max(0, len(match) - 1)))
                tokens.extend(match[i : i + 3] for i in range(max(0, len(match) - 2)))
        seen: set[str] = set()
        out: list[str] = []
        for tok in tokens:
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

    @classmethod
    def _bm25_like_score(cls, query: str, item: dict, corpus_terms: list[set[str]] | None = None, index: int = 0) -> float:
        q_terms = cls._retrieval_tokens(query)
        if not q_terms:
            return 0.0
        item_text = make_embedding_text(item)
        doc_terms = cls._retrieval_tokens(item_text)
        if not doc_terms:
            return 0.0
        tf: dict[str, int] = {}
        for tok in doc_terms:
            tf[tok] = tf.get(tok, 0) + 1
        doc_len = max(1, len(doc_terms))
        avgdl = doc_len
        n_docs = 1
        if corpus_terms:
            n_docs = max(1, len(corpus_terms))
            avgdl = max(1.0, sum(len(row) for row in corpus_terms) / n_docs)

        score = 0.0
        k1 = 1.2
        b = 0.75
        for tok in q_terms:
            freq = tf.get(tok, 0)
            if not freq:
                continue
            if corpus_terms:
                df = sum(1 for row in corpus_terms if tok in row)
            else:
                df = 1
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avgdl))
        return max(0.0, min(1.0, score / max(1.0, len(q_terms))))

    @classmethod
    def _entity_score(cls, query: str, item: dict) -> float:
        query_terms = set(cls._retrieval_tokens(query))
        if not query_terms:
            return 0.0
        source_meta = item.get("source_meta", {}) or {}
        entity_text = " ".join(
            [
                " ".join(item.get("math_topic", []) or []),
                " ".join(item.get("matrix_properties", []) or []),
                source_meta.get("api_family", ""),
                source_meta.get("lapack_driver", ""),
                source_meta.get("tool_name", ""),
                item.get("code_hint", ""),
            ]
        )
        entity_terms = set(cls._retrieval_tokens(entity_text))
        if not entity_terms:
            return 0.0
        overlap = query_terms & entity_terms
        if not overlap:
            return 0.0
        important = {"jacobi", "svd", "gejsv", "gesvj", "lapack", "scipy", "cg", "gmres", "pcg", "spd"}
        weighted = len(overlap) + sum(2 for tok in overlap if tok in important)
        return min(1.0, weighted / 8.0)

    @staticmethod
    def _recency_usage_score(item: dict) -> float:
        use_count = int(item.get("use_count", 0) or 0)
        usage = min(0.05, 0.01 * use_count)
        return usage

    def _boosted_similarity(self, query: str, item: dict, sim: float) -> float:
        boosted = float(sim)
        if self._ISSUE_QUERY_RE.search(query or "") and self._is_issue_resolution_item(item):
            # 问题排障语境下，优先召回历史“故障->修复”经验。
            boosted += 0.08
        if self._EXEC_ISSUE_QUERY_RE.search(query or "") and self._is_execution_issue_item(item):
            # 代码执行故障场景下，优先召回“执行错误 -> 修复方案”经验。
            boosted += 0.12
        if self._API_USAGE_QUERY_RE.search(query or "") and self._is_api_usage_item(item):
            # API 调用坑点场景下，优先召回“正确调用约定/返回值”经验。
            boosted += 0.14
        boosted += self._token_overlap_score(query, item)
        return max(0.0, min(1.0, boosted))

    def _hybrid_score(
        self,
        query: str,
        item: dict,
        vector_similarity: float,
        corpus_terms: list[set[str]] | None = None,
        corpus_index: int = 0,
    ) -> float:
        vector_score = self._boosted_similarity(query, item, vector_similarity)
        lexical_score = self._bm25_like_score(query, item, corpus_terms=corpus_terms, index=corpus_index)
        entity_score = self._entity_score(query, item)
        return max(
            0.0,
            min(
                1.0,
                0.55 * vector_score
                + 0.30 * lexical_score
                + 0.10 * entity_score
                + self._recency_usage_score(item),
            ),
        )

    def _log_version(self, action: str, memory_id: str, item: dict):
        line = {
            "timestamp": _utc_now(),
            "action": action,
            "memory_id": memory_id,
            "item": item,
        }
        with self.version_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _normalize_item(self, item: dict) -> dict:
        cleaned = dict(item or {})
        cleaned["problem_pattern"] = normalize_problem(cleaned.get("problem_pattern", ""))
        cleaned["solution_pattern"] = normalize_problem(cleaned.get("solution_pattern", ""))
        cleaned["math_topic"] = list(cleaned.get("math_topic", []) or [])
        cleaned["matrix_properties"] = list(cleaned.get("matrix_properties", []) or [])
        cleaned["failure_modes"] = list(cleaned.get("failure_modes", []) or [])
        cleaned["assumptions"] = list(cleaned.get("assumptions", []) or [])
        cleaned["source_meta"] = dict(cleaned.get("source_meta", {}) or {})
        cleaned.setdefault("method_reason", "")
        cleaned.setdefault("complexity_hint", "")
        cleaned.setdefault("code_hint", "")
        cleaned.setdefault("version", 1)
        cleaned.setdefault("status", "active")
        cleaned.setdefault("merged_from", [])
        cleaned.setdefault("memory_type", cleaned.get("source_meta", {}).get("experience_type", "solution"))
        cleaned.setdefault("namespace", cleaned.get("source_meta", {}).get("namespace", "nla/default"))
        cleaned.setdefault("confidence", float(cleaned.get("quality_score", 0.6)))
        cleaned.setdefault("evidence", [])
        cleaned.setdefault("source_turn_id", cleaned.get("source_meta", {}).get("source_turn_id", ""))
        cleaned.setdefault("last_used_at", "")
        cleaned.setdefault("use_count", 0)
        cleaned.setdefault("valid_from", cleaned.get("created_at", _utc_now()))
        cleaned.setdefault("invalid_at", "")
        cleaned.setdefault("created_at", _utc_now())
        cleaned["updated_at"] = _utc_now()
        cleaned["embedding_text"] = make_embedding_text(cleaned)
        cleaned["quality_score"] = float(cleaned.get("quality_score", 0.6))
        cleaned["confidence"] = float(cleaned.get("confidence", cleaned["quality_score"]))
        cleaned["signature"] = memory_signature(cleaned)
        return cleaned

    def _to_metadata(self, item: dict) -> dict[str, Any]:
        return {
            "problem_pattern": item["problem_pattern"],
            "math_topic": "|".join(item["math_topic"]),
            "matrix_properties": "|".join(item["matrix_properties"]),
            "version": int(item.get("version", 1)),
            "status": item.get("status", "active"),
            "quality_score": float(item.get("quality_score", 0.0)),
            "memory_type": item.get("memory_type", "solution"),
            "namespace": item.get("namespace", "nla/default"),
            "confidence": float(item.get("confidence", item.get("quality_score", 0.0))),
            "use_count": int(item.get("use_count", 0) or 0),
            "last_used_at": item.get("last_used_at", ""),
            "valid_from": item.get("valid_from", ""),
            "invalid_at": item.get("invalid_at", ""),
            "signature": item.get("signature", ""),
            "item_json": json.dumps(item, ensure_ascii=False),
        }

    def _from_metadata(self, metadata: dict[str, Any]) -> dict:
        if "item_json" in metadata:
            return json.loads(metadata["item_json"])
        return metadata

    def _similarity(self, distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - float(distance)))

    def _merge_items(self, old_item: dict, new_item: dict) -> dict:
        merged = dict(old_item)
        merged["math_topic"] = sorted(set(old_item["math_topic"]) | set(new_item["math_topic"]))
        merged["matrix_properties"] = sorted(
            set(old_item["matrix_properties"]) | set(new_item["matrix_properties"])
        )
        merged["failure_modes"] = sorted(set(old_item["failure_modes"]) | set(new_item["failure_modes"]))
        merged["assumptions"] = sorted(set(old_item["assumptions"]) | set(new_item["assumptions"]))
        merged["quality_score"] = max(old_item["quality_score"], new_item["quality_score"])
        merged["solution_pattern"] = old_item["solution_pattern"] or new_item["solution_pattern"]
        merged["method_reason"] = old_item["method_reason"] or new_item["method_reason"]
        merged["complexity_hint"] = old_item["complexity_hint"] or new_item["complexity_hint"]
        merged["code_hint"] = old_item["code_hint"] or new_item["code_hint"]
        merged["source_meta"] = {
            **old_item.get("source_meta", {}),
            **new_item.get("source_meta", {}),
            "merged_at": _utc_now(),
        }
        merged["merged_from"] = sorted(
            set(old_item.get("merged_from", [])) | set(new_item.get("merged_from", []))
        )
        merged["version"] = int(old_item.get("version", 1)) + 1
        merged["updated_at"] = _utc_now()
        merged["embedding_text"] = make_embedding_text(merged)
        merged["signature"] = memory_signature(merged)
        return merged

    def add_memory(self, item: dict, require_confirmation: bool = True) -> dict:
        normalized = self._normalize_item(item)
        if normalized["quality_score"] < self.config.min_quality_score and not self._is_issue_resolution_item(normalized):
            return {"status": "filtered", "reason": "low_quality", "item": normalized}
        if self.config.readonly_mode:
            return {"status": "skipped", "reason": "readonly_mode"}

        should_prioritize_store = self._is_issue_resolution_item(normalized) or self._is_api_usage_item(normalized)
        target = self.collection if (should_prioritize_store or not require_confirmation) else self.pending_collection
        existing = self.search_memory(normalized["embedding_text"], top_k=1)
        if existing["items"]:
            top = existing["items"][0]
            sim = top["similarity"]
            if sim >= self.config.dedup_similarity_threshold:
                return {"status": "deduplicated", "memory_id": top["id"], "similarity": sim}
            if sim >= self.config.merge_similarity_threshold and (should_prioritize_store or not require_confirmation):
                merged = self._merge_items(top["item"], normalized)
                return self.update_memory(top["id"], merged, _action="merge")

        memory_id = "mem_" + uuid.uuid4().hex[:12]
        target.add(
            ids=[memory_id],
            documents=[normalized["embedding_text"]],
            metadatas=[self._to_metadata(normalized)],
        )
        action = "add"
        if require_confirmation and not should_prioritize_store:
            action = "add_pending"
        self._log_version(action, memory_id, normalized)
        return {
            "status": "pending" if action == "add_pending" else "stored",
            "memory_id": memory_id,
            "item": normalized,
        }

    def approve_pending_memory(self, memory_id: str) -> dict:
        found = self.pending_collection.get(ids=[memory_id])
        if not found["ids"]:
            return {"status": "error", "message": "pending memory not found"}
        metadata = found["metadatas"][0]
        item = self._from_metadata(metadata)
        item.setdefault("source_meta", {})
        item["source_meta"]["human_confirmed"] = True
        item["updated_at"] = _utc_now()
        self.collection.add(
            ids=[memory_id],
            documents=[item["embedding_text"]],
            metadatas=[self._to_metadata(item)],
        )
        self.pending_collection.delete(ids=[memory_id])
        self._log_version("approve", memory_id, item)
        return {"status": "approved", "memory_id": memory_id}

    def list_pending_memories(self, limit: int = 20) -> dict:
        rows = self.pending_collection.get()
        limit = max(1, int(limit or 20))
        items = []
        for idx, memory_id in enumerate(rows["ids"][:limit]):
            item = self._from_metadata(rows["metadatas"][idx])
            items.append({"id": memory_id, "item": item})
        return {"status": "ok", "total": len(rows["ids"]), "items": items}

    def get_memory(self, memory_id: str, include_pending: bool = True) -> dict:
        active = self.collection.get(ids=[memory_id])
        if active["ids"]:
            return {"status": "ok", "memory_id": memory_id, "item": self._from_metadata(active["metadatas"][0]), "pending": False}
        if include_pending:
            pending = self.pending_collection.get(ids=[memory_id])
            if pending["ids"]:
                return {"status": "ok", "memory_id": memory_id, "item": self._from_metadata(pending["metadatas"][0]), "pending": True}
        return {"status": "error", "message": "memory id not found"}

    def search_memory(self, query: str, top_k: int = 5, filters: dict | None = None) -> dict:
        normalized_q = expand_query_text(query)
        top_k = int(top_k or self.config.top_k_default)
        # 先取更大的候选池，再做规则重排以便“问题修复经验”优先召回。
        n_results = max(1, max(top_k * 3, top_k + 5))
        rows = self.collection.get()
        corpus_terms = [set(self._retrieval_tokens(doc)) for doc in rows.get("documents", [])]
        result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        if rows.get("ids"):
            result = self.collection.query(query_texts=[normalized_q], n_results=min(n_results, len(rows["ids"])))
        by_id: dict[str, dict] = {}
        filter_map = filters or self.config.normalized_filters()
        for i, _id in enumerate(result["ids"][0]):
            md = result["metadatas"][0][i]
            item = self._from_metadata(md)
            sim = self._similarity(result["distances"][0][i])
            if filter_map:
                match = True
                for k, v in filter_map.items():
                    field = item.get(k)
                    if isinstance(field, list):
                        if v not in field:
                            match = False
                            break
                    else:
                        if str(field) != str(v):
                            match = False
                            break
                if not match:
                    continue
            corpus_index = rows["ids"].index(_id) if _id in rows.get("ids", []) else 0
            by_id[_id] = {
                "id": _id,
                "similarity": sim,
                "score": self._hybrid_score(normalized_q, item, sim, corpus_terms=corpus_terms, corpus_index=corpus_index),
                "item": item,
            }
        for i, _id in enumerate(rows.get("ids", [])):
            if _id in by_id:
                continue
            item = self._from_metadata(rows["metadatas"][i])
            if filter_map:
                match = True
                for k, v in filter_map.items():
                    field = item.get(k)
                    if isinstance(field, list):
                        if v not in field:
                            match = False
                            break
                    else:
                        if str(field) != str(v):
                            match = False
                            break
                if not match:
                    continue
            lexical = self._bm25_like_score(normalized_q, item, corpus_terms=corpus_terms, index=i)
            entity = self._entity_score(normalized_q, item)
            if lexical <= 0 and entity <= 0:
                continue
            by_id[_id] = {
                "id": _id,
                "similarity": 0.0,
                "score": min(1.0, 0.70 * lexical + 0.20 * entity + self._recency_usage_score(item)),
                "item": item,
            }
        items = list(by_id.values())
        items.sort(
            key=lambda row: (
                row.get("score", row["similarity"]),
                row["similarity"],
                float(row["item"].get("quality_score", 0.0)),
            ),
            reverse=True,
        )
        trimmed = []
        for row in items[:top_k]:
            trimmed.append({"id": row["id"], "similarity": row["similarity"], "score": row.get("score", row["similarity"]), "item": row["item"]})
        self._record_memory_use(trimmed)
        return {"status": "ok", "query": normalized_q, "items": trimmed}

    def _record_memory_use(self, rows: list[dict]):
        if self.config.readonly_mode:
            return
        for row in rows:
            memory_id = row.get("id")
            item = dict(row.get("item") or {})
            if not memory_id or not item:
                continue
            try:
                item["use_count"] = int(item.get("use_count", 0) or 0) + 1
                item["last_used_at"] = _utc_now()
                self.collection.update(
                    ids=[memory_id],
                    documents=[item.get("embedding_text") or make_embedding_text(item)],
                    metadatas=[self._to_metadata(item)],
                )
            except Exception:
                pass

    def update_memory(self, memory_id: str, item: dict, _action: str = "update") -> dict:
        if self.config.readonly_mode:
            return {"status": "skipped", "reason": "readonly_mode"}
        existing = self.collection.get(ids=[memory_id])
        if not existing["ids"]:
            return {"status": "error", "message": "memory id not found"}
        prev_item = self._from_metadata(existing["metadatas"][0])
        normalized = self._normalize_item(item)
        normalized["version"] = int(prev_item.get("version", 1)) + 1
        normalized["created_at"] = prev_item.get("created_at", _utc_now())
        self.collection.update(
            ids=[memory_id],
            documents=[normalized["embedding_text"]],
            metadatas=[self._to_metadata(normalized)],
        )
        self._log_version(_action, memory_id, normalized)
        return {"status": "updated", "memory_id": memory_id, "item": normalized}

    def delete_memory(self, memory_id: str) -> dict:
        if self.config.readonly_mode:
            return {"status": "skipped", "reason": "readonly_mode"}
        self.collection.delete(ids=[memory_id])
        self.pending_collection.delete(ids=[memory_id])
        tombstone = {"status": "deleted", "updated_at": _utc_now()}
        self._log_version("delete", memory_id, tombstone)
        return {"status": "deleted", "memory_id": memory_id}

    def summarize_memory(self, query_results: dict) -> str:
        items = query_results.get("items", [])
        if not items:
            return "无相关历史记忆。"
        lines = ["可复用历史解法："]
        for idx, row in enumerate(items, start=1):
            item = row["item"]
            code_hint = item.get("code_hint", "")
            code_part = f"; 调用提示: {self._short_text(code_hint, max_len=220)}" if code_hint else ""
            lines.append(
                f"{idx}. 问题模式: {item.get('problem_pattern', '')}; "
                f"解法: {item.get('solution_pattern', '')}; "
                f"适用条件: {', '.join(item.get('assumptions', []))}; "
                f"坑点: {', '.join(item.get('failure_modes', []))}"
                f"{code_part}"
            )
        return "\n".join(lines)

    def evaluate_memory_quality(self) -> dict:
        all_rows = self.collection.get()
        total = len(all_rows["ids"])
        if total == 0:
            return {"total": 0, "avg_quality": 0.0}
        scores = []
        for md in all_rows["metadatas"]:
            item = self._from_metadata(md)
            scores.append(float(item.get("quality_score", 0.0)))
        avg = sum(scores) / len(scores)
        return {"total": total, "avg_quality": round(avg, 4)}

    def cluster_similar_memories(self, similarity_threshold: float = 0.88) -> list[list[str]]:
        rows = self.collection.get()
        ids = rows["ids"]
        docs = rows["documents"]
        embs = self.embedding_fn(docs)
        clusters: list[list[str]] = []
        visited = set()
        for i, mem_id in enumerate(ids):
            if mem_id in visited:
                continue
            cur = [mem_id]
            visited.add(mem_id)
            for j in range(i + 1, len(ids)):
                if ids[j] in visited:
                    continue
                sim = sum(a * b for a, b in zip(embs[i], embs[j]))
                if sim >= similarity_threshold:
                    visited.add(ids[j])
                    cur.append(ids[j])
            clusters.append(cur)
        return clusters

    def ingest_dialogue(
        self,
        user_question: str,
        assistant_answer: str,
        require_confirmation: bool = True,
    ) -> dict:
        item = extract_memory_from_dialogue(user_question, assistant_answer)
        return self.add_memory(item=item, require_confirmation=require_confirmation)

    def auto_ingest_last_turn(self, user_question: str, assistant_answer: str) -> dict:
        signature_seed = extract_memory_from_dialogue(user_question, "")
        signature = memory_signature(
            {
                "problem_pattern": signature_seed.get("problem_pattern", user_question),
                "solution_pattern": "atomic_question_memory",
                "math_topic": signature_seed.get("math_topic", []),
                "matrix_properties": signature_seed.get("matrix_properties", []),
            }
        )
        if signature in self._seen_dialogue_signatures:
            return {"status": "skipped", "reason": "already_ingested"}
        self._seen_dialogue_signatures.add(signature)
        return self.ingest_dialogue(
            user_question=user_question,
            assistant_answer=assistant_answer,
            require_confirmation=not self.config.auto_write_mode,
        )

    def ingest_execution_issue(
        self,
        tool_name: str,
        error_text: str,
        fix_text: str = "",
        context_text: str = "",
        require_confirmation: bool = False,
    ) -> dict:
        """
        将“代码执行报错 -> 修复经验”结构化入库，供后续排障召回复用。
        """
        tool = normalize_problem(tool_name or "python_executor") or "python_executor"
        err_short = self._short_text(error_text, max_len=220)
        fix_short = self._short_text(fix_text, max_len=220)
        ctx_short = self._short_text(context_text, max_len=220)
        if not err_short:
            return {"status": "skipped", "reason": "empty_error_text"}

        item = {
            "problem_pattern": f"{tool} 执行报错 {err_short}",
            "math_topic": ["通用数值线性代数", "代码执行"],
            "matrix_properties": [],
            "solution_pattern": "复现报错并定位根因后修复；下次遇到同类报错先复用该修复路径。",
            "method_reason": "将运行期故障经验结构化沉淀，避免重复踩坑并提高修复速度。",
            "failure_modes": [
                err_short,
                "运行环境编码/依赖/参数不匹配导致执行失败",
            ],
            "assumptions": [
                "报错信息可复现且与当前环境相关",
                "修复步骤在同类任务中可迁移",
            ],
            "complexity_hint": "排障复杂度通常由错误定位深度决定。",
            "code_hint": fix_short,
            "source_meta": {
                "source": "runtime_execution",
                "experience_type": "execution_issue",
                "priority": "high",
                "tool_name": tool,
                "error_text": err_short,
                "fix_text": fix_short,
                "context_text": ctx_short,
                "timestamp": _utc_now(),
                "human_confirmed": False,
            },
            "quality_score": 0.78 if fix_short else 0.65,
        }
        return self.add_memory(item=item, require_confirmation=require_confirmation)
