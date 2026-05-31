from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class MemoryItem:
    problem_pattern: str
    math_topic: list[str]
    matrix_properties: list[str]
    solution_pattern: str
    method_reason: str
    failure_modes: list[str]
    assumptions: list[str]
    complexity_hint: str
    code_hint: str = ""
    source_meta: dict[str, Any] = field(default_factory=dict)
    memory_type: str = "semantic_solution"
    namespace: str = "nla/default"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source_turn_id: str = ""
    embedding_text: str = ""
    quality_score: float = 0.0
    version: int = 1
    status: str = "active"
    merged_from: list[str] = field(default_factory=list)
    last_used_at: str = ""
    use_count: int = 0
    valid_from: str = field(default_factory=_utc_now)
    invalid_at: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_pattern": self.problem_pattern,
            "math_topic": self.math_topic,
            "matrix_properties": self.matrix_properties,
            "solution_pattern": self.solution_pattern,
            "method_reason": self.method_reason,
            "failure_modes": self.failure_modes,
            "assumptions": self.assumptions,
            "complexity_hint": self.complexity_hint,
            "code_hint": self.code_hint,
            "source_meta": self.source_meta,
            "memory_type": self.memory_type,
            "namespace": self.namespace,
            "confidence": float(self.confidence),
            "evidence": list(self.evidence),
            "source_turn_id": self.source_turn_id,
            "embedding_text": self.embedding_text,
            "quality_score": float(self.quality_score),
            "version": int(self.version),
            "status": self.status,
            "merged_from": list(self.merged_from),
            "last_used_at": self.last_used_at,
            "use_count": int(self.use_count),
            "valid_from": self.valid_from,
            "invalid_at": self.invalid_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
