from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryConfig:
    persist_dir: str = "memory_db"
    collection_name: str = "nla_memory"
    pending_collection_name: str = "nla_memory_pending"
    embedding_dim: int = 384
    top_k_default: int = 5
    dedup_similarity_threshold: float = 0.9
    merge_similarity_threshold: float = 0.82
    min_quality_score: float = 0.4
    auto_write_mode: bool = False
    readonly_mode: bool = False
    min_recall_score: float = 0.08
    recall_filters_default: dict[str, str] | None = None

    def normalized_filters(self) -> dict[str, str]:
        return dict(self.recall_filters_default or {})


def load_memory_config(config_path: str | Path | None = None) -> MemoryConfig:
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "memory_config.json"
    config_file = Path(config_path)
    if not config_file.exists():
        return MemoryConfig()
    raw = json.loads(config_file.read_text(encoding="utf-8"))
    return MemoryConfig(**raw)
