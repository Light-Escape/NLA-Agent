from __future__ import annotations

import hashlib
import math
from typing import Any


class HashEmbeddingFunction:
    """
    轻量本地向量化：
    - 无需下载模型，便于离线与测试
    - 通过 token hashing 构建固定维度向量
    """

    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def _token_to_bucket(self, token: str) -> int:
        h = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(h[:16], 16) % self.dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [tok for tok in (text or "").split() if tok]
        if not tokens:
            return vec
        for tok in tokens:
            vec[self._token_to_bucket(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    @staticmethod
    def name() -> str:
        return "nla_hash_embedding"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(dim=int(config.get("dim", 384)))

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        return

    def is_legacy(self) -> bool:
        return False
