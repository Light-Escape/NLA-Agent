from __future__ import annotations

import hashlib
import re


_MATH_SYMBOL_MAP = {
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "∞": "infinity",
    "∑": "sum",
    "∏": "prod",
    "√": "sqrt",
    "λ": "lambda",
    "σ": "sigma",
    "μ": "mu",
    "∈": "in",
    "∉": "notin",
    "ℓ": "l",
}

_TOPIC_KEYWORDS = {
    "qr": "QR",
    "svd": "SVD",
    "jacobi svd": "Jacobi SVD",
    "jacobi": "Jacobi 方法",
    "gejsv": "Jacobi SVD",
    "gesvj": "Jacobi SVD",
    "lapack": "LAPACK",
    "scipy linalg lapack": "SciPy LAPACK",
    "krylov": "Krylov",
    "precondition": "预条件",
    "预条件": "预条件",
    "low-rank": "低秩近似",
    "低秩": "低秩近似",
    "eigen": "特征值",
    "特征值": "特征值",
    "least squares": "最小二乘",
    "最小二乘": "最小二乘",
    "randomized": "随机化算法",
    "随机化": "随机化算法",
}

_PROP_KEYWORDS = {
    "sparse": "稀疏",
    "稀疏": "稀疏",
    "dense": "稠密",
    "稠密": "稠密",
    "symmetric": "对称",
    "对称": "对称",
    "nonsymmetric": "非对称",
    "非对称": "非对称",
    "spd": "SPD",
    "positive definite": "SPD",
    "低秩": "低秩",
    "low rank": "低秩",
    "ill-conditioned": "病态",
    "病态": "病态",
    "overdetermined": "超定",
    "超定": "超定",
    "underdetermined": "欠定",
    "欠定": "欠定",
}


def normalize_math_text(text: str) -> str:
    t = (text or "").strip()
    for k, v in _MATH_SYMBOL_MAP.items():
        t = t.replace(k, f" {v} ")
    t = t.replace("$$", " ").replace("$", " ")
    t = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[_^{}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def normalize_problem(text: str) -> str:
    t = normalize_math_text(text)
    t = re.sub(r"[，。；：、,.!?()\[\]<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def expand_query_text(text: str) -> str:
    """
    为轻量 hash 检索补充等价 API/算法别名，提升短查询召回率。
    """
    t = normalize_problem(text)
    expansions: list[str] = []
    if "jacobi" in t and "svd" in t:
        expansions.extend(
            [
                "jacobi svd",
                "preconditioned jacobi svd",
                "lapack gejsv",
                "dgejsv sgejsv",
                "scipy linalg lapack get_lapack_funcs gejsv",
            ]
        )
    if "gesvj" in t:
        expansions.extend(["gejsv", "lapack gejsv", "scipy linalg lapack"])
    if "gejsv" in t:
        expansions.extend(["jacobi svd", "preconditioned jacobi svd", "scipy linalg lapack"])
    if "scipy" in t and "lapack" in t and "svd" in t:
        expansions.extend(["gejsv", "lapack svd driver", "get_lapack_funcs"])
    if not expansions:
        return t
    return normalize_problem(" ".join([t, *expansions]))


def infer_topics(text: str) -> list[str]:
    t = normalize_problem(text)
    out: list[str] = []
    for key, topic in _TOPIC_KEYWORDS.items():
        if key in t and topic not in out:
            out.append(topic)
    return out or ["通用数值线性代数"]


def infer_matrix_properties(text: str) -> list[str]:
    t = normalize_problem(text)
    out: list[str] = []
    for key, prop in _PROP_KEYWORDS.items():
        if key in t and prop not in out:
            out.append(prop)
    return out


def make_embedding_text(item: dict) -> str:
    fields = [
        item.get("problem_pattern", ""),
        " ".join(item.get("math_topic", []) or []),
        " ".join(item.get("matrix_properties", []) or []),
        item.get("solution_pattern", ""),
        item.get("method_reason", ""),
        " ".join(item.get("assumptions", []) or []),
        " ".join(item.get("failure_modes", []) or []),
        item.get("complexity_hint", ""),
        item.get("code_hint", ""),
    ]
    joined = " | ".join(str(x).strip() for x in fields if str(x).strip())
    return normalize_problem(joined)


def memory_signature(item: dict) -> str:
    signature_base = "||".join(
        [
            normalize_problem(item.get("problem_pattern", "")),
            normalize_problem(item.get("solution_pattern", "")),
            ",".join(sorted(item.get("math_topic", []) or [])),
            ",".join(sorted(item.get("matrix_properties", []) or [])),
        ]
    )
    return hashlib.sha256(signature_base.encode("utf-8")).hexdigest()[:24]
