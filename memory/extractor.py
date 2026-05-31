from __future__ import annotations

import re
from datetime import datetime, timezone

from .normalizer import infer_matrix_properties, infer_topics, make_embedding_text, normalize_problem


def _quality_score(item: dict) -> float:
    score = 0.0
    if len(item.get("problem_pattern", "")) >= 12:
        score += 0.2
    if len(item.get("solution_pattern", "")) >= 12:
        score += 0.2
    if item.get("method_reason"):
        score += 0.2
    if item.get("assumptions"):
        score += 0.15
    if item.get("failure_modes"):
        score += 0.15
    if item.get("complexity_hint"):
        score += 0.1
    source_meta = item.get("source_meta", {}) or {}
    if source_meta.get("experience_type") == "issue_resolution":
        # “问题->定位->修复”经验可复用价值更高，适度提高质量分。
        score += 0.1
    return min(1.0, score)


_ISSUE_KEYWORDS = (
    "报错",
    "错误",
    "异常",
    "失败",
    "崩溃",
    "bug",
    "error",
    "exception",
    "traceback",
)
_FIX_KEYWORDS = (
    "修复",
    "解决",
    "已解决",
    "定位",
    "根因",
    "排查",
    "workaround",
    "fix",
    "resolved",
)

_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```", re.IGNORECASE)
_JACOBI_SVD_RE = re.compile(r"(jacobi\s+svd|svd.*jacobi|gejsv|gesvj)", re.IGNORECASE)
_SCIPY_LAPACK_RE = re.compile(
    r"(scipy\.linalg\.lapack|scipy\s+linalg\s+lapack|lapack\.get_lapack_funcs|get_lapack_funcs|gejsv|gesvj)",
    re.IGNORECASE,
)
_SOLUTION_SIGNAL_RE = re.compile(
    r"(建议|推荐|优先|可使用|使用|选择|采用|避免|不要|应当|需要|适合|不适合|"
    r"pcg|cg|gmres|svd|qr|cholesky|lu|lapack|blas|scipy|numpy|正则化|预条件|迭代|直接法)",
    re.IGNORECASE,
)
_ASSUMPTION_SIGNAL_RE = re.compile(r"(前提|假设|条件|要求|当.*时|如果|需满足|适用于|适合)", re.IGNORECASE)
_FAILURE_SIGNAL_RE = re.compile(r"(失败|失效|不稳定|报错|错误|异常|病态|发散|不收敛|放大|避免|不要)", re.IGNORECASE)


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(k.lower() in lower for k in keywords)


def _extract_question_atoms(text: str) -> list[str]:
    """
    将用户问题拆成可复用的最小问题单元（原子子问题）。
    只基于用户问题文本，不使用 assistant_answer 内容。
    """
    raw = (text or "").strip()
    if not raw:
        return []

    split_text = raw
    split_text = re.sub(r"[\n\r\t]+", " ", split_text)
    split_text = re.sub(r"[，,。；;！？?!]+", "|", split_text)
    split_text = re.sub(r"\b(and|then|also|or)\b", "|", split_text, flags=re.IGNORECASE)
    split_text = re.sub(r"(并且|并|以及|然后|再|同时|另外|且)", "|", split_text)

    atoms: list[str] = []
    for part in split_text.split("|"):
        p = normalize_problem(part)
        p = re.sub(r"^(请问|请|帮我|麻烦|我想问|想问下|请教)\s*", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 4:
            continue
        if p not in atoms:
            atoms.append(p)
        if len(atoms) >= 8:
            break

    if atoms:
        return atoms

    fallback = normalize_problem(raw)
    return [fallback] if fallback else []


def _extract_error_clues(text: str) -> list[str]:
    clues: list[str] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        l = raw.lower()
        if any(k in l for k in ("error", "exception", "traceback")):
            clues.append(raw[:160])
            if len(clues) >= 5:
                break
    return clues


def _extract_code_hint(answer: str, max_len: int = 700) -> str:
    snippets: list[str] = []
    for match in _CODE_FENCE_RE.finditer(answer or ""):
        code = match.group(1).strip()
        if code:
            snippets.append(code)
        if len(snippets) >= 2:
            break
    if not snippets:
        for line in (answer or "").splitlines():
            if any(token in line for token in ("get_lapack_funcs", "gejsv", "gesvj", "scipy.linalg.lapack")):
                snippets.append(line.strip())
            if len(snippets) >= 4:
                break
    hint = "\n".join(snippets).strip()
    if len(hint) > max_len:
        hint = hint[: max_len - 3].rstrip() + "..."
    return hint


def _split_answer_sentences(answer: str) -> list[str]:
    raw = re.sub(r"```[\s\S]*?```", " ", answer or "")
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    parts = re.split(r"[。！？!?；;]\s*|\s+-\s+", raw)
    out: list[str] = []
    for part in parts:
        p = re.sub(r"\s+", " ", part).strip(" -:：")
        if 8 <= len(p) <= 220 and p not in out:
            out.append(p)
        if len(out) >= 12:
            break
    return out


def _pick_sentences(sentences: list[str], pattern: re.Pattern, limit: int = 3) -> list[str]:
    picked: list[str] = []
    for sent in sentences:
        if pattern.search(sent):
            picked.append(sent)
        if len(picked) >= limit:
            break
    return picked


def _extract_solution_pattern(answer: str, fallback: str) -> str:
    sentences = _split_answer_sentences(answer)
    picked = _pick_sentences(sentences, _SOLUTION_SIGNAL_RE, limit=3)
    if not picked:
        picked = sentences[:2]
    text = "；".join(picked).strip()
    if not text:
        return fallback
    return normalize_problem(text)


def _extract_assumptions(answer: str, question: str) -> list[str]:
    sentences = _split_answer_sentences(answer)
    picked = _pick_sentences(sentences, _ASSUMPTION_SIGNAL_RE, limit=4)
    props = infer_matrix_properties(f"{question} {answer}")
    assumptions = picked[:]
    if props:
        assumptions.append("已识别矩阵性质: " + "、".join(props))
    if not assumptions:
        assumptions = ["问题满足所选算法的基础前提", "输入矩阵维度与数据类型正确"]
    return sorted(set(normalize_problem(x) for x in assumptions if x))


def _extract_failure_modes(answer: str, is_issue_resolution: bool) -> list[str]:
    sentences = _split_answer_sentences(answer)
    picked = _pick_sentences(sentences, _FAILURE_SIGNAL_RE, limit=4)
    if is_issue_resolution:
        picked.extend(["运行时报错或工具调用失败", "参数/环境不匹配导致执行异常"])
    if not picked:
        picked = ["矩阵性质判断错误导致方法不匹配", "病态问题中不稳定算法可能放大误差"]
    return sorted(set(normalize_problem(x) for x in picked if x))


def _extract_method_reason(answer: str, is_issue_resolution: bool) -> str:
    sentences = _split_answer_sentences(answer)
    reason_re = re.compile(r"(因为|因此|所以|原因|根因|稳定|收敛|复杂度|精度|条件数|误差)", re.IGNORECASE)
    picked = _pick_sentences(sentences, reason_re, limit=2)
    if picked:
        return normalize_problem("；".join(picked))
    if is_issue_resolution:
        return "从报错信号和修复说明中抽取可复用排障经验，后续同类问题优先复用。"
    return "从已回答方案中抽取可复用算法选择依据，后续相似问题优先检索验证。"


def _infer_memory_type(is_issue_resolution: bool, topics: list[str], answer: str) -> str:
    if is_issue_resolution:
        return "issue_resolution"
    lowered = (answer or "").lower()
    if any(tok in lowered for tok in ("get_lapack_funcs", "scipy.linalg.lapack", "gejsv", "gesvj")):
        return "api_usage"
    if any(topic in topics for topic in ("Krylov", "SVD", "QR", "最小二乘", "特征值")):
        return "episodic_solution"
    return "semantic_solution"


def _jacobi_svd_api_memory(user_question: str, assistant_answer: str) -> dict | None:
    combined = f"{user_question}\n{assistant_answer}"
    if not (_JACOBI_SVD_RE.search(combined) and _SCIPY_LAPACK_RE.search(combined)):
        return None

    code_hint = _extract_code_hint(assistant_answer)
    if not code_hint:
        code_hint = (
            "from scipy.linalg import lapack\n"
            "A_f = np.asfortranarray(A, dtype=np.float64)\n"
            "gejsv = lapack.get_lapack_funcs('gejsv', (A_f,))\n"
            "s, u, v, workout, iworkout, info = gejsv(A_f, jobu=1, jobv=1)\n"
            "# info == 0 时，A ≈ u[:, :len(s)] @ np.diag(s) @ v[:, :len(s)].T"
        )

    item = {
        "problem_pattern": (
            "要求使用 Jacobi SVD，并通过 scipy.linalg.lapack 调用 LAPACK driver"
        ),
        "math_topic": ["SVD", "Jacobi SVD", "LAPACK", "SciPy LAPACK"],
        "matrix_properties": infer_matrix_properties(user_question),
        "solution_pattern": (
            "在 SciPy 的 LAPACK 包装中优先查找/调用 gejsv（如 dgejsv/sgejsv），"
            "不要把它误当作 numpy.linalg.svd 或常规 gesvd 的返回约定。"
        ),
        "method_reason": (
            "Jacobi SVD 在 SciPy 中暴露为 LAPACK gejsv 系列包装；返回顺序和参数是常见踩坑点，"
            "再次遇到 Jacobi SVD 时应先复用该调用模板并核对 info。"
        ),
        "failure_modes": [
            "误用 gesvj 名称导致当前 SciPy 找不到 LAPACK 函数",
            "把 gejsv 返回值误解为 numpy.linalg.svd 的 (U, s, Vh)",
            "未使用 Fortran 连续数组或未检查 info 导致结果错误",
        ],
        "assumptions": [
            "当前 SciPy/LAPACK 构建暴露 gejsv 系列函数",
            "输入矩阵可转换为 Fortran 连续的浮点数组",
            "需要左/右奇异向量时设置合适的 jobu/jobv 参数",
        ],
        "complexity_hint": "SVD 总体仍为稠密分解复杂度，适合中小规模稠密矩阵或需高相对精度的场景。",
        "code_hint": code_hint,
        "source_meta": {
            "source": "dialogue",
            "user_question_summary": user_question[:120],
            "memory_granularity": "api_usage",
            "answer_ingested": True,
            "experience_type": "api_usage",
            "priority": "high",
            "api_family": "scipy.linalg.lapack",
            "lapack_driver": "gejsv",
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "human_confirmed": False,
        },
    }
    item["embedding_text"] = make_embedding_text(item)
    item["quality_score"] = max(0.86, _quality_score(item))
    return item


def extract_memory_from_dialogue(user_question: str, assistant_answer: str) -> dict:
    """
    轻量规则抽取器：将问答抽象为结构化记忆条目。
    """
    uq = (user_question or "").strip()
    ans = (assistant_answer or "").strip()
    api_item = _jacobi_svd_api_memory(uq, ans)
    if api_item is not None:
        return api_item

    question_atoms = _extract_question_atoms(uq)
    atom_text = " ".join(question_atoms) if question_atoms else normalize_problem(uq)
    topics = infer_topics(atom_text)
    matrix_props = infer_matrix_properties(atom_text)
    has_issue_signal = _contains_keywords(uq, _ISSUE_KEYWORDS)
    has_fix_signal = _contains_keywords(ans, _FIX_KEYWORDS)
    is_issue_resolution = bool(has_issue_signal and has_fix_signal)

    reason = "记忆条目仅存储问题的最小组成部分，便于跨问题复用与检索。"
    if is_issue_resolution:
        reason = "故障类问题按最小问题单元入库，优先支持后续报错定位与修复检索。"

    item = {
        "problem_pattern": " | ".join(question_atoms) if question_atoms else normalize_problem(uq),
        "math_topic": topics,
        "matrix_properties": matrix_props,
        "solution_pattern": _extract_solution_pattern(ans, "基于原子问题单元进行检索与策略匹配"),
        "method_reason": _extract_method_reason(ans, is_issue_resolution),
        "failure_modes": _extract_failure_modes(ans, is_issue_resolution),
        "assumptions": _extract_assumptions(ans, uq),
        "complexity_hint": "复杂度取决于矩阵规模、稀疏性和目标精度。",
        "code_hint": _extract_code_hint(ans),
        "source_meta": {
            "source": "dialogue",
            "user_question_summary": uq[:120],
            "question_atoms": question_atoms,
            "memory_granularity": "answer_experience",
            "answer_ingested": True,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "human_confirmed": False,
            "experience_type": "issue_resolution" if is_issue_resolution else "general_solution",
            "priority": "high" if is_issue_resolution else "normal",
            "issue_detected": has_issue_signal,
            "fix_detected": has_fix_signal,
            "error_clues": _extract_error_clues(uq),
        },
        "memory_type": _infer_memory_type(is_issue_resolution, topics, ans),
        "namespace": "nla/default",
        "confidence": 0.75 if ans else 0.45,
        "evidence": _split_answer_sentences(ans)[:3],
    }
    if is_issue_resolution:
        item["failure_modes"] = sorted(
            set(item["failure_modes"]) | {"运行时报错或工具调用失败", "参数/环境不匹配导致执行异常"}
        )
        item["assumptions"] = sorted(
            set(item["assumptions"]) | {"修复步骤在当前运行环境可复现", "相关依赖与路径配置一致"}
        )
        if not re.search(r"(修复|解决|fix|resolved)", item["solution_pattern"], flags=re.IGNORECASE):
            item["solution_pattern"] = f"问题定位并修复: {item['solution_pattern']}"
    item["embedding_text"] = make_embedding_text(item)
    item["quality_score"] = _quality_score(item)
    if item["code_hint"]:
        item["quality_score"] = min(1.0, item["quality_score"] + 0.08)
    return item
