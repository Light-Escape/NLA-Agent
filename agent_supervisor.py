"""
数值线性代数步骤监督模块（agent_supervisor）。

职责：
- 只做审核与纠错建议，不直接求解问题。
- 对主 agent 的每一步计划、工具调用和工具结果给出结构化判定。
"""

from __future__ import annotations

from typing import Any, Optional


_ALLOWED_VERDICTS = {"approve", "revise", "reject"}
_ALLOWED_STEP_STATUS = {"ok", "warning", "error"}
_ALLOWED_NEXT_ACTION = {"continue", "retry", "ask_user", "stop"}
_ALLOWED_PHASE = {
    "unknown",
    "clarify_goal",
    "preconditions",
    "property_check",
    "plan",
    "solve",
}
_PHASE_ORDER = ["clarify_goal", "preconditions", "property_check", "plan", "solve"]
_TOOL_MIN_PHASE = {
    "route_user_task": "clarify_goal",
    "infer_solution_preference": "clarify_goal",
    "build_precondition_checklist": "preconditions",
    "analyze_matrix_properties": "property_check",
    "plan_coach_next_step": "plan",
    "choose_nla_algorithm": "plan",
    "search_numpy_scipy_docs": "plan",
    "get_linalg_backend_info": "solve",
    "call_lapack": "solve",
    "call_blas": "solve",
    "solve_linear_lapack": "solve",
    "least_squares_lapack": "solve",
    "gemm_blas": "solve",
    "run_python_snippet": "solve",
}
_VERIFIED_FACTS_SCHEMA_VERSION = "v1"
_VERIFIED_FACT_KEYS = (
    "goal_clarified",
    "preference_confirmed",
    "preconditions_checked",
    "properties_checked",
    "algorithm_planned",
)

_ALLOWED_ISSUE_TYPES = {
    "task_understanding",
    "algorithm_choice",
    "tool_arguments",
    "shape_check",
    "numerical_sanity",
    "stability",
    "safety",
    "format",
}
_ALLOWED_SEVERITY = {"low", "medium", "high"}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_problem_type(problem_type: str) -> str:
    p = (problem_type or "").strip().lower()
    alias = {
        "solve": "solve_linear_system",
        "linear_solve": "solve_linear_system",
        "least_squares": "least_squares",
        "ls": "least_squares",
        "eigenvalue": "eigen",
        "eigenvalues": "eigen",
    }
    return alias.get(p, p)


def _safe_shape(matrix_info: Optional[dict]) -> Optional[tuple[int, int]]:
    if not isinstance(matrix_info, dict):
        return None
    shape = matrix_info.get("shape")
    if (
        isinstance(shape, (list, tuple))
        and len(shape) == 2
        and all(isinstance(x, int) and x >= 0 for x in shape)
    ):
        return int(shape[0]), int(shape[1])
    return None


def _new_issue(
    issue_type: str,
    severity: str,
    message: str,
    suggestion: str,
    confidence: Optional[float] = None,
    evidence: str = "",
) -> dict:
    t = issue_type if issue_type in _ALLOWED_ISSUE_TYPES else "format"
    s = severity if severity in _ALLOWED_SEVERITY else "medium"
    conf = confidence
    if not isinstance(conf, (int, float)):
        conf = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(s, 0.7)
    conf = min(max(float(conf), 0.0), 1.0)
    return {
        "type": t,
        "severity": s,
        "message": _to_text(message) or "检测到问题。",
        "suggestion": _to_text(suggestion) or "请补充信息并修正后重试。",
        "confidence": conf,
        "evidence": _to_text(evidence),
    }


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def _normalize_phase(phase: str) -> str:
    p = _to_text(phase).lower()
    alias = {
        "clarify": "clarify_goal",
        "goal": "clarify_goal",
        "precondition": "preconditions",
        "property": "property_check",
        "planning": "plan",
        "execution": "solve",
    }
    p = alias.get(p, p)
    if p not in _ALLOWED_PHASE:
        return "unknown"
    return p


def _phase_rank(phase: str) -> int:
    p = _normalize_phase(phase)
    if p == "unknown":
        return -1
    try:
        return _PHASE_ORDER.index(p)
    except ValueError:
        return -1


def _normalize_verified_facts(verified_facts: Optional[dict]) -> dict:
    """
    将 verified_facts 归一为稳定 schema（v1），并兼容旧字段别名。
    """
    src = verified_facts if isinstance(verified_facts, dict) else {}
    normalized: dict[str, Any] = {"schema_version": _VERIFIED_FACTS_SCHEMA_VERSION}

    normalized["goal_clarified"] = bool(
        src.get("goal_clarified") or src.get("goal_confirmed") or src.get("clarify_goal_done")
    )
    normalized["preference_confirmed"] = bool(
        src.get("preference_confirmed")
        or src.get("solution_preference_confirmed")
        or src.get("solution_goal_confirmed")
    )
    normalized["preconditions_checked"] = bool(
        src.get("preconditions_checked")
        or src.get("precondition_checklist_ready")
        or src.get("preconditions_passed")
    )
    normalized["properties_checked"] = bool(
        src.get("properties_checked")
        or src.get("matrix_properties_checked")
        or src.get("matrix_info_confirmed")
    )
    normalized["algorithm_planned"] = bool(
        src.get("algorithm_planned") or src.get("algorithm_selected") or src.get("plan_ready")
    )

    completed: list[str] = []
    for key in ("completed_phases", "phase_history"):
        arr = src.get(key)
        if isinstance(arr, (list, tuple)):
            for item in arr:
                p = _normalize_phase(_to_text(item))
                if p != "unknown" and p not in completed:
                    completed.append(p)

    for key in ("last_phase", "current_phase"):
        p = _normalize_phase(_to_text(src.get(key)))
        if p != "unknown" and p not in completed:
            completed.append(p)

    # 用布尔事实补齐 completed_phases
    if normalized["goal_clarified"] or normalized["preference_confirmed"]:
        if "clarify_goal" not in completed:
            completed.append("clarify_goal")
    if normalized["preconditions_checked"] and "preconditions" not in completed:
        completed.append("preconditions")
    if normalized["properties_checked"] and "property_check" not in completed:
        completed.append("property_check")
    if normalized["algorithm_planned"] and "plan" not in completed:
        completed.append("plan")

    normalized["completed_phases"] = completed
    normalized["_has_structured_facts"] = any(key in src for key in _VERIFIED_FACT_KEYS)
    return normalized


def _has_explicit_fact(verified_facts: dict, key: str) -> bool:
    return isinstance(verified_facts, dict) and key in verified_facts


def _extract_completed_phases(merged_info: dict) -> set[str]:
    done: set[str] = set()
    if not isinstance(merged_info, dict):
        return done

    # 优先使用归一后的 completed_phases；同时兼容 phase_history。
    for key in ("completed_phases", "phase_history"):
        arr = merged_info.get(key)
        if isinstance(arr, (list, tuple)):
            for item in arr:
                p = _normalize_phase(_to_text(item))
                if p != "unknown":
                    done.add(p)

    # 兼容“上一步阶段”标记
    for key in ("last_phase", "current_phase"):
        p = _normalize_phase(_to_text(merged_info.get(key)))
        if p != "unknown":
            done.add(p)

    # 兼容布尔事实标记
    if bool(merged_info.get("goal_clarified")) or bool(merged_info.get("preference_confirmed")):
        done.add("clarify_goal")
    if bool(merged_info.get("preconditions_checked")) or bool(
        merged_info.get("precondition_checklist_ready")
    ) or bool(merged_info.get("preconditions_passed")):
        done.add("preconditions")
    if bool(merged_info.get("matrix_properties_checked")) or bool(
        merged_info.get("properties_checked")
    ) or bool(merged_info.get("matrix_info_confirmed")):
        done.add("property_check")
    if bool(merged_info.get("algorithm_planned")) or bool(merged_info.get("algorithm_selected")) or bool(
        merged_info.get("plan_ready")
    ):
        done.add("plan")

    return done


def _phase_allows(phase: str, check_name: str) -> bool:
    phase_checks = {
        "unknown": {
            "task_alignment",
            "algorithm_stability",
            "math_simplification",
            "interactive_gates",
            "tool_args",
            "tool_result",
        },
        "clarify_goal": {"task_alignment"},
        "preconditions": {"task_alignment", "interactive_gates"},
        "property_check": {"interactive_gates", "tool_args"},
        "plan": {"task_alignment", "math_simplification", "interactive_gates", "tool_args"},
        "solve": {
            "task_alignment",
            "algorithm_stability",
            "math_simplification",
            "interactive_gates",
            "tool_args",
            "tool_result",
        },
    }
    checks = phase_checks.get(_normalize_phase(phase), phase_checks["unknown"])
    return check_name in checks


def _check_phase_sequence(
    issues: list[dict],
    phase: str,
    merged_info: dict,
    tool_call: Optional[dict],
) -> None:
    phase_norm = _normalize_phase(phase)
    if phase_norm == "unknown":
        issues.append(
            _new_issue(
                "format",
                "medium",
                "未提供有效 phase，无法进行严格顺序控制。",
                "显式传入 phase（clarify_goal/preconditions/property_check/plan/solve）。",
            )
        )
        return

    completed = _extract_completed_phases(merged_info)
    allow_phase_skip = bool(
        merged_info.get("allow_phase_skip")
        or merged_info.get("fast_track")
        or merged_info.get("direct_mode")
    )
    current_idx = _phase_rank(phase_norm)
    if current_idx > 0:
        for prev_phase in _PHASE_ORDER[:current_idx]:
            if prev_phase not in completed:
                if allow_phase_skip:
                    continue
                issues.append(
                    _new_issue(
                        "task_understanding",
                        "high",
                        f"检测到阶段跳步：当前 phase={phase_norm}，缺少前序阶段 {prev_phase} 的完成证据。",
                        "若信息已充分，可在 verified_facts 中标记 allow_phase_skip=true 或补充 completed_phases 证据。",
                    )
                )
                break

    # 工具级闸门：限制关键工具最早调用阶段
    tool_name = _to_text((tool_call or {}).get("name")).lower()
    if tool_name:
        min_phase = _TOOL_MIN_PHASE.get(tool_name)
        if min_phase and _phase_rank(phase_norm) < _phase_rank(min_phase):
            # 在已有充分证据时，允许前置调用 plan/solve 相关工具，减少机械等待。
            has_preconditions = "preconditions" in completed or bool(merged_info.get("preconditions_checked"))
            has_properties = "property_check" in completed or bool(merged_info.get("properties_checked"))
            has_plan = "plan" in completed or bool(merged_info.get("algorithm_planned"))
            can_early_call = False
            if tool_name in {"choose_nla_algorithm", "search_numpy_scipy_docs"}:
                can_early_call = has_preconditions
            elif tool_name in {
                "call_lapack",
                "call_blas",
                "solve_linear_lapack",
                "least_squares_lapack",
                "gemm_blas",
                "run_python_snippet",
            }:
                can_early_call = has_preconditions and (has_properties or has_plan)
            elif tool_name == "get_linalg_backend_info":
                can_early_call = has_preconditions
            if can_early_call or allow_phase_skip:
                return
            issues.append(
                _new_issue(
                    "tool_arguments",
                    "high",
                    f"工具 {tool_name} 在 phase={phase_norm} 调用过早。",
                    f"建议在 {min_phase} 阶段后调用，或先补充 verified_facts 证明前置条件已满足。",
                )
            )


def _issue_score(issue: dict) -> float:
    severity_score = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(
        _to_text(issue.get("severity")).lower(), 1.0
    )
    confidence = issue.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.6
    return severity_score * (0.5 + 0.5 * min(max(float(confidence), 0.0), 1.0))


def _resolve_conflicts(issues: list[dict], matrix_info: Optional[dict]) -> list[dict]:
    # 先做精确去重，避免重复提醒压低有效信息密度。
    dedup: dict[tuple[str, str, str, str], dict] = {}
    for issue in issues:
        key = (
            _to_text(issue.get("type")),
            _to_text(issue.get("severity")),
            _to_text(issue.get("message")),
            _to_text(issue.get("suggestion")),
        )
        prev = dedup.get(key)
        if prev is None or _issue_score(issue) > _issue_score(prev):
            dedup[key] = issue
    merged = list(dedup.values())

    # 对“特征值算法建议”的互斥提示做冲突消解。
    prefer_sym = None
    if isinstance(matrix_info, dict):
        if isinstance(matrix_info.get("symmetric"), bool):
            prefer_sym = bool(matrix_info.get("symmetric"))
        elif isinstance(matrix_info.get("is_symmetric"), bool):
            prefer_sym = bool(matrix_info.get("is_symmetric"))

    sym_issues: list[dict] = []
    nonsym_issues: list[dict] = []
    others: list[dict] = []
    for issue in merged:
        suggestion = _to_text(issue.get("suggestion")).lower()
        msg = _to_text(issue.get("message")).lower()
        blob = f"{msg} | {suggestion}"
        if "eigh" in blob or "eigsh" in blob:
            sym_issues.append(issue)
        elif "eig/eigs" in blob or "eig(" in blob:
            nonsym_issues.append(issue)
        else:
            others.append(issue)

    if sym_issues and nonsym_issues:
        if prefer_sym is True:
            others.extend(sym_issues)
        elif prefer_sym is False:
            others.extend(nonsym_issues)
        else:
            # 性质不明时保留更高分的一侧，避免自相矛盾。
            sym_best = max(sym_issues, key=_issue_score)
            nonsym_best = max(nonsym_issues, key=_issue_score)
            others.append(sym_best if _issue_score(sym_best) >= _issue_score(nonsym_best) else nonsym_best)
        return others

    return merged


def _rank_and_trim_issues(issues: list[dict], top_k: int = 3) -> list[dict]:
    ranked = sorted(issues, key=_issue_score, reverse=True)
    if top_k <= 0:
        return ranked
    return ranked[:top_k]


def _collect_context_text(
    user_goal: str,
    current_step: str,
    proposed_action: str,
    tool_call: Optional[dict],
) -> str:
    tool_name = ""
    if isinstance(tool_call, dict):
        tool_name = _to_text(tool_call.get("name"))
    return " | ".join(
        part for part in [user_goal, current_step, proposed_action, tool_name] if part
    ).lower()


def _check_input_completeness(
    issues: list[dict],
    user_goal: str,
    problem_type: str,
    current_step: str,
    proposed_action: str,
) -> None:
    if not user_goal:
        issues.append(
            _new_issue(
                "task_understanding",
                "high",
                "缺少 user_goal，无法判断当前步骤是否对齐用户目标。",
                "补充 user_goal 后再审核该步骤。",
            )
        )
    if not problem_type:
        issues.append(
            _new_issue(
                "format",
                "high",
                "缺少 problem_type，无法执行任务类型特定校验。",
                "补充 problem_type（如 solve_linear_system / least_squares / eigen）。",
            )
        )
    if not current_step:
        issues.append(
            _new_issue(
                "format",
                "medium",
                "缺少 current_step，无法判断步骤语义。",
                "补充当前步骤说明。",
            )
        )
    if not proposed_action:
        issues.append(
            _new_issue(
                "format",
                "high",
                "缺少 proposed_action，无法审核动作合理性。",
                "补充主 agent 计划动作。",
            )
        )


def _check_task_alignment(issues: list[dict], user_goal: str, problem_type: str) -> None:
    goal = user_goal.lower()
    p = _normalize_problem_type(problem_type)
    if any(k in goal for k in ("最小二乘", "least squares", "拟合")) and p != "least_squares":
        issues.append(
            _new_issue(
                "task_understanding",
                "medium",
                "用户目标偏向最小二乘，但 problem_type 不是 least_squares。",
                "修正 problem_type 为 least_squares，或在 current_step 说明为何不是该任务。",
            )
        )
    if any(k in goal for k in ("特征值", "eigen", "特征向量")) and p != "eigen":
        issues.append(
            _new_issue(
                "task_understanding",
                "medium",
                "用户目标偏向特征值问题，但 problem_type 不是 eigen。",
                "修正 problem_type 为 eigen，或明确说明当前步骤是前置处理。",
            )
        )
    if any(k in goal for k in ("线性方程", "ax=b", "求解")) and p not in {
        "solve_linear_system",
        "least_squares",
    }:
        issues.append(
            _new_issue(
                "task_understanding",
                "medium",
                "用户目标偏向线性系统求解，但 problem_type 不匹配。",
                "将 problem_type 修正为 solve_linear_system（或 least_squares）。",
            )
        )


def _check_safety(issues: list[dict], context_text: str) -> None:
    if _contains_any(
        context_text,
        ["rm -rf", "os.system", "subprocess", "curl ", "wget ", "powershell -enc"],
    ):
        issues.append(
            _new_issue(
                "safety",
                "high",
                "检测到潜在危险执行行为，超出数值线性代数任务范围。",
                "停止该步骤，改为仅使用受限数值计算工具。",
            )
        )


def _check_algorithm_and_stability(
    issues: list[dict],
    problem_type: str,
    matrix_info: Optional[dict],
    context_text: str,
    tool_call: Optional[dict],
) -> None:
    p = _normalize_problem_type(problem_type)
    shape = _safe_shape(matrix_info)
    tool_name = _to_text((tool_call or {}).get("name")).lower()
    tool_args = (tool_call or {}).get("args", {})
    args_text = str(tool_args).lower()
    full_text = f"{context_text} | {tool_name} | {args_text}"

    uses_explicit_inverse = _contains_any(
        full_text, ["linalg.inv", "np.linalg.inv", "scipy.linalg.inv", "求逆后再解"]
    )
    if p in {"solve_linear_system", "least_squares"} and uses_explicit_inverse:
        issues.append(
            _new_issue(
                "stability",
                "high",
                "检测到显式求逆用于解方程/最小二乘，数值稳定性较差。",
                "改用 solve / lstsq / QR / SVD / 稀疏迭代法。",
            )
        )

    if p == "least_squares" and not _contains_any(
        full_text, ["lstsq", "qr", "svd", "least squares"]
    ):
        issues.append(
            _new_issue(
                "algorithm_choice",
                "medium",
                "最小二乘任务未体现 QR/SVD/lstsq 路线。",
                "优先使用 numpy.linalg.lstsq 或 QR/SVD 分解。",
            )
        )

    if p == "eigen":
        is_square = bool(matrix_info.get("shape", [0, 1])[0] == matrix_info.get("shape", [1, 0])[1]) if isinstance(matrix_info, dict) and isinstance(matrix_info.get("shape"), list) and len(matrix_info.get("shape")) == 2 else None
        symmetric = bool(matrix_info.get("symmetric", matrix_info.get("is_symmetric", False))) if isinstance(matrix_info, dict) else False
        if is_square is False:
            issues.append(
                _new_issue(
                    "shape_check",
                    "high",
                    "特征值问题要求方阵，但 matrix_info.shape 显示非方阵。",
                    "先确认任务是否应改为 SVD 或最小二乘相关问题。",
                )
            )
        if symmetric and _contains_any(full_text, ["eig(", "eigs("]) and not _contains_any(
            full_text, ["eigh", "eigsh"]
        ):
            issues.append(
                _new_issue(
                    "algorithm_choice",
                    "medium",
                    "对称矩阵特征值问题未优先使用 eigh/eigsh。",
                    "稠密对称用 eigh，稀疏对称用 eigsh。",
                )
            )
        if (not symmetric) and _contains_any(full_text, ["eigh", "eigsh"]) and not _contains_any(
            full_text, ["广义", "hermitian"]
        ):
            issues.append(
                _new_issue(
                    "algorithm_choice",
                    "high",
                    "非对称问题疑似使用了仅适配对称/Hermitian 的特征值算法。",
                    "改用 eig/eigs，或先证明矩阵满足对称/Hermitian 条件。",
                )
            )

    is_sparse = bool(matrix_info.get("sparse", matrix_info.get("is_sparse_input", False))) if isinstance(matrix_info, dict) else False
    if is_sparse and _contains_any(full_text, ["toarray()", "todense()", "np.array(a)", "dense"]) and shape:
        m, n = shape
        if m * n > 200_000:
            issues.append(
                _new_issue(
                    "stability",
                    "high",
                    "稀疏矩阵疑似被转为稠密，规模较大可能导致内存与性能问题。",
                    "保持稀疏格式，使用 spsolve/cg/gmres/eigs/eigsh 等稀疏算法。",
                )
            )
        else:
            issues.append(
                _new_issue(
                    "algorithm_choice",
                    "medium",
                    "稀疏矩阵步骤中出现转稠密操作，可能不必要。",
                    "优先保持 CSC/CSR 并调用稀疏线性代数 API。",
                )
            )


def _check_math_simplification_first(
    issues: list[dict],
    problem_type: str,
    current_step: str,
    proposed_action: str,
    tool_call: Optional[dict],
) -> None:
    """
    在进入数值算法前，要求先评估是否可做数学化简。
    """
    p = _normalize_problem_type(problem_type)
    # 对这些典型 NLA 任务强制执行“先化简后数值”检查
    if p not in {"solve_linear_system", "least_squares", "eigen", "determinant", "inverse"}:
        return

    step_text = f"{current_step} | {proposed_action}".lower()
    tool_name = _to_text((tool_call or {}).get("name")).lower()
    tool_args = str((tool_call or {}).get("args", {})).lower()
    full_text = f"{step_text} | {tool_name} | {tool_args}"

    mentioned_simplify = _contains_any(
        full_text,
        [
            "化简",
            "解析",
            "symbolic",
            "elimination",
            "消元",
            "代数",
            "降维",
            "分块",
            "schur",
            "变量替换",
            "结构利用",
            "对称性",
            "正交",
        ],
    )
    jumps_to_numeric = _contains_any(
        full_text,
        [
            "numpy.linalg",
            "scipy.linalg",
            "solve_linear_lapack",
            "least_squares_lapack",
            "gemm_blas",
            "get_linalg_backend_info",
            "spsolve",
            "cg",
            "gmres",
            "lstsq",
            "eig(",
            "eigh",
            "eigs",
            "eigsh",
            "run_python_snippet",
            "choose_nla_algorithm",
        ],
    )

    if jumps_to_numeric and not mentioned_simplify:
        issues.append(
            _new_issue(
                "algorithm_choice",
                "medium",
                "检测到步骤可能直接进入数值代数算法，缺少“先做数学化简可行性评估”。",
                "先补充化简分析（结构性质、代数变形、消元/降维/分块等），确认不足后再进入数值算法。",
            )
        )


def _check_interactive_coach_gates(
    issues: list[dict],
    problem_type: str,
    matrix_info: Optional[dict],
    verified_facts: Optional[dict],
    current_step: str,
    proposed_action: str,
    tool_call: Optional[dict],
) -> None:
    """
    教练模式流程闸门：
    先确认精确/近似 -> 先过必要条件 -> 先确认矩阵性质 -> 再进算法。
    """
    p = _normalize_problem_type(problem_type)
    if p not in {"solve_linear_system", "least_squares", "eigen", "determinant", "inverse"}:
        return

    tool_name = _to_text((tool_call or {}).get("name")).lower()
    tool_args = str((tool_call or {}).get("args", {})).lower()
    text = f"{current_step} | {proposed_action} | {tool_name} | {tool_args}".lower()
    if _contains_any(text, ["直接给答案", "只要结果", "快速求解", "尽快", "马上", "direct"]):
        # 用户明确要求快速直给时，不做重型交互闸门，避免拖慢节奏。
        return

    jumps_to_algorithm = _contains_any(
        text,
        [
            "choose_nla_algorithm",
            "solve_linear_lapack",
            "least_squares_lapack",
            "gemm_blas",
            "get_linalg_backend_info",
            "run_python_snippet",
            "numpy.linalg",
            "scipy.linalg",
            "spsolve",
            "lstsq",
            "cg",
            "gmres",
            "eig(",
            "eigh",
            "eigs",
            "eigsh",
        ],
    )
    if not jumps_to_algorithm:
        return

    vf = verified_facts if isinstance(verified_facts, dict) else {}
    has_explicit_preference = _has_explicit_fact(vf, "preference_confirmed")
    has_explicit_preconditions = _has_explicit_fact(vf, "preconditions_checked")
    has_explicit_properties = _has_explicit_fact(vf, "properties_checked")

    mentioned_solution_goal = _contains_any(
        text, ["精确", "近似", "exact", "approx", "误差", "容忍度", "tolerance"]
    )
    preference_ok = bool(vf.get("preference_confirmed")) if has_explicit_preference else mentioned_solution_goal
    if not preference_ok:
        issues.append(
            _new_issue(
                "task_understanding",
                "medium",
                "进入算法前缺少“精确解/近似解”偏好确认。",
                "先向用户确认解目标（精确或近似）后，再选择算法。",
            )
        )

    mentioned_preconditions = _contains_any(
        text,
        ["必要条件", "前提", "维度匹配", "方阵", "秩", "可逆", "条件数", "checklist"],
    )
    preconditions_ok = bool(vf.get("preconditions_checked")) if has_explicit_preconditions else mentioned_preconditions
    if not preconditions_ok:
        issues.append(
            _new_issue(
                "algorithm_choice",
                "medium",
                "进入算法前缺少必要条件确认。",
                "先列出并确认任务必要条件（维度、方阵性、秩、条件数等）。",
            )
        )

    info = matrix_info if isinstance(matrix_info, dict) else {}
    has_property_evidence = bool(
        isinstance(info.get("is_square"), bool)
        or isinstance(info.get("is_symmetric"), bool)
        or isinstance(info.get("rank_estimate"), int)
        or isinstance(info.get("condition_number_2"), (int, float))
    )
    mentioned_property_check = _contains_any(
        text,
        [
            "matrix_properties",
            "analyze_matrix_properties",
            "矩阵性质",
            "对称",
            "正定",
            "稀疏",
            "秩",
            "条件数",
        ],
    )
    properties_ok = bool(vf.get("properties_checked")) if has_explicit_properties else (
        has_property_evidence or mentioned_property_check
    )
    if not properties_ok:
        issues.append(
            _new_issue(
                "task_understanding",
                "medium",
                "进入算法前缺少矩阵关键性质确认。",
                "先确认至少一组关键性质（方阵/秩/对称性/条件数等）后再继续。",
            )
        )


def _check_tool_arguments(
    issues: list[dict],
    problem_type: str,
    matrix_info: Optional[dict],
    tool_call: Optional[dict],
) -> None:
    if tool_call is None:
        return
    if not isinstance(tool_call, dict):
        issues.append(
            _new_issue(
                "format",
                "high",
                "tool_call 必须是对象（dict）。",
                "按 {\"name\": ..., \"args\": {...}} 提供工具调用信息。",
            )
        )
        return

    name = _to_text(tool_call.get("name")).lower()
    args = tool_call.get("args")
    if not name:
        issues.append(
            _new_issue(
                "tool_arguments",
                "high",
                "tool_call 缺少工具名 name。",
                "补充工具名并重试。",
            )
        )
    if args is None or not isinstance(args, dict):
        issues.append(
            _new_issue(
                "tool_arguments",
                "high",
                "tool_call.args 缺失或格式错误。",
                "补充参数对象 args，并确保是 key-value 结构。",
            )
        )
        return

    p = _normalize_problem_type(problem_type)
    shape = _safe_shape(matrix_info)

    if "eigsh" in name or "eigs" in name:
        if "k" not in args:
            issues.append(
                _new_issue(
                    "tool_arguments",
                    "medium",
                    "部分特征值算法缺少 k 参数。",
                    "显式传入 k，并保证 1 <= k < n。",
                )
            )
        elif shape and isinstance(args.get("k"), int):
            n = shape[1]
            k = int(args["k"])
            if k < 1 or k >= n:
                issues.append(
                    _new_issue(
                        "tool_arguments",
                        "high",
                        "k 参数超出合法范围，可能导致求解失败。",
                        "将 k 调整到 1 <= k < n。",
                    )
                )

    if p in {"solve_linear_system", "least_squares"} and shape:
        m, n = shape
        if "b" in args:
            b = args.get("b")
            if isinstance(b, list):
                if len(b) != m:
                    issues.append(
                        _new_issue(
                            "shape_check",
                            "high",
                            "右端项 b 维度与 A 的行数不一致。",
                            f"将 b 长度改为 {m}，或检查 A/b 输入来源。",
                        )
                    )
            elif isinstance(b, dict) and "shape" in b:
                bshape = b.get("shape")
                if isinstance(bshape, (list, tuple)) and len(bshape) >= 1:
                    if int(bshape[0]) != m:
                        issues.append(
                            _new_issue(
                                "shape_check",
                                "high",
                                "b.shape[0] 与 A.shape[0] 不一致。",
                                f"修正 b.shape[0] 为 {m}。",
                            )
                        )
        if p == "solve_linear_system" and m != n:
            issues.append(
                _new_issue(
                    "algorithm_choice",
                    "high",
                    "当前标记为线性方程组直接求解，但 A 非方阵。",
                    "改为 least_squares 或使用约束求解策略。",
                )
            )


def _check_tool_result(
    issues: list[dict],
    tool_result: Optional[dict],
    matrix_info: Optional[dict],
) -> None:
    if tool_result is None:
        return
    if not isinstance(tool_result, dict):
        issues.append(
            _new_issue(
                "format",
                "medium",
                "tool_result 不是对象，无法进行结果审查。",
                "将 tool_result 规范为 dict，并包含关键数值指标。",
            )
        )
        return

    status = _to_text(tool_result.get("status")).lower()
    if status in {"error", "failed", "fail"}:
        issues.append(
            _new_issue(
                "numerical_sanity",
                "high",
                "工具返回失败状态，当前结果不可直接采信。",
                "分析报错并修正参数/算法后重试。",
            )
        )

    converged = tool_result.get("converged")
    if converged is False:
        issues.append(
            _new_issue(
                "numerical_sanity",
                "high",
                "迭代法未收敛，结果可能不可信。",
                "调整预条件器、迭代上限、容忍度或更换算法。",
            )
        )

    for key in ("residual_norm", "relative_residual", "residual"):
        val = tool_result.get(key)
        if isinstance(val, (int, float)):
            if val > 1e-2:
                issues.append(
                    _new_issue(
                        "numerical_sanity",
                        "high",
                        f"{key}={val:.3e} 偏大，结果可靠性不足。",
                        "检查模型/维度/条件数并重跑；必要时更换稳定算法。",
                    )
                )
            elif val > 1e-4:
                issues.append(
                    _new_issue(
                        "numerical_sanity",
                        "medium",
                        f"{key}={val:.3e} 偏高，需要额外核验。",
                        "补充残差与后验误差检查。",
                    )
                )
            break

    cond = None
    if isinstance(tool_result.get("condition_number"), (int, float)):
        cond = float(tool_result["condition_number"])
    elif isinstance(matrix_info, dict) and isinstance(
        matrix_info.get("condition_number"), (int, float)
    ):
        cond = float(matrix_info["condition_number"])
    elif isinstance(matrix_info, dict) and isinstance(
        matrix_info.get("condition_number_2"), (int, float)
    ):
        cond = float(matrix_info["condition_number_2"])
    if cond is not None and cond > 1e12:
        issues.append(
            _new_issue(
                "stability",
                "high",
                f"条件数较大（{cond:.3e}），问题病态，数值敏感。",
                "采用正则化/缩放/更稳定分解，并报告不确定性。",
            )
        )


def _finalize(issues: list[dict]) -> dict:
    has_high = any(i.get("severity") == "high" for i in issues)
    has_medium = any(i.get("severity") == "medium" for i in issues)
    has_safety_high = any(
        i.get("severity") == "high" and i.get("type") == "safety" for i in issues
    )

    if not issues:
        result = {
            "verdict": "approve",
            "step_status": "ok",
            "issues": [],
            "next_action": "continue",
            "short_reason": "当前步骤合理，可继续执行下一步。",
        }
    elif has_high:
        result = {
            "verdict": "reject",
            "step_status": "error",
            "issues": issues,
            "next_action": "stop" if has_safety_high else "retry",
            "short_reason": "存在高风险或明显错误，需停止并重做当前步骤。",
        }
    elif has_medium:
        result = {
            "verdict": "revise",
            "step_status": "warning",
            "issues": issues,
            "next_action": "retry",
            "short_reason": "步骤方向基本正确，但需按建议修正后再继续。",
        }
    else:
        result = {
            "verdict": "revise",
            "step_status": "warning",
            "issues": issues,
            "next_action": "continue",
            "short_reason": "存在轻微问题，建议修正并继续。",
        }

    # 二次保险：保证字段枚举合法，输出可机器处理。
    if result["verdict"] not in _ALLOWED_VERDICTS:
        result["verdict"] = "revise"
    if result["step_status"] not in _ALLOWED_STEP_STATUS:
        result["step_status"] = "warning"
    if result["next_action"] not in _ALLOWED_NEXT_ACTION:
        result["next_action"] = "retry"
    return result


def supervise_step(
    user_goal: str,
    problem_type: str,
    matrix_info: Optional[dict] = None,
    current_step: str = "",
    proposed_action: str = "",
    tool_call: Optional[dict] = None,
    tool_result: Optional[dict] = None,
    phase: str = "unknown",
    verified_facts: Optional[dict] = None,
) -> dict:
    """
    审核主 agent 的单步执行计划/结果，返回严格结构化判定 JSON（dict）。

    输入字段建议：
    - user_goal: 用户最终目标
    - problem_type: 任务类型
    - matrix_info: 矩阵信息（shape/sparse/symmetric/positive_definite/rank/condition_number 等）
    - current_step: 当前步骤说明
    - proposed_action: 主 agent 计划动作
    - tool_call: 工具调用（建议包含 name/args）
    - tool_result: 工具返回结果（若已有）
    - phase: 当前流程阶段（clarify_goal/preconditions/property_check/plan/solve）
    - verified_facts: 已确认事实（建议 schema v1）
      {
        "schema_version": "v1",
        "goal_clarified": bool,
        "preference_confirmed": bool,
        "preconditions_checked": bool,
        "properties_checked": bool,
        "algorithm_planned": bool,
        "completed_phases": [phase...]
      }
    """
    issues: list[dict] = []
    goal = _to_text(user_goal)
    ptype = _to_text(problem_type)
    step = _to_text(current_step)
    action = _to_text(proposed_action)
    m_info = matrix_info if isinstance(matrix_info, dict) else {}
    v_facts = _normalize_verified_facts(verified_facts)
    merged_info = {**m_info, **v_facts}
    phase_norm = _normalize_phase(phase)

    _check_input_completeness(issues, goal, ptype, step, action)
    context_text = _collect_context_text(goal, step, action, tool_call)
    _check_phase_sequence(issues, phase_norm, merged_info, tool_call)
    _check_safety(issues, context_text)
    if _phase_allows(phase_norm, "task_alignment"):
        _check_task_alignment(issues, goal, ptype)
    if _phase_allows(phase_norm, "algorithm_stability"):
        _check_algorithm_and_stability(issues, ptype, merged_info, context_text, tool_call)
    if _phase_allows(phase_norm, "math_simplification"):
        _check_math_simplification_first(issues, ptype, step, action, tool_call)
    if _phase_allows(phase_norm, "interactive_gates"):
        _check_interactive_coach_gates(issues, ptype, merged_info, v_facts, step, action, tool_call)
    if _phase_allows(phase_norm, "tool_args"):
        _check_tool_arguments(issues, ptype, merged_info, tool_call)
    if _phase_allows(phase_norm, "tool_result"):
        _check_tool_result(issues, tool_result, merged_info)

    issues = _resolve_conflicts(issues, merged_info)
    issues = _rank_and_trim_issues(issues, top_k=3)

    # 信息不足但并非明显错误时，要求补充信息，避免猜测。
    if (
        not any(i["severity"] == "high" for i in issues)
        and (not goal or not ptype or not action)
    ):
        issues.append(
            _new_issue(
                "format",
                "medium",
                "关键信息不足，无法可靠完成步骤审查。",
                "补充 user_goal/problem_type/proposed_action 后重试。",
            )
        )
        result = _finalize(issues)
        result["next_action"] = "ask_user"
        result["short_reason"] = "信息不足，需补充关键上下文后再继续。"
        return result

    result = _finalize(issues)
    if (
        result.get("verdict") != "reject"
        and any(i.get("type") == "task_understanding" for i in issues)
    ):
        result["next_action"] = "ask_user"
        result["short_reason"] = "继续执行前需先向用户补充确认关键信息。"
    return result

